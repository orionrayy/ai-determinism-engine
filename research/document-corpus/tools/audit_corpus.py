#!/usr/bin/env python3
"""Metadata-only structural audit for public document reference artifacts."""
from __future__ import annotations
import argparse, csv, hashlib, html, json, math, re, statistics, time, urllib.error, urllib.parse, urllib.request, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
MAX_BYTES = 25 * 1024 * 1024
UA = "DocumentCorpusResearchBot/1.0 (public-document-structure-research; metadata-only)"
GUIDE = re.compile(r"\b(template|blank form|format|contoh laporan|panduan|pedoman|manual|guide|leaflet|cara mengunduh|petunjuk|how to write|sample template|formulir kosong)\b", re.I)
TRUSTED = {"institutional_repository_candidate", "institutional_or_government_candidate", "institutional_or_professional_candidate"}
WORDS = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ0-9]+)*", re.UNICODE)
SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9“\"(])")
NUMBERED = re.compile(r"^\s*(?:BAB\s+[IVXLC]+|CHAPTER\s+[IVXLC]+|[0-9]+(?:\.[0-9]+){0,4}\.?|[A-Z]\.)\s+\S+", re.I)
SECTIONS = {
 "cover_or_title_page": r"\b(halaman sampul|halaman judul|cover page|title page)\b",
 "approval_or_signoff": r"\b(lembar pengesahan|lembar persetujuan|approval page|approved by|signature|pengesahan)\b",
 "table_of_contents": r"\b(daftar isi|table of contents|contents)\b",
 "abstract_or_summary": r"\b(abstrak|abstract|ringkasan eksekutif|executive summary|summary)\b",
 "introduction_or_background": r"\b(pendahuluan|latar belakang|introduction|background|chapter i\b|bab i\b)\b",
 "methods_or_process": r"\b(metode penelitian|metodologi|metode pelaksanaan|methods?|methodology|prosedur|procedure|workflow)\b",
 "results_or_activities": r"\b(hasil penelitian|hasil kegiatan|hasil dan pembahasan|results?|activities|pelaksanaan kegiatan|implementasi)\b",
 "discussion_or_analysis": r"\b(pembahasan|analisis dan pembahasan|discussion|analysis)\b",
 "conclusion_or_recommendations": r"\b(kesimpulan|simpulan|saran|rekomendasi|conclusions?|recommendations?)\b",
 "references": r"\b(daftar pustaka|bibliography|references|referensi)\b",
 "appendices": r"\b(lampiran|appendices|appendix|attachments?)\b",
 "organization_profile": r"\b(gambaran umum|profil perusahaan|profil instansi|company profile|organizational structure)\b",
 "nursing_assessment": r"\b(pengkajian|diagnosa keperawatan|diagnosis keperawatan|intervensi keperawatan|implementasi keperawatan|evaluasi keperawatan|nursing assessment)\b",
 "internship_activities": r"\b(tempat pkl|pelaksanaan pkl|kegiatan pkl|kegiatan magang|internship procedure|job description|uraian tugas)\b",
 "financial_statements": r"\b(balance sheet|income statement|financial position|arus kas|laba rugi|posisi keuangan|realisasi anggaran|budget realization)\b",
}
STOCK = {
 "selain_itu": re.compile(r"\bselain itu\b", re.I),
 "di_sisi_lain": re.compile(r"\bdi sisi lain\b", re.I),
 "dengan_demikian": re.compile(r"\bdengan demikian\b", re.I),
 "oleh_karena_itu": re.compile(r"\boleh karena itu\b", re.I),
 "in_conclusion": re.compile(r"\bin conclusion\b", re.I),
 "furthermore": re.compile(r"\bfurthermore\b", re.I),
 "moreover": re.compile(r"\bmoreover\b", re.I),
 "not_only_but_also": re.compile(r"\btidak hanya\b.{0,100}\btetapi juga\b|\bnot only\b.{0,100}\bbut also\b", re.I | re.S),
}
AI_CUES = {
 "contrast_bukan_melainkan": re.compile(r"\bbukan\b.{0,100}\bmelainkan\b", re.I | re.S),
 "contrast_bukan_tetapi": re.compile(r"\bbukan\b.{0,100}\btetapi\b", re.I | re.S),
 "contrast_not_but": re.compile(r"\bnot\b.{0,100}\bbut\b", re.I | re.S),
 "digital_era_cliche": re.compile(r"\bera digital yang semakin berkembang\b|\bdi era digital saat ini\b", re.I),
 "rapid_development_cliche": re.compile(r"\bsemakin berkembang pesat\b|\bperkembangan pesat\b", re.I),
 "various_aspects": re.compile(r"\bberbagai aspek\b|\bvarious aspects\b", re.I),
}

def now():
 return datetime.now(timezone.utc).isoformat()

def sid(url):
 return hashlib.sha256(url.encode("utf-8","ignore")).hexdigest()[:16]

def normurl(url):
 p=urllib.parse.urlsplit(url.strip())
 return urllib.parse.urlunsplit((p.scheme.lower(),p.netloc.lower(),p.path.rstrip("/"),p.query,""))

def load_rows(root):
 rows={}
 for path in sorted(root.glob("discovery_batch_*.jsonl")):
  for line in path.read_text(encoding="utf-8").splitlines():
   if not line.strip(): continue
   try: r=json.loads(line)
   except Exception: continue
   url=r.get("url","")
   if not url.startswith(("http://","https://")): continue
   key=normurl(url)
   if key not in rows:
    r["source_batch"]=path.name; rows[key]=r
   else:
    prev=rows[key]
    prev["discovery_queries"]=sorted(set(prev.get("discovery_queries",[prev.get("discovery_query","")]))|{r.get("discovery_query","")})
 return list(rows.values())

def eligible(r, secondary=False):
 title=str(r.get("title","")); url=str(r.get("url",""))
 if GUIDE.search(title+" "+urllib.parse.urlsplit(url).path): return False
 prov=r.get("provenance_candidate","")
 intent=bool(r.get("title_matches_document_intent"))
 direct=r.get("artifact_signal")=="direct_file_link"
 if prov in TRUSTED and (direct or intent): return True
 return secondary and prov=="primary_source_review_needed" and intent

def score(r):
 s=0
 if r.get("provenance_candidate") in TRUSTED: s+=30
 if r.get("artifact_signal")=="direct_file_link": s+=24
 if r.get("title_matches_document_intent"): s+=12
 if r.get("published") not in (None,"","N/A"): s+=4
 if r.get("author") not in (None,"","N/A"): s+=3
 if r.get("format")=="spreadsheet": s+=8
 if r.get("likely_guide_or_template"): s-=100
 return s

def interleave(rows):
 buckets=defaultdict(list)
 for r in rows: buckets[str(r.get("genre","unknown"))].append(r)
 for b in buckets.values(): b.sort(key=score,reverse=True)
 out=[]; genres=sorted(buckets)
 while any(buckets[g] for g in genres):
  for g in genres:
   if buckets[g]: out.append(buckets[g].pop(0))
 return out

class HTMLDoc(HTMLParser):
 def __init__(self):
  super().__init__(convert_charrefs=True); self.title=[]; self.headings=[]; self.text=[]; self.links=[]
  self.in_title=False; self.in_heading=None; self.anchor=None; self.skip=0
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag in ("script","style","noscript","svg","nav","footer","header"): self.skip+=1
  if tag=="title": self.in_title=True
  if re.fullmatch(r"h[1-6]",tag): self.in_heading=tag
  if tag=="a" and a.get("href"): self.anchor={"href":a["href"],"text":""}
  if tag in ("br","p","div","li","tr","section"): self.text.append("\n")
 def handle_endtag(self,tag):
  if tag in ("script","style","noscript","svg","nav","footer","header") and self.skip: self.skip-=1
  if tag=="title": self.in_title=False
  if re.fullmatch(r"h[1-6]",tag): self.in_heading=None
  if tag=="a" and self.anchor: self.links.append(self.anchor); self.anchor=None
  if tag in ("p","div","li","tr","section"): self.text.append("\n")
 def handle_data(self,data):
  if self.skip:return
  s=data.strip()
  if not s:return
  if self.in_title:self.title.append(s)
  elif self.in_heading:self.headings.append((self.in_heading,s[:160]))
  self.text.append(s)
  if self.anchor is not None:self.anchor["text"]+=" "+s

def fetch(url):
 req=urllib.request.Request(url,headers={"User-Agent":UA,"Accept":"application/pdf,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.openxmlformats-officedocument.wordprocessingml.document,text/html,*/*"})
 with urllib.request.urlopen(req,timeout=25) as res:
  data=res.read(MAX_BYTES+1)
  if len(data)>MAX_BYTES:raise ValueError("file_too_large")
  return data,res.geturl(),res.headers.get("Content-Type","").split(";")[0].lower().strip(),getattr(res,"status",200)

def percentile(xs,q):
 if not xs:return 0
 return round(sorted(xs)[min(len(xs)-1,max(0,math.ceil(q*len(xs))-1))],2)

def text_metrics(text):
 words=WORDS.findall(text or ""); low=[w.lower() for w in words]
 lines=[re.sub(r"\s+"," ",x).strip() for x in (text or "").splitlines() if x.strip()]
 paras=[p.strip() for p in re.split(r"\n\s*\n+",text or "") if len(p.split())>=3]
 sentences=[s for s in SENTENCES.split(text or "") if len(WORDS.findall(s))>=3]
 pw=[len(WORDS.findall(p)) for p in paras]; sw=[len(WORDS.findall(s)) for s in sentences]
 grams=Counter(tuple(low[i:i+4]) for i in range(max(0,len(low)-3)))
 repeated=sum(v-1 for v in grams.values() if v>1)/max(1,len(low)-3)
 letters=lambda s: re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]","",s)
 return {
  "word_count":len(words),"character_count":len(text or ""),"nonempty_line_count":len(lines),
  "paragraph_count":len(paras),"sentence_count_estimate":len(sentences),
  "mean_words_per_paragraph":round(statistics.mean(pw),2) if pw else 0,
  "median_words_per_paragraph":round(statistics.median(pw),2) if pw else 0,
  "p90_words_per_paragraph":percentile(pw,.9),
  "mean_words_per_sentence":round(statistics.mean(sw),2) if sw else 0,
  "median_words_per_sentence":round(statistics.median(sw),2) if sw else 0,
  "sentence_length_stdev":round(statistics.pstdev(sw),2) if len(sw)>1 else 0,
  "type_token_ratio":round(len(set(low))/max(1,len(low)),4),
  "repeated_4gram_ratio":round(repeated,4),
  "colon_count":(text or "").count(":"),"semicolon_count":(text or "").count(";"),
  "em_dash_count":(text or "").count("—")+(text or "").count("–"),
  "contrast_frame_counts":{k:len(v.findall(text or "")) for k,v in AI_CUES.items() if v.search(text or "")},
  "stock_transition_counts":{k:len(v.findall(text or "")) for k,v in STOCK.items() if v.search(text or "")},
  "numbered_heading_count":sum(1 for line in lines if NUMBERED.match(line)),
  "uppercase_line_count":sum(1 for line in lines if len(line)<=140 and len(letters(line))>4 and letters(line).isupper()),
  "table_like_line_count":sum(1 for line in lines if line.count("|")>=2 or len(re.split(r"\s{2,}",line))>=4),
  "section_presence":{k:bool(re.search(p,text or "",re.I)) for k,p in SECTIONS.items()},
 }

def parse_pdf(data):
 import fitz
 doc=fitz.open(stream=data,filetype="pdf"); texts=[]; counts=[]; sizes=[]; lands=imgs=0
 for page in doc:
  t=page.get_text("text") or ""; texts.append(t); counts.append(len(WORDS.findall(t)))
  if page.rect.width>page.rect.height:lands+=1
  try:
   for block in page.get_text("dict").get("blocks",[]):
    if block.get("type")==1:imgs+=1
    for line in block.get("lines",[]):
     for span in line.get("spans",[]):
      size=span.get("size")
      if size and 3<float(size)<80 and span.get("text","").strip():sizes.append(round(float(size),1))
  except Exception: pass
 meta={"page_count":len(doc),"pages_with_text":sum(x>=10 for x in counts),"pages_with_low_text":sum(x<10 for x in counts),
  "page_words_min":min(counts or [0]),"page_words_median":statistics.median(counts) if counts else 0,"page_words_max":max(counts or [0]),
  "landscape_page_count":lands,"portrait_page_count":len(doc)-lands,"image_blocks_count":imgs,
  "dominant_font_sizes":[{"pt":k,"n":v} for k,v in Counter(sizes).most_common(8)],
  "pdf_metadata_present":bool(doc.metadata and any(doc.metadata.values())),"pdf_encrypted":bool(doc.is_encrypted)}
 doc.close();return "\n\n".join(texts),meta

def parse_docx(data):
 with zipfile.ZipFile(BytesIO(data)) as z:
  root=ET.fromstring(z.read("word/document.xml")); ns={"w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
  paras=[]; hs=Counter()
  for p in root.findall(".//w:p",ns):
   txt="".join((x.text or "") for x in p.findall(".//w:t",ns)).strip()
   if txt:paras.append(txt)
   st=p.find("./w:pPr/w:pStyle",ns)
   if st is not None:
    val=st.attrib.get("{%s}val"%ns["w"],"")
    if re.search("heading|judul|title",val,re.I):hs[val]+=1
  tables=root.findall(".//w:tbl",ns); dims=[]
  for t in tables:
   rows=t.findall("./w:tr",ns)
   dims.append({"rows":len(rows),"max_cells":max([len(r.findall("./w:tc",ns)) for r in rows] or [0])})
  sections=len(root.findall(".//w:sectPr",ns))
 return "\n\n".join(paras),{"paragraph_count_docx":len(paras),"table_count":len(tables),"table_dimensions":dims[:50],"section_count":sections,"heading_style_counts":dict(hs)}

def parse_xlsx(data):
 import openpyxl
 wb=openpyxl.load_workbook(BytesIO(data),read_only=False,data_only=False); sheets=[]; all_text=[]
 for i,ws in enumerate(wb.worksheets,1):
  nonempty=formulas=dates=0; fmts=Counter()
  for row in ws.iter_rows():
   for c in row:
    if c.value is not None:
     nonempty+=1; all_text.append(str(c.value)[:200])
     if isinstance(c.value,str) and c.value.startswith("="):formulas+=1
     if c.is_date:dates+=1
     fmts[c.number_format]+=1
  try: cfr=sum(len(v) for v in ws.conditional_formatting._cf_rules.values())
  except Exception:cfr=0
  sheets.append({"sheet_index":i,"max_row":ws.max_row,"max_column":ws.max_column,"nonempty_cells":nonempty,"formula_cells":formulas,"date_cells":dates,
   "merged_range_count":len(list(ws.merged_cells.ranges)),"table_count":len(getattr(ws,"tables",{})),"chart_count":len(getattr(ws,"_charts",[])),
   "data_validation_count":len(getattr(getattr(ws,"data_validations",None),"dataValidation",[])),"conditional_format_rule_count":cfr,
   "freeze_panes_set":bool(ws.freeze_panes),"hidden":ws.sheet_state!="visible","distinct_number_formats":len(fmts)})
 props={"sheet_count":len(sheets),"sheet_metrics":sheets,"defined_name_count":len(wb.defined_names),"external_links_count":len(getattr(wb,"_external_links",[])),
  "formula_cells_total":sum(x["formula_cells"] for x in sheets),"nonempty_cells_total":sum(x["nonempty_cells"] for x in sheets),
  "charts_total":sum(x["chart_count"] for x in sheets),"tables_total":sum(x["table_count"] for x in sheets)}
 wb.close();return "\n".join(all_text),props

def parse_html_doc(data,ctype,url):
 p=HTMLDoc();p.feed(data.decode("utf-8","replace"))
 links=[]
 for l in p.links:
  u=urllib.parse.urljoin(url,l["href"])
  if u.startswith(("http://","https://")):links.append({"url":u,"text":re.sub(r"\s+"," ",l["text"]).strip()[:180]})
 return "\n".join(p.text),{"html_title_present":bool(p.title),"html_heading_count":len(p.headings),"html_heading_levels":dict(Counter(x[0] for x in p.headings)),"html_link_count":len(links)},links

def choose_link(links):
 candidates=[]
 file_re=re.compile(r"\.(pdf|docx?|xlsx?|odt|ods)(?:$|[?#])",re.I)
 for l in links:
  u,t=l["url"],l["text"]; hay=u+" "+t
  if not file_re.search(u) and not re.search(r"\b(download|unduh|full text|fulltext|pdf|dokumen)\b",t,re.I):continue
  if GUIDE.search(hay):continue
  score=(12 if file_re.search(u) else 0)+(6 if re.search(r"\b(download|unduh|full text|fulltext|pdf|dokumen)\b",t,re.I) else 0)
  if re.search(r"\b(laporan|skripsi|thesis|makalah|report|karya ilmiah|asuhan keperawatan|xlsx|excel)\b",hay,re.I):score+=5
  if re.search(r"\b(login|sign in|preview|citation only)\b",hay,re.I):score-=20
  candidates.append((score,u))
 candidates.sort(reverse=True);return candidates[0][1] if candidates and candidates[0][0]>0 else None

def analyse(row,delay):
 url=row["url"];result={"corpus_id":sid(normurl(url)),"source_url":url,"source_domain":row.get("host") or (urllib.parse.urlsplit(url).hostname or "").lower(),
  "genre":row.get("genre","unknown"),"title_hash":hashlib.sha256(str(row.get("title","")).encode("utf-8","ignore")).hexdigest()[:12],
  "source_batch":row.get("source_batch",""),"source_provenance_candidate":row.get("provenance_candidate",""),"validation_timestamp_utc":now(),
  "artifact_url":None,"status":"pending","http_status":None,"content_type":None,"bytes_downloaded":0,"sha256":None,
  "source_authenticity_note":"Automated public-access/structure check only; human source-record provenance review remains necessary."}
 try:
  data,final,ctype,status=fetch(url)
  result.update({"artifact_url":final,"http_status":status,"content_type":ctype,"bytes_downloaded":len(data),"sha256":hashlib.sha256(data).hexdigest()})
  time.sleep(delay)
  if data.startswith(b"%PDF-") or ctype=="application/pdf" or urllib.parse.urlsplit(final).path.lower().endswith(".pdf"):
   text,fmt=parse_pdf(data);kind="pdf"
  elif zipfile.is_zipfile(BytesIO(data)):
   with zipfile.ZipFile(BytesIO(data)) as z:names=set(z.namelist())
   if "xl/workbook.xml" in names:text,fmt=parse_xlsx(data);kind="xlsx"
   elif "word/document.xml" in names:text,fmt=parse_docx(data);kind="docx"
   else:result.update({"status":"unsupported_archive","error_category":"unsupported_zip"});return result
  elif "html" in ctype or data.lstrip().lower().startswith((b"<!doctype html",b"<html")) or ctype in ("","text/plain"):
   text,fmt,links=parse_html_doc(data,ctype,final)
   target=choose_link(links)
   record_path=bool(re.search(r"/(?:id/eprint|eprint|handle|record|items?|repository|view|details)(?:/|$)",urllib.parse.urlsplit(final).path,re.I))
   if target and target.rstrip("/")!=url.rstrip("/"):
    data2,final2,ctype2,status2=fetch(target)
    result.update({"artifact_url":final2,"http_status":status2,"content_type":ctype2,"bytes_downloaded":result["bytes_downloaded"]+len(data2),"sha256":hashlib.sha256(data2).hexdigest()})
    time.sleep(delay)
    if data2.startswith(b"%PDF-") or ctype2=="application/pdf" or urllib.parse.urlsplit(final2).path.lower().endswith(".pdf"):
     text,fmt=parse_pdf(data2);kind="pdf_from_record"
    elif zipfile.is_zipfile(BytesIO(data2)):
     with zipfile.ZipFile(BytesIO(data2)) as z:names=set(z.namelist())
     if "xl/workbook.xml" in names:text,fmt=parse_xlsx(data2);kind="xlsx_from_record"
     elif "word/document.xml" in names:text,fmt=parse_docx(data2);kind="docx_from_record"
     else:result.update({"status":"unsupported_archive","error_category":"unsupported_attachment_archive"});return result
    else:
     text,fmt,links2=parse_html_doc(data2,ctype2,final2);kind="html"
     if record_path:result.update({"status":"metadata_landing_page","error_category":"no_full_file_link"});return result
   else:
    kind="html"
    if record_path:result.update({"status":"metadata_landing_page","error_category":"no_full_file_link"});return result
  else:result.update({"status":"unsupported_file_type","error_category":"unsupported_file_type"});return result
  tm=text_metrics(text);result["document_type"]=kind;result["format_metrics"]=fmt;result["text_metrics"]=tm;result["document_words"]=tm["word_count"]
  if kind.startswith("xlsx"):enough=fmt.get("sheet_count",0)>0 and fmt.get("nonempty_cells_total",0)>=50
  else:enough=tm["word_count"]>=300 and (fmt.get("html_heading_count",2)>=2 if kind=="html" else True)
  result["status"]="analyzed_full_document" if enough else "insufficient_extracted_content"
  result["analysis_depth"]="full_text_structural_and_style_metrics" if enough else "partial_scanned_or_metadata_only"
 except urllib.error.HTTPError as e:result.update({"status":"fetch_http_error","http_status":e.code,"error_category":"http_error"})
 except urllib.error.URLError as e:result.update({"status":"fetch_network_error","error_category":"network_error","error_message":str(getattr(e,"reason",e))[:180]})
 except Exception as e:result.update({"status":"fetch_or_parse_error","error_category":type(e).__name__,"error_message":str(e)[:180]})
 return result

def report_for(rows,candidates,target):
 ok=[x for x in rows if x.get("status")=="analyzed_full_document"]; grouped=defaultdict(list)
 for x in ok:grouped[x.get("genre","unknown")].append(x)
 genre_stats={}
 for genre,docs in sorted(grouped.items()):
  t=[x.get("text_metrics",{}) for x in docs];f=[x.get("format_metrics",{}) for x in docs]; sec={}
  for name in SECTIONS:sec[name]=round(sum(bool(m.get("section_presence",{}).get(name)) for m in t)/len(docs),3)
  stock=Counter();cues=Counter()
  for m in t:stock.update(m.get("stock_transition_counts",{}));cues.update(m.get("contrast_frame_counts",{}))
  genre_stats[genre]={"count":len(docs),"formats":dict(Counter(x.get("document_type","unknown") for x in docs)),
   "median_word_count":round(statistics.median(m.get("word_count",0) for m in t),1),
   "median_page_or_sheet_count":round(statistics.median(m.get("page_count",m.get("sheet_count",0)) for m in f),1),
   "median_paragraph_words":round(statistics.median(m.get("median_words_per_paragraph",0) for m in t),1),
   "median_sentence_words":round(statistics.median(m.get("median_words_per_sentence",0) for m in t),1),
   "median_colons":round(statistics.median(m.get("colon_count",0) for m in t),1),
   "median_contrast_frames":round(statistics.median(sum(m.get("contrast_frame_counts",{}).values()) for m in t),1),
   "section_presence_rates":sec,"stock_transition_counts":dict(stock),"contrast_cue_counts":dict(cues)}
 status=Counter(x.get("status","unknown") for x in rows)
 summary={"generated_at_utc":now(),"target_full_documents":target,"candidate_records_discovered":candidates,
  "candidate_records_attempted":len(rows),"full_documents_analyzed":len(ok),"target_reached":len(ok)>=target,"status_counts":dict(status),
  "genres_analyzed":len(genre_stats),"genre_analysis":genre_stats,
  "limitations":["URL access and extracted structure do not prove all pages were written by a human; manually inspect provenance and author metadata.",
   "Phrase metrics are contextual review cues, not an AI-authorship detector and not evidence of AI authorship.",
   "OCR is not run; scanned PDFs may be marked partial. PDF table and visual layout analysis are approximations.",
   "Stored outputs contain metadata and structural measurements, not the source document body text."]}
 lines=["# Original Document Corpus — Audit Report","",f"Generated UTC: {summary['generated_at_utc']}","",
 f"- Candidate records discovered: **{candidates}**","- Records attempted: **%d**"%len(rows),
 f"- Full documents structurally analyzed: **{len(ok)} / {target} target**",
 f"- Acceptance target reached: **{'YES' if summary['target_reached'] else 'NO'}**",
 "- Status distribution: "+json.dumps(dict(status),ensure_ascii=False,sort_keys=True),"",
 "A document is counted only when a public file or substantive authored HTML artifact was retrieved and enough content extracted. Landing pages alone do not count.","",
 "## Results by genre","","| Genre | n | Median words | Median pages/sheets | Median paragraph words | Median sentence words |","|---|---:|---:|---:|---:|---:|"]
 for g,v in genre_stats.items():
  lines.append(f"| {g} | {v['count']} | {v['median_word_count']} | {v['median_page_or_sheet_count']} | {v['median_paragraph_words']} | {v['median_sentence_words']} |")
 lines+=["","## Section conventions by genre",""]
 for g,v in genre_stats.items():
  top=sorted(v["section_presence_rates"].items(),key=lambda p:p[1],reverse=True)[:8]
  lines += [f"### {g} (n={v['count']})","", "; ".join(f"{k.replace('_',' ')}: {v:.0%}" for k,v in top)]
  if v["stock_transition_counts"]:lines += ["","Observed transition-token counts (not normalized by length): "+", ".join(f"{k}={n}" for k,n in sorted(v["stock_transition_counts"].items()))]
  if v["contrast_cue_counts"]:lines += ["","Contrast-pattern counts (not normalized by length; do not treat as universally bad): "+", ".join(f"{k}={n}" for k,n in sorted(v["contrast_cue_counts"].items()))]
  lines.append("")
 lines += ["## Reading the indicators","",
 "- Compare examples within the same genre, purpose and institutional template. Do not force a universal structure on case reports, internships, theses, operational documents and spreadsheets.",
 "- Section frequency is a convention to investigate, not a rule to impose on every document. User rubrics and source templates remain controlling.",
 "- Phrase counts are descriptive; ordinary use may be contextually appropriate. No inference of AI authorship is made.","","## Limitations"]+["- "+x for x in summary["limitations"]]
 return summary,"\n".join(lines)+"\n"

def write_jsonl(path,rows):
 with path.open("w",encoding="utf-8") as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")

def main():
 p=argparse.ArgumentParser();p.add_argument("--input-dir",type=Path,default=ROOT);p.add_argument("--output-dir",type=Path,default=OUT)
 p.add_argument("--target",type=int,default=500);p.add_argument("--max-attempts",type=int,default=850);p.add_argument("--delay",type=float,default=.15);p.add_argument("--self-test",action="store_true")
 a=p.parse_args()
 if a.self_test:
  x=text_metrics("HALAMAN JUDUL\n\nABSTRAK\n\nBAB I PENDAHULUAN\n\nMetode penelitian. Hasil penelitian. Kesimpulan.\n\nDaftar Pustaka")
  assert x["word_count"]>5 and x["section_presence"]["cover_or_title_page"] and x["section_presence"]["abstract_or_summary"] and x["section_presence"]["references"]
  print("self-test passed");return 0
 a.output_dir.mkdir(parents=True,exist_ok=True);all_rows=load_rows(a.input_dir)
 primary=[r for r in all_rows if eligible(r,False)];secondary=[r for r in all_rows if eligible(r,True) and r not in primary]
 candidates=(interleave(primary)+interleave(secondary))[:a.max_attempts];results=[];seen=set()
 for offset in range(0,len(candidates),6):
  batch=[]
  for i,row in enumerate(candidates[offset:offset+6],start=offset+1):
   key=sid(normurl(row["url"]))
   if key not in seen:seen.add(key);batch.append((i,row))
  with ThreadPoolExecutor(max_workers=6) as ex:
   pending={ex.submit(analyse,row,a.delay):(i,row) for i,row in batch}
   for future in as_completed(pending):
    i,row=pending[future];r=future.result();r["sequence"]=i;results.append(r)
  full=sum(z.get("status")=="analyzed_full_document" for z in results)
  print(f"progress attempted={len(results)} fully_analyzed={full} target={a.target} workers=6",flush=True)
  if full>=a.target:break
 results.sort(key=lambda x:x.get("sequence",0))
 successful=[r for r in results if r.get("status")=="analyzed_full_document"]
 summary,report=report_for(results,len(all_rows),a.target)
 write_jsonl(a.output_dir/"all_attempts.jsonl",results);write_jsonl(a.output_dir/"full_documents.jsonl",successful)
 write_jsonl(a.output_dir/"failed_or_partial.jsonl",[r for r in results if r.get("status")!="analyzed_full_document"])
 (a.output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 (a.output_dir/"report.md").write_text(report,encoding="utf-8")
 with (a.output_dir/"candidate_inventory.csv").open("w",newline="",encoding="utf-8") as f:
  fields=["corpus_id","title","url","genre","host","published","author","format","provenance_candidate","artifact_signal","likely_guide_or_template","validation_status","detailed_analysis_status"]
  w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader()
  for row in all_rows:
   x=dict(row);x["corpus_id"]=sid(normurl(x.get("url","")));w.writerow(x)
 print(json.dumps({"candidate_records":len(all_rows),"attempted":len(results),"full_documents_analyzed":len(successful),"target":a.target,"target_reached":summary["target_reached"],"status_counts":summary["status_counts"]},ensure_ascii=False))
 return 0
if __name__=="__main__":raise SystemExit(main())
