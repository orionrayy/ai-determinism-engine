#!/usr/bin/env python3
"""Deterministic, side-effect-free workload blueprint compiler."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import hashlib
import json
import re

SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 256 * 1024
MAX_SECTIONS = 256
MAX_REQUIREMENTS = 512
MAX_WORKSTREAMS = 128
MAX_UNITS = 64
MAX_REQUIREMENTS_PER_UNIT = 8
MAX_UNIT_CONTEXT_BYTES = 24 * 1024
MAX_MANIFEST_BYTES = 480 * 1024
MAX_ID_LENGTH = 100
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_KINDS = {"hard_constraint","design_intent","implementation","acceptance","optional"}
_PRIORITY = {"critical":0,"high":1,"medium":2,"low":3}

class BlueprintError(ValueError):
    pass

def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()

def _id(label: str, value: Any) -> str:
    value = str(value or "").strip()
    if not _SAFE_ID.fullmatch(value):
        raise BlueprintError(f"{label} must match [A-Za-z0-9._-] and be <= {MAX_ID_LENGTH} chars")
    return value

def parse_source_document(source: str) -> dict[str, Any]:
    source = str(source or "")
    raw = source.encode("utf-8")
    if len(raw) > MAX_SOURCE_BYTES:
        raise BlueprintError("source exceeds 256 KiB")
    lines = source.splitlines()
    heading = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
    starts = []
    for lineno, line in enumerate(lines, 1):
        match = heading.match(line)
        if match:
            starts.append((lineno, len(match.group(1)), match.group(2).strip()))
    sections = []
    if not starts and source.strip():
        body = source.strip()
        sections.append({"section_id":"section-001","heading":"DOCUMENT","level":1,"line_start":1,"line_end":max(1,len(lines)),"text_bytes":len(body.encode("utf-8")),"digest":hashlib.sha256(body.encode("utf-8")).hexdigest()})
    else:
        for i,(line_start,level,title) in enumerate(starts):
            line_end = starts[i+1][0]-1 if i+1 < len(starts) else len(lines)
            body = "\n".join(lines[line_start:line_end]).strip()
            sections.append({"section_id":f"section-{i+1:03d}","heading":title[:200],"level":level,"line_start":line_start,"line_end":line_end,"text_bytes":len(body.encode("utf-8")),"digest":hashlib.sha256(body.encode("utf-8")).hexdigest()})
    if len(sections) > MAX_SECTIONS:
        raise BlueprintError(f"source contains more than {MAX_SECTIONS} sections")
    return {"schema_version":SCHEMA_VERSION,"source_kind":"markdown-or-text","source_digest":hashlib.sha256(raw).hexdigest(),"source_bytes":len(raw),"sections":sections}

def _list(value: Any, label: str, limit: int=256) -> list[Any]:
    if value is None: return []
    if isinstance(value,str): value=[value]
    if not isinstance(value,list): raise BlueprintError(f"{label} must be a list")
    if len(value)>limit: raise BlueprintError(f"{label} exceeds {limit} items")
    return value

def normalize_blueprint(blueprint: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(blueprint, Mapping): raise BlueprintError("blueprint must be an object")
    if len(canonical_json(blueprint)) > MAX_SOURCE_BYTES: raise BlueprintError("structured blueprint exceeds 256 KiB")
    raw_requirements=_list(blueprint.get("requirements"),"requirements",MAX_REQUIREMENTS)
    requirements=[]
    for index,raw in enumerate(raw_requirements,1):
        if not isinstance(raw,Mapping): raise BlueprintError("requirement entries must be objects")
        rid=_id("requirement.id",raw.get("id") or f"req-{index:03d}")
        summary=str(raw.get("summary") or raw.get("text") or "").strip()
        if not summary: raise BlueprintError(f"{rid}: summary is required")
        kind=str(raw.get("kind") or "implementation").strip()
        priority=str(raw.get("priority") or "medium").strip().lower()
        if kind not in _KINDS: raise BlueprintError(f"{rid}: invalid kind {kind}")
        if priority not in _PRIORITY: raise BlueprintError(f"{rid}: invalid priority {priority}")
        requirements.append({
            "id":rid,"summary":summary[:4000],"kind":kind,"priority":priority,
            "category":str(raw.get("category") or "general")[:100],
            "workstream":str(raw.get("workstream") or "default")[:100],
            "depends_on":[str(x).strip() for x in _list(raw.get("depends_on"),f"{rid}.depends_on",64) if str(x).strip()],
            "acceptance_criteria":[str(x).strip()[:1000] for x in _list(raw.get("acceptance_criteria"),f"{rid}.acceptance_criteria",64) if str(x).strip()],
            "artifacts":[str(x).strip()[:300] for x in _list(raw.get("artifacts"),f"{rid}.artifacts",64) if str(x).strip()],
            "agent_roles":[str(x).strip()[:80] for x in _list(raw.get("agent_roles"),f"{rid}.agent_roles",16) if str(x).strip()],
            "capabilities":[str(x).strip()[:80] for x in _list(raw.get("capabilities"),f"{rid}.capabilities",16) if str(x).strip()],
            "source_refs":[str(x).strip()[:200] for x in _list(raw.get("source_refs"),f"{rid}.source_refs",64) if str(x).strip()],
            "risk":str(raw.get("risk") or "low").strip().lower(),
        })
    ids=[r["id"] for r in requirements]
    if len(ids)!=len(set(ids)): raise BlueprintError("duplicate requirement id")
    idset=set(ids)
    for r in requirements:
        unknown=[d for d in r["depends_on"] if d not in idset]
        if unknown: raise BlueprintError(f"{r['id']}: unknown dependencies {unknown}")
    raw_workstreams=_list(blueprint.get("workstreams"),"workstreams",MAX_WORKSTREAMS)
    workstreams=[]
    for index,raw in enumerate(raw_workstreams,1):
        if not isinstance(raw,Mapping): raise BlueprintError("workstream entries must be objects")
        wid=_id("workstream.id",raw.get("id") or f"workstream-{index:03d}")
        workstreams.append({"id":wid,"name":str(raw.get("name") or wid)[:200],"description":str(raw.get("description") or "")[:1000]})
    if not workstreams:
        names=sorted({r["workstream"] for r in requirements}) or ["default"]
        workstreams=[{"id":_id("workstream.id",n.replace(" ","-")[:100]),"name":n,"description":""} for n in names]
    constraints=[]
    for index,raw in enumerate(_list(blueprint.get("constraints"),"constraints",512),1):
        if isinstance(raw,Mapping):
            cid=_id("constraint.id",raw.get("id") or f"constraint-{index:03d}")
            text=str(raw.get("text") or raw.get("summary") or "").strip()
            kind=str(raw.get("kind") or "hard_constraint").strip()
        else:
            cid,text,kind=f"constraint-{index:03d}",str(raw).strip(),"hard_constraint"
        if text: constraints.append({"id":cid,"text":text[:2000],"kind":kind})
    normalized={"schema_version":SCHEMA_VERSION,"blueprint_id":_id("blueprint_id",blueprint.get("blueprint_id") or "blueprint"),"version":str(blueprint.get("version") or "1")[:80],"title":str(blueprint.get("title") or "")[:300],"source":dict(blueprint.get("source") or {}),"constraints":constraints,"workstreams":workstreams,"requirements":sorted(requirements,key=lambda x:(_PRIORITY[x["priority"]],x["id"])),"non_goals":[str(x)[:500] for x in _list(blueprint.get("non_goals"),"non_goals",128) if str(x).strip()],"policy":dict(blueprint.get("policy") or {})}
    normalized["blueprint_digest"]=digest(normalized)
    return normalized

def _levels(requirements: Sequence[Mapping[str,Any]]) -> dict[str,int]:
    by_id={str(r["id"]):r for r in requirements}
    result={}; visiting=set()
    def visit(rid:str)->int:
        if rid in result: return result[rid]
        if rid in visiting: raise BlueprintError(f"requirement dependency cycle at {rid}")
        visiting.add(rid)
        deps=by_id[rid].get("depends_on") or []
        result[rid]=0 if not deps else max(visit(d)+1 for d in deps)
        visiting.remove(rid)
        return result[rid]
    for rid in by_id: visit(rid)
    return result

def _unit_id(blueprint_id:str,ordinal:int,workstream:str)->str:
    return "unit-"+hashlib.sha256(f"{blueprint_id}:{ordinal}:{workstream}".encode("utf-8")).hexdigest()[:16]

def compile_execution_units(normalized: Mapping[str,Any], *, max_requirements_per_unit:int=MAX_REQUIREMENTS_PER_UNIT, max_units:int=MAX_UNITS)->list[dict[str,Any]]:
    if not 1<=max_requirements_per_unit<=MAX_REQUIREMENTS_PER_UNIT: raise BlueprintError("invalid max_requirements_per_unit")
    if not 1<=max_units<=MAX_UNITS: raise BlueprintError("invalid max_units")
    requirements=list(normalized.get("requirements") or [])
    if not requirements: return []
    levels=_levels(requirements)
    ordered=sorted(requirements,key=lambda r:(levels[r["id"]],str(r.get("workstream") or "default"),_PRIORITY[r["priority"]],r["id"]))
    groups=[]; current=[]; key=None
    for r in ordered:
        next_key=(levels[r["id"]],str(r.get("workstream") or "default"))
        if current and (next_key!=key or len(current)>=max_requirements_per_unit):
            groups.append(current); current=[]
        if not current: key=next_key
        current.append(r)
    if current: groups.append(current)
    if len(groups)>max_units: raise BlueprintError(f"compiled units exceed {max_units}")
    req_to_unit={}; units=[]
    for ordinal,group in enumerate(groups,1):
        streams=sorted({str(r.get("workstream") or "default") for r in group})
        uid=_unit_id(str(normalized["blueprint_id"]),ordinal,"+".join(streams))
        for r in group: req_to_unit[str(r["id"])]=uid
        risks={str(r.get("risk") or "low") for r in group}
        risk="critical" if "critical" in risks else "high" if "high" in risks else "medium" if "medium" in risks else "low"
        units.append({"unit_id":uid,"ordinal":ordinal,"workstreams":streams,"requirement_ids":[str(r["id"]) for r in group],"depends_on_units":[],"risk":risk,"agent_roles":sorted({x for r in group for x in r["agent_roles"]}),"capabilities":sorted({x for r in group for x in r["capabilities"]}),"acceptance_criteria":[x for r in group for x in r["acceptance_criteria"]],"artifacts":sorted({x for r in group for x in r["artifacts"]}),"source_refs":sorted({x for r in group for x in r["source_refs"]}),"requirements":[dict(r) for r in group]})
    by_id={u["unit_id"]:u for u in units}; req_by_id={str(r["id"]):r for r in requirements}
    for u in units:
        deps=set()
        for rid in u["requirement_ids"]:
            for dep in req_by_id[rid].get("depends_on",[]):
                du=req_to_unit[dep]
                if du!=u["unit_id"]: deps.add(du)
        u["depends_on_units"]=sorted(deps,key=lambda x:by_id[x]["ordinal"])
        u["parallel_candidate"]=not u["depends_on_units"]
        if len(canonical_json(u))>MAX_UNIT_CONTEXT_BYTES: raise BlueprintError(f"{u['unit_id']} exceeds 24 KiB execution-unit bound")
    return units

def validate_unit_dag(units:Sequence[Mapping[str,Any]],normalized:Mapping[str,Any])->dict[str,Any]:
    ids=[str(u.get("unit_id") or "") for u in units]
    if len(ids)!=len(set(ids)): raise BlueprintError("duplicate unit id")
    by_id={str(u["unit_id"]):u for u in units}
    expected={str(r["id"]) for r in normalized.get("requirements",[])}
    actual={str(rid) for u in units for rid in u.get("requirement_ids",[])}
    if actual!=expected: raise BlueprintError("execution units do not cover every requirement")
    visiting=set(); visited=set()
    def visit(uid:str)->None:
        if uid in visiting: raise BlueprintError(f"unit dependency cycle at {uid}")
        if uid in visited: return
        visiting.add(uid)
        for dep in [str(x) for x in (by_id[uid].get("depends_on_units") or [])]:
            if dep not in by_id: raise BlueprintError(f"{uid}: unknown unit dependency {dep}")
            if dep==uid: raise BlueprintError(f"{uid}: self dependency")
            visit(dep)
        visiting.remove(uid); visited.add(uid)
    for uid in by_id: visit(uid)
    return {"unit_count":len(units),"requirement_count":len(expected),"root_units":sorted(uid for uid in by_id if not by_id[uid].get("depends_on_units")),"leaf_units":sorted(uid for uid in by_id if not any(uid in (u.get("depends_on_units") or []) for u in units)),"parallel_candidate_units":sorted(uid for uid in by_id if by_id[uid].get("parallel_candidate") is True)}

def build_compilation_manifest(blueprint:Mapping[str,Any]|str,*,max_requirements_per_unit:int=MAX_REQUIREMENTS_PER_UNIT,max_units:int=MAX_UNITS)->dict[str,Any]:
    if isinstance(blueprint,str):
        parsed=parse_source_document(blueprint)
        blueprint={"blueprint_id":"source-document","version":"1","title":"Parsed Source Document","source":parsed,"requirements":[{"id":s["section_id"],"summary":s["heading"],"workstream":"source-sections","source_refs":[s["section_id"]],"acceptance_criteria":["Preserve source section provenance; perform semantic extraction separately."]} for s in parsed["sections"]]}
    normalized=normalize_blueprint(blueprint)
    units=compile_execution_units(normalized,max_requirements_per_unit=max_requirements_per_unit,max_units=max_units)
    manifest={"schema_version":SCHEMA_VERSION,"compiler":"deterministic-blueprint-compiler","blueprint":normalized,"units":units,"graph":validate_unit_dag(units,normalized)}
    manifest["manifest_digest"]=digest(manifest)
    if len(canonical_json(manifest))>MAX_MANIFEST_BYTES: raise BlueprintError("compiled manifest exceeds 480 KiB safety limit")
    return manifest

def render_execution_packet(manifest:Mapping[str,Any],unit_id:str,*,max_bytes:int=MAX_UNIT_CONTEXT_BYTES)->dict[str,Any]:
    unit=next((u for u in manifest.get("units",[]) if str(u.get("unit_id"))==str(unit_id)),None)
    if unit is None: raise BlueprintError(f"unknown unit {unit_id}")
    packet={"schema_version":SCHEMA_VERSION,"manifest_digest":str(manifest.get("manifest_digest") or ""),"blueprint_id":str((manifest.get("blueprint") or {}).get("blueprint_id") or ""),"blueprint_version":str((manifest.get("blueprint") or {}).get("version") or ""),"unit":unit,"execution_contract":{"read_previous_artifacts":True,"re_audit_previous_results":True,"preserve_traceability":True,"do_not_modify_control_plane_unless_explicit":True,"validate_before_terminal_success":True,"record_unknowns_instead_of_inventing":True}}
    if len(canonical_json(packet))>max_bytes: raise BlueprintError(f"execution packet exceeds {max_bytes} bytes")
    return packet

def load_blueprint_file(path:str,*,root:str)->str:
    from pathlib import Path
    base=Path(root).resolve()
    candidate=(base/Path(path)).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    try: candidate.relative_to(base)
    except ValueError as exc: raise BlueprintError("blueprint path escapes configured workload root") from exc
    if not candidate.is_file(): raise BlueprintError(f"blueprint file does not exist: {candidate}")
    raw=candidate.read_bytes()
    if len(raw)>MAX_SOURCE_BYTES: raise BlueprintError("blueprint file exceeds 256 KiB")
    return raw.decode("utf-8")
