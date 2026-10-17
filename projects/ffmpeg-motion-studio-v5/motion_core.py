"""Shared color, typography, easing, and procedural drawing primitives."""
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DURATION = 1920, 1080, 24, 41.75
BASE = Path(__file__).resolve().parent
FONT_REG = '/usr/share/fonts/truetype/inter/Inter-Regular.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/inter/Inter-Bold.ttf'
if not Path(FONT_REG).exists(): FONT_REG = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
if not Path(FONT_BOLD).exists(): FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

C = {
 'bg0':(12,22,30),'bg1':(17,41,49),'deep':(7,16,25),'panel':(17,43,53),'panel2':(24,56,64),
 'ink':(247,246,235),'muted':(191,202,202),'teal':(69,224,198),'gold':(255,195,94),
 'coral':(255,116,101),'ivory':(250,245,229),'paperline':(214,207,184),'darktext':(34,45,45),
 'skin':(229,164,119),'skinshadow':(187,118,83),'hair':(35,29,28),'hairlight':(63,47,40),
 'shirt':(27,115,119),'shirtlight':(43,154,148),'pants':(39,54,76),'shoe':(26,32,39),
 'white':(255,255,255),'green':(87,199,133),'line':(62,95,105)
}
BEATS = [0.0,2.45,6.35,11.0,17.75,22.90,30.45,37.25,DURATION]
CAPTIONS = [
 (0.30,1.95,'Perhatikan angka pada struk ini.'),
 (2.45,6.25,'Dua minuman berjumlah tiga puluh ribu rupiah.'),
 (6.35,10.95,'Tiga makanan ringan berjumlah dua puluh empat ribu.'),
 (11.00,17.65,'Jadi, sebelum diskon, subtotalnya lima puluh empat ribu rupiah.'),
 (17.75,22.85,'Diskon sepuluh persen berarti lima ribu empat ratus rupiah.'),
 (22.95,30.35,'Kurangi potongan itu dari subtotal: lima puluh empat ribu dikurangi lima ribu empat ratus.'),
 (30.45,37.20,'Total yang dibayar adalah empat puluh delapan ribu enam ratus rupiah.'),
 (37.25,41.68,'Ingat urutannya: subtotal, potongan, lalu total akhir.'),
]

def clamp(x, lo=0.0, hi=1.0): return max(lo,min(hi,x))
def smooth(x):
    x=clamp(x); return x*x*(3-2*x)
def out_cubic(x):
    x=clamp(x); return 1-(1-x)**3
def spring(dt, amp=1.0, decay=3.6, freq=10.5):
    if dt < 0: return 0.0
    return amp*math.exp(-decay*dt)*math.sin(freq*dt)
def fnt(size,bold=False): return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, max(10,int(size)))
def text(d,xy,s,size,color=None,bold=False,anchor='la',stroke=0):
    d.text(xy,str(s),font=fnt(size,bold),fill=color or C['ink'],anchor=anchor,stroke_width=stroke,stroke_fill=(0,0,0,180))
def tw(d,s,size,bold=False): return d.textbbox((0,0),str(s),font=fnt(size,bold))[2]
def rounded(d,box,r,fill,outline=None,width=2): d.rounded_rectangle(tuple(int(v) for v in box),radius=int(r),fill=fill,outline=outline,width=int(width))
def center(d,xy,s,size,color=None,bold=True): text(d,xy,s,size,color,bold,'mm')
def fit(d,s,maxw,start,bold=True,minimum=28):
    n=int(start)
    while n>minimum and tw(d,s,n,bold)>maxw: n-=2
    return n
def line(d,p1,p2,color,width): d.line((tuple(map(int,p1)),tuple(map(int,p2))),fill=color,width=max(1,int(width)))
def round_line(d,p1,p2,color,width):
    w=max(1,int(width)); d.line((tuple(map(int,p1)),tuple(map(int,p2))),fill=color,width=w)
    r=w/2
    for p in (p1,p2): d.ellipse((int(p[0]-r),int(p[1]-r),int(p[0]+r),int(p[1]+r)),fill=color)
def lerp(a,b,t): return a+(b-a)*t
def lpoint(a,b,t): return (lerp(a[0],b[0],t),lerp(a[1],b[1],t))
def rotate_points(points,cx,cy,ang):
    co,si=math.cos(ang),math.sin(ang); out=[]
    for x,y in points:
        dx,dy=x-cx,y-cy; out.append((cx+dx*co-dy*si,cy+dx*si+dy*co))
    return out

def draw_bottle(d,x,y,s=1.0):
    rounded(d,(x-28*s,y-47*s,x+28*s,y+51*s),13*s,(37,137,136),(69,224,198),max(2,3*s))
    rounded(d,(x-13*s,y-65*s,x+13*s,y-44*s),4*s,C['gold'])
    rounded(d,(x-22*s,y-10*s,x+22*s,y+19*s),6*s,C['ivory'])
    center(d,(x,y+4*s),'AIR',12*s,C['darktext'])
def draw_snack(d,x,y,s=1.0):
    pts=[(x-34*s,y-37*s),(x+34*s,y-37*s),(x+27*s,y+42*s),(x-27*s,y+42*s)]
    d.polygon(pts,fill=(206,132,61),outline=C['gold'],width=max(2,int(3*s)))
    line(d,(x-34*s,y-37*s),(x-18*s,y-53*s),C['gold'],3*s); line(d,(x-18*s,y-53*s),(x+18*s,y-53*s),C['gold'],3*s); line(d,(x+18*s,y-53*s),(x+34*s,y-37*s),C['gold'],3*s)
    for k in range(3): d.ellipse((int(x-16*s+k*14*s),int(y-4*s),int(x-8*s+k*14*s),int(y+4*s)),fill=(255,230,165))

def make_bg(digital=False):
    top=C['bg0'] if not digital else (8,21,34); bot=(33,48,45) if not digital else (14,43,54)
    im=Image.new('RGB',(W,H)); pix=im.load()
    for y in range(H):
        q=y/(H-1); co=tuple(int(lerp(top[i],bot[i],q)) for i in range(3))
        for x in range(W):
            ed=1-.12*abs((x-W/2)/(W/2))**1.5
            pix[x,y]=tuple(int(v*ed) for v in co)
    d=ImageDraw.Draw(im,'RGBA')
    if not digital:
        # Deep, low-contrast store silhouette: shelves and soft product blocks.
        for sy in (275,430,590):
            d.rectangle((0,sy,W,sy+8),fill=(120,164,160,18))
        for sx in range(55,W,180):
            d.rectangle((sx,205,sx+4,660),fill=(130,165,157,16))
        for row,sy in enumerate((235,315,470,550)):
            for j,sx in enumerate(range(92,W,180)):
                ww=46+(j%3)*13; hh=40+(row%2)*12
                d.rounded_rectangle((sx,sy,sx+ww,sy+hh),radius=8,fill=(199,171,122,14 if (j+row)%2 else 20))
        d.rectangle((0,720,W,930),fill=(15,30,35,80))
        d.rectangle((0,720,W,728),fill=(77,115,111,35))
    else:
        # A sparse rule/grid graph is revealed only after the receipt turns into a calculation.
        for x in range(80,W,160): d.line((x,205,x,900),fill=(80,156,164,12),width=2)
        for y in range(220,910,130): d.line((60,y,W-60,y),fill=(80,156,164,14),width=2)
        for i in range(9):
            x=140+i*195; y=260+(i%3)*155
            d.ellipse((x-4,y-4,x+4,y+4),fill=(69,224,198,44))
    # restrained edge glow pools; no noisy particle field competing with copy
    for cx,cy,r,col in [(1520,300,260,(53,188,171,13)),(430,680,270,(249,174,83,10))]:
        for rr in range(r,0,-12):
            a=int(col[2]*(1-rr/r)**2)
            d.ellipse((cx-rr,cy-rr,cx+rr,cy+rr),fill=(col[0],col[1],col[2],a))
    return im
