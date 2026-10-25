# -*- coding: utf-8 -*-
import re, html, sys
src=open('docs/report/DRAFT_ALIGNED_DECK.html',encoding='utf-8').read()
slides=re.findall(r'<section class="slide[^"]*">(.*?)</section>', src, re.S)
def wpc(fam, fs):
    if 'Mono' in fam: return fs*0.600
    if 'Serif' in fam: return fs*0.468
    if 'Bricolage' in fam: return fs*0.545
    return fs*0.500
rep=[]
for n,sl in enumerate(slides,1):
    for svg in re.finditer(r'<svg[^>]*viewBox="0 0 ([\d.]+) ([\d.]+)"(.*?)</svg>', sl, re.S):
        VW,VH=float(svg.group(1)),float(svg.group(2)); body=svg.group(3)
        boxes=[]
        for g in re.finditer(r'<g([^>]*)>(.*?)</g>', body, re.S):
            gattr, inner = g.group(1), g.group(2)
            for it in re.finditer(r'<text([^>]*)>(.*?)</text>', inner, re.S):
                boxes.append((it.group(1), gattr, it.group(2)))
        stripped = re.sub(r'<g[^>]*>.*?</g>', '', body, flags=re.S)
        for it in re.finditer(r'<text([^>]*)>(.*?)</text>', stripped, re.S):
            boxes.append((it.group(1), '', it.group(2)))
        out=[]
        for a,gattr,raw in boxes:
            t=html.unescape(re.sub(r'<[^>]+>','',raw)).strip()
            if not t: continue
            def get(k):
                m=re.search(k+r'="([^"]*)"',a) or re.search(k+r'="([^"]*)"',gattr)
                return m.group(1) if m else None
            fs=float(get('font-size') or 12); fam=get('font-family') or 'sans'
            xm=re.search(r'\bx="([-\d.]+)"',a); ym=re.search(r'\by="([-\d.]+)"',a)
            if not xm or not ym: continue
            x=float(xm.group(1)); y=float(ym.group(1)); anc=get('text-anchor') or 'start'
            w=len(t)*wpc(fam,fs)
            x0 = x-w/2 if anc=='middle' else (x-w if anc=='end' else x)
            if x0+w > VW+1 or y > VH+1 or x0 < -1:
                out.append((round(x0+w-VW), round(y-VH), t[:64]))
        if out:
            rep.append((n,int(VW),int(VH),out))
for n,VW,VH,out in rep:
    print("slide %2d  viewBox %dx%d"%(n,VW,VH))
    for over_x, over_y, t in out:
        tag = ("right +%d"%over_x) if over_x>0 else ""
        tag += ("  bottom +%d"%over_y) if over_y>0 else ""
        print("    %-14s %s"%(tag, t))

# Usage: /venv/main/bin/python docs/report/check_deck_layout.py
# Reports any <text> in any deck SVG whose estimated box leaves its viewBox.
# Text past the viewBox is clipped by the browser, which is how "overlapping"
# and "cut off" labels appear on a rendered slide.
