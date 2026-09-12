"""Export this project's three fixed tensor diagrams to SVG and PNG.

Labels and connectivity come from docs/diagrams/*.mmd. Layout is explicit to keep
long tensor labels and residual connections readable. This is not a general
Mermaid compiler. Requires Pillow and a Chinese font (override with --font).
"""
import argparse
import html
import math
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent / "docs/diagrams"
INK, LINE, BLUE = "#192c3b", "#647888", "#245a81"


def render(name, boxes, size, routes, accent):
    source = (ROOT / f"{name}.mmd").read_text()
    labels, edges = {}, []
    for line in source.splitlines():
        match = re.fullmatch(r'\s*(\w+)\["(.*)"\]', line)
        if match:
            labels[match[1]] = match[2].split("<br/>")
        elif "-->" in line:
            ids = re.sub(r'-->\|[^|]+\|', '-->', line).strip().split('-->')
            edges.extend(zip([v.strip() for v in ids[:-1]], [v.strip() for v in ids[1:]]))
    if labels.keys() != boxes.keys():
        raise ValueError(f"Missing/extra layout nodes in {name}")
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT, 27)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size[0]}" height="{size[1]}" viewBox="0 0 {size[0]} {size[1]}">',
           f'<title>{html.escape(name)}: Transformer tensor operations</title>',
           '<rect width="100%" height="100%" fill="white"/>',
           '<g font-family="Noto Sans CJK SC, sans-serif" font-size="27" fill="#192c3b">']

    def port(node, which):
        x, y, w, h = boxes[node]
        return {"top": (x + w/2, y), "bottom": (x + w/2, y+h),
                "left": (x, y+h/2), "right": (x+w, y+h/2)}[which]

    for a, b in edges:
        custom = routes.get((a, b))
        if custom:
            first, last, bends = custom
            pts = [port(a, first), *bends, port(b, last)]
        else:
            start, end = port(a, "bottom"), port(b, "top")
            middle = (start[1]+end[1])/2
            pts = [start, (start[0], middle), (end[0], middle), end]
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i-1]]
        color = BLUE if (a, b) in accent else LINE
        draw.line(pts, fill=color, width=3)
        points = ' '.join(f'{x},{y}' for x, y in pts)
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        (x0,y0),(x1,y1)=pts[-2:]
        length=math.hypot(x1-x0,y1-y0)
        dx,dy=(x1-x0)/length,(y1-y0)/length
        tip=[(x1,y1),(x1-14*dx+6*dy,y1-14*dy-6*dx),(x1-14*dx-6*dy,y1-14*dy+6*dx)]
        draw.polygon(tip, fill=color)
        svg.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in tip)}" fill="{color}"/>')

    for key, lines in labels.items():
        x,y,w,h=boxes[key]
        if (len(lines)-1)*37+38 > h:
            raise ValueError(f"Label too tall in {name}/{key}")
        fill="#eef4f8" if key in {"ADD","ADD1","ADD2","OUT","LOSS","NEXT"} else "#fafbfc"
        draw.rounded_rectangle((x,y,x+w,y+h),radius=8,fill=fill,outline=LINE,width=2)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{LINE}" stroke-width="2"/>')
        for i,label in enumerate(lines):
            if draw.textlength(label,font=font)>w-28:
                raise ValueError(f"Label too wide in {name}/{key}: {label}")
            cy=y+h/2+(i-(len(lines)-1)/2)*37
            draw.text((x+w/2,cy),label,font=font,fill=INK,anchor="mm")
            svg.append(f'<text x="{x+w/2}" y="{cy}" text-anchor="middle" dominant-baseline="central">{html.escape(label)}</text>')
    svg.append('</g></svg>')
    (ROOT/f'{name}.svg').write_text('\n'.join(svg)+'\n')
    image.save(ROOT/f'{name}.png')
    print(f'{name}: {len(labels)} nodes, {len(edges)} edges, {size}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font',default='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    FONT=parser.parse_args().font
    forward={
        'IDs':(50,30,600,95),'P':(750,30,600,95),
        'TE':(50,175,600,135),'PE':(750,175,600,135),
        'ADD':(340,370,720,135),'B0':(340,560,720,135),
        'B1':(340,750,720,135),'LN':(340,940,720,135),
        'OUT':(300,1130,800,135),'FLAT':(340,1320,720,95),
        'TARGET':(1120,1370,470,135),'LOSS':(340,1535,720,135)}
    # Space for the independent target branch beside the loss calculation.
    render('forward',forward,(1640,1720),{
        ('TARGET','LOSS'):('bottom','right',[(1355,1602.5)])},set())
    block={
        'X':(420,30,680,95),'LN1':(420,175,680,135),
        'Q':(40,370,430,135),'K':(545,370,430,135),'VAL':(1050,370,430,135),
        'KT':(420,565,680,95),'SCORE':(370,720,780,135),
        'MASK':(370,915,780,135),'SOFT':(370,1110,780,135),
        'ADROP':(370,1305,780,135),'HEAD':(370,1500,780,135),
        'CAT':(370,1695,780,135),'PROJ':(370,1890,780,135),
        'ADD1':(370,2085,780,135),'LN2':(370,2280,780,95),
        'FF1':(370,2435,780,135),'ACT':(370,2630,780,135),
        'FF2':(370,2825,780,135),'ADD2':(370,3020,780,135)}
    render('block',block,(1560,3200),{
        ('Q','SCORE'):('bottom','left',[(255,787.5)]),
        ('VAL','HEAD'):('bottom','right',[(1265,1567.5)]),
        ('X','ADD1'):('left','left',[(20,77.5),(20,2152.5)]),
        ('ADD1','ADD2'):('bottom','left',[(760,2250),(240,2250),(240,3087.5)])},
        {('X','ADD1'),('ADD1','ADD2')})
    cache={
        'X':(455,30,690,95),'Q':(50,185,430,95),
        'K':(585,185,430,95),'V':(1120,185,430,95),
        'PK':(585,360,430,95),'PV':(1120,360,430,95),
        'ALLK':(520,530,560,95),'ALLV':(1120,530,460,95),
        'SCORE':(50,695,900,135),'ATTN':(50,890,900,135),
        'OUT':(50,1085,900,135),'NEXT':(1020,1085,580,135)}
    render('kv-cache',cache,(1660,1270),{
        ('X','Q'):('left','top',[(265,77.5)]),
        ('X','V'):('right','top',[(1335,77.5)]),
        ('K','ALLK'):('left','left',[(490,232.5),(490,577.5)]),
        ('V','ALLV'):('right','right',[(1625,232.5),(1625,577.5)]),
        ('Q','SCORE'):('bottom','top',[(265,657),(500,657)]),
        ('ALLK','SCORE'):('bottom','right',[(800,655),(990,655),(990,762.5)]),
        ('ALLV','OUT'):('bottom','right',[(1350,850),(990,850),(990,1152.5)]),
        ('ALLK','NEXT'):('right','top',[(1095,577.5),(1095,1045),(1310,1045)]),
        ('ALLV','NEXT'):('right','top',[(1630,577.5),(1630,1055),(1310,1055)])},
        {('ALLK','NEXT'),('ALLV','NEXT')})
