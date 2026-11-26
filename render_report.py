"""将同一份报告内容按固定 5 页 A4 排版，溢出即报错，不通过缩小整页隐藏问题。"""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle, Image

ROOT=Path(__file__).resolve().parent
WIDTH,HEIGHT=595.2756,841.8898
LEFT,RIGHT,TOP,BOTTOM=45.14,45.14,49,42
CONTENT=WIDTH-LEFT-RIGHT
NAVY=HexColor('#243c54')
TEXT=HexColor('#202b35')


def main():
    pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc'))
    pdfmetrics.registerFont(TTFont('Hei','C:/Windows/Fonts/simhei.ttf'))
    pdfmetrics.registerFontFamily('Song',normal='Song',bold='Hei',italic='Song',boldItalic='Hei')
    styles={
       'title':ParagraphStyle('title',fontName='Hei',fontSize=18,leading=24,textColor=NAVY,spaceAfter=4,wordWrap='CJK'),
       'subtitle':ParagraphStyle('subtitle',fontName='Song',fontSize=10,leading=15,textColor=HexColor('#5b6e7f'),wordWrap='CJK'),
       'h':ParagraphStyle('heading',fontName='Hei',fontSize=12,leading=18,textColor=NAVY,wordWrap='CJK'),
       'p':ParagraphStyle('body',fontName='Song',fontSize=10.4,leading=16.2,textColor=TEXT,wordWrap='CJK'),
       'caption':ParagraphStyle('caption',fontName='Song',fontSize=8.5,leading=12.2,textColor=HexColor('#52616e'),wordWrap='CJK'),
       'ref':ParagraphStyle('reference',fontName='Song',fontSize=8.6,leading=12.3,textColor=TEXT,wordWrap='CJK'),
       'cell':ParagraphStyle('cell',fontName='Song',fontSize=9.1,leading=13.5,textColor=TEXT,wordWrap='CJK'),
       'header':ParagraphStyle('header',fontName='Hei',fontSize=9.0,leading=13,textColor=NAVY,wordWrap='CJK')
    }
    source=json.loads((ROOT/'reports/compact_report.json').read_text(encoding='utf-8'))
    assert len(source['pages'])==5
    out=ROOT/'output/pdf/CIFAR10_HW3_report_5pages.pdf'; out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_suffix('.draft.pdf')
    c=canvas.Canvas(str(temporary),pagesize=(WIDTH,HEIGHT),pageCompression=1)
    c.setTitle(source['title']); c.setAuthor(''); c.setSubject('CIFAR-10 三模型实测报告，5 页提交版')
    layout=[]
    for number,page in enumerate(source['pages'],1):
        c.setFillColor(HexColor('#667989')); c.setFont('Song',8.2)
        c.drawString(LEFT,HEIGHT-28,'当代人工智能 · 作业三')
        c.drawRightString(WIDTH-RIGHT,HEIGHT-28,page['label'])
        c.setStrokeColor(HexColor('#ccd6df')); c.setLineWidth(.45)
        c.line(LEFT,HEIGHT-34,WIDTH-RIGHT,HEIGHT-34)
        y=HEIGHT-TOP
        page_layout=[]
        for index,block in enumerate(page['blocks']):
            kind=block['type']; gap=6
            if kind=='h':
                y-=5; gap=6
            if kind=='table':
                colwidths=[v*CONTENT/505 for v in block['widths']]
                rows=[[Paragraph(escape(value),styles['header']) for value in block['headers']]]
                rows += [[Paragraph(escape(str(value)),styles['cell']) for value in row] for row in block['rows']]
                item=Table(rows,colWidths=colwidths,hAlign='LEFT')
                item.setStyle(TableStyle([
                    ('VALIGN',(0,0),(-1,-1),'TOP'),
                    ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                    ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
                    ('LINEABOVE',(0,0),(-1,0),.8,NAVY),
                    ('LINEBELOW',(0,0),(-1,0),.5,HexColor('#bccbd7')),
                    ('LINEBELOW',(0,-1),(-1,-1),.6,NAVY),
                    ('BACKGROUND',(0,0),(-1,0),HexColor('#edf3f7')),
                    ('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#ffffff'),HexColor('#f7f9fb')])
                ]))
                gap=5
            elif kind=='image':
                item=Image(str(ROOT/block['path']))
                factor=CONTENT/item.imageWidth; item.drawWidth=CONTENT; item.drawHeight=item.imageHeight*factor
                gap=4
            else:
                html=escape(block['text'])
                if kind=='ref' and block.get('url'):
                    html+=' <link href="'+escape(block['url'])+'" color="#256c8a">'+escape(block['url'])+'</link>'
                item=Paragraph(html,styles[kind])
                if kind in ['caption','ref']: gap=4
                if kind=='subtitle': gap=12
            w,h=item.wrap(CONTENT,HEIGHT)
            if y-h<BOTTOM:
                raise RuntimeError(f'第 {number} 页第 {index+1} 块溢出：还需 {BOTTOM-(y-h):.1f} pt；请精简或调整局部排版')
            item.drawOn(c,LEFT,y-h)
            page_layout.append({'block':index,'type':kind,'x':LEFT,'top':y,'bottom':y-h,'height':h})
            y-=h+gap
        c.setFillColor(HexColor('#667989')); c.setFont('Song',8.3)
        c.drawString(LEFT,25,'CIFAR-10 · 正式实验')
        c.drawRightString(WIDTH-RIGHT,25,f'{number} / 5')
        c.showPage()
        layout.append({'page':number,'used_height':HEIGHT-TOP-y,'bottom_content':y,'blocks':page_layout})
    c.save(); temporary.replace(out)
    (ROOT/'reports/compact_layout.json').write_text(json.dumps(layout,ensure_ascii=False,indent=2),encoding='utf-8')
    print(out)
    print('5 页固定排版完成；各页正文底部坐标：',[round(page['bottom_content'],1) for page in layout])


if __name__=='__main__':
    main()
