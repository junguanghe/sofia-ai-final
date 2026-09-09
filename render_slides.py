"""Render a small Markdown subset as PDF slides using ReportLab.

Each --- separator starts a slide. Supports headings, paragraphs, bullets,
fenced code, tables, links and images. No LaTeX, browser or external converter.
"""
import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, PageBreak,
                               PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)


def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<link href="\2" color="#245a81">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default="docs/slides.md")
    parser.add_argument("output", nargs="?", default="output/pdf/mrp.pdf")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Embed ReportLab's bundled fonts so the PDF does not depend on viewer substitution.
    import reportlab
    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("Vera", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("Vera-Bold", str(fonts / "VeraBd.ttf")))
    pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold")
    styles = {
        "title": ParagraphStyle("title", fontName="Vera-Bold", fontSize=30, leading=35,
                                textColor=colors.HexColor("#203e55"), spaceAfter=17),
        "body": ParagraphStyle("body", fontName="Vera", fontSize=18, leading=25, spaceAfter=11),
        "cell": ParagraphStyle("cell", fontName="Vera", fontSize=15, leading=20),
        "code": ParagraphStyle("code", fontName="Courier", fontSize=15, leading=20, spaceAfter=12),
    }
    slides = re.split(r"\n---\s*\n", source.read_text())
    story = []
    for slide_index, slide in enumerate(slides):
        if slide_index:
            story.append(PageBreak())
        lines = slide.strip().splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1; continue
            if line.startswith("# "):
                story.append(Paragraph(inline(line[2:]), styles["title"]))
            elif line.startswith("```"):
                code = []; i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code.append(lines[i]); i += 1
                story.append(Preformatted("\n".join(code), styles["code"]))
            elif line.startswith("|"):
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    cells = [s.strip() for s in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r"[:\- ]+", c) for c in cells):
                        rows.append([Paragraph(inline(c), styles["cell"]) for c in cells])
                    i += 1
                widths = [864 / len(rows[0])] * len(rows[0])
                table = Table(rows, colWidths=widths, hAlign="LEFT")
                table.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef2f5")),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#8192a0")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                ]))
                story.extend([table, Spacer(1, 15)]); continue
            elif re.fullmatch(r"!\[.*\]\(.*\)", line):
                name = re.fullmatch(r"!\[.*\]\((.*)\)", line).group(1)
                image = Image(str(source.parent / name))
                scale = min(864 / image.imageWidth, 270 / image.imageHeight)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                image.hAlign = "LEFT"
                story.extend([image, Spacer(1, 9)])
            elif line.startswith("- "):
                story.append(Paragraph(inline(line[2:]), styles["body"], bulletText="-"))
            else:
                paragraph = [line]
                while i + 1 < len(lines) and lines[i + 1].strip() and not lines[i + 1].startswith(("#", "- ", "|", "```", "![")):
                    i += 1; paragraph.append(lines[i].strip())
                story.append(Paragraph(inline(" ".join(paragraph)), styles["body"]))
            i += 1

    def footer(canvas, doc):
        canvas.setFont("Vera", 10)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(42, 19, "Junguang He / MSCS2201 / September 2026")
        canvas.drawRightString(918, 19, f"{doc.page} / {len(slides)}")

    doc = BaseDocTemplate(str(output), pagesize=(960, 540), title="Understanding KV Caching",
                          author="Junguang He")
    doc.addPageTemplates(PageTemplate(id="slide", frames=Frame(42, 42, 876, 462,
                          leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0), onPage=footer))
    doc.build(story)
    if doc.page != len(slides):
        raise RuntimeError(f"Content overflow: expected {len(slides)} pages, got {doc.page}")
    print(f"Rendered {len(slides)} slides to {output}")


if __name__ == "__main__":
    main()
