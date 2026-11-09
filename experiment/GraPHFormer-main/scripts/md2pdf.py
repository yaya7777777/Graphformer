"""Convert the two GraPHFormer markdown reports to PDF via playwright."""

import os
import re
import sys

import markdown2
from playwright.sync_api import sync_playwright

ROOT = r"e:\my_file\research2\综述\GraPHFormer"

CSS = """
@page { margin: 18mm 16mm; size: A4; }
body { font-family: "Microsoft YaHei", "微软雅黑", "SimSun", "宋体", serif;
       font-size: 11pt; line-height: 1.6; color: #222; }
h1 { font-size: 20pt; border-bottom: 2px solid #2c3e50; padding-bottom: 6px;
     page-break-after: avoid; }
h2 { font-size: 16pt; border-bottom: 1px solid #bdc3c7; padding-bottom: 4px;
     margin-top: 28px; page-break-after: avoid; }
h3 { font-size: 13pt; color: #2c3e50; page-break-after: avoid; }
h4 { font-size: 11.5pt; color: #34495e; page-break-after: avoid; }
p { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10pt;
        page-break-inside: avoid; }
th, td { border: 1px solid #95a5a6; padding: 5px 8px; text-align: left; }
th { background: #ecf0f1; font-weight: bold; }
img { max-width: 95%; height: auto; display: block; margin: 14px auto;
      border: 1px solid #ddd; padding: 3px; background: #fff;
      page-break-inside: avoid; }
code { background: #f5f5f5; padding: 1px 4px; border-radius: 3px;
       font-family: Consolas, "Courier New", monospace; font-size: 10pt; }
pre { background: #f5f5f5; padding: 10px; border-radius: 4px;
      overflow-x: auto; font-size: 9.5pt; border-left: 3px solid #3498db;
      page-break-inside: avoid; white-space: pre-wrap; }
blockquote { border-left: 4px solid #3498db; margin: 10px 0;
             padding: 6px 14px; background: #f8f9fa; color: #555; }
hr { border: none; border-top: 1px dashed #95a5a6; margin: 20px 0; }
"""


def md_to_html(md_path):
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    # convert relative fig/*.png -> absolute file:/// URLs
    base = os.path.dirname(md_path).replace("\\", "/")
    def repl(m):
        alt, rel = m.group(1), m.group(2)
        if rel.startswith(("http", "file://")):
            return m.group(0)
        abs_url = "file:///" + os.path.normpath(os.path.join(base, rel)).replace("\\", "/")
        return f'![{alt}]({abs_url})'
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', repl, text)
    body = markdown2.markdown(text, extras=["tables", "fenced-code-blocks",
                                             "task-list", "header-ids", "strike"])
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title></title>
<style>{CSS}</style></head>
<body>{body}</body></html>"""


def main():
    targets = [
        os.path.join(ROOT, "GraPHFormer复现实验报告.md"),
        os.path.join(ROOT, "GraPHFormer_论文方法与复现实验汇总.md"),
    ]
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for md in targets:
            if not os.path.exists(md):
                print(f"SKIP: {md}"); continue
            pdf = os.path.splitext(md)[0] + ".pdf"
            html = md_to_html(md)
            page.set_content(html, wait_until="networkidle")
            page.pdf(path=pdf, format="A4", print_background=True)
            print(f"OK: {pdf}  ({os.path.getsize(pdf)//1024} KB)")
        browser.close()


if __name__ == "__main__":
    main()
