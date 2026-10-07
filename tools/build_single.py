#!/usr/bin/env python3
"""サーバー不要でダブルクリックで開ける 1 ファイル版 vtuber.html を作る。

    python3 tools/build_single.py
"""
import base64
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "assets" / "model"


def strip_module(src):
    # 同じファイル内に連結するので import / export を外し、
    # 名前がぶつからないよう各ファイルを関数スコープに包んで export だけを外に出す
    src = re.sub(r"^import .*?;\n", "", src, flags=re.M)
    names = re.findall(r"^export (?:const|let|class|function|async function) (\w+)", src, flags=re.M)
    src = re.sub(r"^export ", "", src, flags=re.M)
    if not names:
        return src
    exported = ", ".join(names)
    return f"const {{ {exported} }} = (() => {{\n{src}\nreturn {{ {exported} }};\n}})();\n"


def main():
    model = json.loads((MODEL / "model.json").read_text())
    images = {
        L["file"]: "data:image/png;base64," + base64.b64encode((MODEL / L["file"]).read_bytes()).decode()
        for L in model["layers"]
    }
    js = "\n".join(strip_module((ROOT / "src" / f).read_text())
                   for f in ("renderer.js", "rig.js", "tracker.js", "main.js"))
    html = (ROOT / "index.html").read_text()
    css = (ROOT / "style.css").read_text()
    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}</style>")
    embedded = json.dumps({"model": model, "images": images}, separators=(",", ":"))
    html = html.replace(
        '<script type="module" src="src/main.js"></script>',
        f"<script>window.__EMBEDDED_MODEL={embedded};</script>\n<script type=\"module\">\n{js}</script>",
    )
    out = ROOT / "vtuber.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
