#!/usr/bin/env python
"""Assemble docs/report/DRAFT_ALIGNED_EXPLAINER.html (the published artifact) from its parts.

The published page must be self-contained -- the artifact CSP blocks external images -- so the
matplotlib figures under results/figures/draftalign/ are inlined as base64 data URIs. Edit the
_explainer_*.html parts (or the figure scripts), then re-run this, then republish the assembled file.

Usage: /venv/main/bin/python docs/report/build_explainer.py
"""
import base64
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "docs", "report")
F = os.path.join(ROOT, "results", "figures", "draftalign")
FILES = {os.path.basename(p).split("_")[0]: p for p in glob.glob(os.path.join(F, "F*.png"))}
CAPS = os.path.join(R, "_explainer_captions.py")


def img(key, alt, caption):
    b = base64.b64encode(open(FILES[key], "rb").read()).decode()
    return (f'<figure class="wide"><img alt="{alt}" src="data:image/png;base64,{b}">'
            f'<figcaption>{caption}</figcaption></figure>')


def main():
    ns = {}
    exec(open(CAPS).read(), ns)
    body = "".join(open(os.path.join(R, f"_explainer_body{i}.html")).read() for i in (1, 2, 3, 4))
    body = body.replace("{{DIA_READS}}", open(os.path.join(R, "_dia_reads.html")).read())
    body = body.replace("{{DIA_LAYER}}", open(os.path.join(R, "_dia_layer.html")).read())
    for k, (alt, cap) in ns["CAPTIONS"].items():
        body = body.replace("{{FIG_%s}}" % k, img(k, alt, cap))
    assert "{{" not in body, "unsubstituted placeholder"
    out = open(os.path.join(R, "_explainer_head.html")).read() + "\n" + body
    path = os.path.join(R, "DRAFT_ALIGNED_EXPLAINER.html")
    open(path, "w").write(out)
    print("wrote %s (%.2f MB)" % (path, len(out) / 1e6))


if __name__ == "__main__":
    main()
