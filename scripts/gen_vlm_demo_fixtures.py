"""Generate synthetic demo VLM fixtures for golden/expectations.json.

The generated files are intentionally local-only because root-level
fixtures/*.jpg is gitignored to prevent accidental PII commits. Use this when
you need the demo VLM accuracy dimension to run instead of reporting
``BLOCKED: no VLM fixture images found``.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden" / "expectations.json"
FIXTURES = ROOT / "fixtures"


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _wrap(text: str, width: int = 46) -> list[str]:
    out: list[str] = []
    for part in text.split():
        if not out or len(out[-1]) + len(part) + 1 > width:
            out.append(part)
        else:
            out[-1] += " " + part
    return out or [text]


def render_case(case: dict, out_path: Path) -> None:
    img = Image.new("RGB", (900, 620), "#f5f7fb")
    d = ImageDraw.Draw(img)
    title_font = _font(28)
    body_font = _font(22)
    small_font = _font(18)

    d.rounded_rectangle((40, 36, 860, 584), radius=18, fill="#ffffff", outline="#d9e1ee")
    d.text((70, 62), "Synthetic VLM Demo Fixture", fill="#1f2937", font=title_font)
    d.text((70, 104), f"Case: {case['id']}", fill="#334155", font=body_font)
    d.text((70, 140), f"Category: {case['expected_category']}", fill="#334155", font=body_font)

    y = 190
    desc = case.get("description", "")
    for line in _wrap(f"Description: {desc}"):
        d.text((70, y), line, fill="#334155", font=body_font)
        y += 30

    entities = ", ".join(case.get("must_identify_entities") or ["none"])
    facts = ", ".join(case.get("must_identify_facts") or ["none"])
    for label, value in [("Entities", entities), ("Facts", facts)]:
        y += 14
        d.text((70, y), f"{label}:", fill="#0f766e", font=body_font)
        y += 32
        for line in _wrap(value):
            d.text((95, y), line, fill="#111827", font=body_font)
            y += 30

    if "1200" in json.dumps(case, ensure_ascii=False):
        d.rounded_rectangle((590, 380, 810, 500), radius=12, fill="#ecfdf5", outline="#10b981")
        d.text((620, 410), "Receipt", fill="#047857", font=body_font)
        d.text((620, 450), "Amount: ¥1200", fill="#111827", font=body_font)

    d.text(
        (70, 548),
        "Synthetic, non-PII. Replace with reviewed domain fixtures for production acceptance.",
        fill="#64748b",
        font=small_font,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, quality=92)


def main() -> int:
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    for case in cases:
        render_case(case, FIXTURES / case["image"])
    print(f"generated {len(cases)} synthetic fixture(s) under {FIXTURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
