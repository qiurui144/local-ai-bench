"""确定性渲染合成微信风格聊天截图(PIL,无浏览器依赖)。

用法: python scripts/render_wechat_case.py   # 渲染 datasets 内全部 wechat case
输入: datasets/scenarios/wechat_intent/dialogs.json
      [{"id": "c1", "messages": [{"side": "left|right", "text": "..."}]}]
输出: fixtures/scenarios/wechat_intent/<id>.png(672x↕,微信式气泡布局)

只生成合成图;真实截图(PII)永不入 git。
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DIALOGS = ROOT / "datasets/scenarios/wechat_intent/dialogs.json"
OUT_DIR = ROOT / "fixtures/scenarios/wechat_intent"

W, PAD, BUBBLE_W = 672, 16, 440
BG, LEFT_BG, RIGHT_BG, FG = "#ededed", "#ffffff", "#95ec69", "#111111"


def _font(size=24):
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"):
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > max_w:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def render(case_id: str, messages: list[dict]) -> Path:
    font = _font()
    probe = ImageDraw.Draw(Image.new("RGB", (W, 100)))
    blocks = []
    y = PAD
    for m in messages:
        lines = _wrap(probe, m["text"], font, BUBBLE_W - 2 * PAD)
        h = len(lines) * 34 + 2 * PAD
        blocks.append((m["side"], lines, y, h))
        y += h + 12
    img = Image.new("RGB", (W, y + PAD), BG)
    d = ImageDraw.Draw(img)
    for side, lines, top, h in blocks:
        bw = max(probe.textlength(ln, font=font) for ln in lines) + 2 * PAD
        x0 = PAD if side == "left" else W - PAD - int(bw)
        d.rounded_rectangle([x0, top, x0 + int(bw), top + h], radius=10,
                            fill=LEFT_BG if side == "left" else RIGHT_BG)
        for i, ln in enumerate(lines):
            d.text((x0 + PAD, top + PAD + i * 34), ln, fill=FG, font=font)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{case_id}.png"
    img.save(out)
    return out


def main():
    dialogs = json.loads(DIALOGS.read_text(encoding="utf-8"))
    for dlg in dialogs:
        print("rendered", render(dlg["id"], dlg["messages"]))


if __name__ == "__main__":
    main()
