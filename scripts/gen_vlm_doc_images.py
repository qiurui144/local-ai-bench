#!/usr/bin/env python3
"""One-time script: generate 30 synthetic document PNG images + cases.jsonl for S7.

Usage: python3 scripts/gen_vlm_doc_images.py
Requires: Pillow
Output:
  datasets/scenarios/vlm_document_extraction/cases.jsonl   (30 lines)
  fixtures/scenarios/vlm_document_extraction/**/*.png       (30 files, VLM convention per DEVELOP.md §4)
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent.parent
IMG_BASE = ROOT / "fixtures/scenarios/vlm_document_extraction"
CASES_PATH = ROOT / "datasets/scenarios/vlm_document_extraction/cases.jsonl"
FONT_PATH = Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")


# ── image helpers ─────────────────────────────────────────────────────────────

def _fonts(size_lg=26, size_sm=20):
    try:
        return (ImageFont.truetype(str(FONT_PATH), size_lg),
                ImageFont.truetype(str(FONT_PATH), size_sm))
    except OSError:
        d = ImageFont.load_default()
        return d, d


def make_doc_image(title: str, rows: list[tuple[str, str]], size=(800, 500)) -> Image.Image:
    """White background document with title bar and key:value rows."""
    img = Image.new("RGB", size, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font_lg, font_sm = _fonts()
    W = size[0]
    pad = 40

    # Title bar
    draw.rectangle([0, 0, W, 64], fill=(215, 228, 248))
    bbox = draw.textbbox((0, 0), title, font=font_lg)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 16), title, fill=(25, 40, 100), font=font_lg)

    # Divider
    draw.line([pad, 74, W - pad, 74], fill=(170, 170, 190), width=2)

    # Rows
    y = 94
    for key, val in rows:
        draw.text((pad, y), key + "：", fill=(90, 90, 110), font=font_sm)
        draw.text((pad + 210, y), val, fill=(15, 15, 15), font=font_sm)
        draw.line([pad, y + 36, W - pad, y + 36], fill=(230, 230, 235), width=1)
        y += 44

    # Outer border
    draw.rectangle([pad - 4, 20, W - pad + 4, y + 12], outline=(190, 200, 215), width=2)
    return img


# ── document generators ───────────────────────────────────────────────────────

def gen_bank_statement(d: dict, path: Path):
    rows = [
        ("交易日期", d["transaction_date"]),
        ("交易金额", d["transaction_amount"] + " 元"),
        ("账户余额", d["balance"] + " 元"),
        ("对方户名", d["counterparty"]),
        ("摘    要", d["description"]),
    ]
    make_doc_image("中 国 银 行  流 水 明 细", rows).save(path)


def gen_vat_invoice(d: dict, path: Path):
    rows = [
        ("发票号码", d["invoice_number"]),
        ("开票日期", d["invoice_date"]),
        ("销 售 方", d["seller"]),
        ("购 买 方", d["buyer"]),
        ("金  额", d["amount"] + " 元"),
        ("税  率", d["tax_rate"]),
        ("价税合计", d["total_amount"] + " 元"),
    ]
    make_doc_image("增 值 税 普 通 发 票", rows, size=(800, 560)).save(path)


def gen_receipt(d: dict, path: Path):
    rows = [
        ("收据编号", d["receipt_number"]),
        ("日    期", d["date"]),
        ("商    家", d["merchant"]),
        ("消费金额", d["amount"] + " 元"),
        ("支付方式", d["payment_method"]),
    ]
    make_doc_image("消  费  收  据", rows).save(path)


def gen_bank_transfer(d: dict, path: Path):
    rows = [
        ("汇款日期", d["transfer_date"]),
        ("汇款方账号", d["sender_account"]),
        ("收款方姓名", d["receiver_name"]),
        ("收款方账号", d["receiver_account"]),
        ("汇款金额", d["amount"] + " 元"),
        ("开 户 行", d["bank_name"]),
    ]
    make_doc_image("银 行 汇 款 凭 证", rows).save(path)


# ── case data ─────────────────────────────────────────────────────────────────

_BS_FIELDS = ["transaction_date", "transaction_amount", "balance", "counterparty", "description"]
_VI_FIELDS = ["invoice_number", "invoice_date", "seller", "buyer", "amount", "tax_rate", "total_amount"]
_RC_FIELDS = ["receipt_number", "date", "merchant", "amount", "payment_method"]
_BT_FIELDS = ["transfer_date", "sender_account", "receiver_name", "receiver_account", "amount", "bank_name"]

BANK_STATEMENT_DATA = [
    {"transaction_date": "2024-01-10", "transaction_amount": "5000.00",
     "balance": "28500.00", "counterparty": "张三丰", "description": "工资"},
    {"transaction_date": "2024-01-15", "transaction_amount": "-320.50",
     "balance": "28179.50", "counterparty": "美团外卖", "description": "外卖消费"},
    {"transaction_date": "2024-02-01", "transaction_amount": "1200.00",
     "balance": "29379.50", "counterparty": "支付宝", "description": "退款"},
    {"transaction_date": "2024-02-14", "transaction_amount": "-688.00",
     "balance": "28691.50", "counterparty": "携程旅行", "description": "机票"},
    {"transaction_date": "2024-03-01", "transaction_amount": "8000.00",
     "balance": "36691.50", "counterparty": "上海科技有限公司", "description": "项目款"},
    {"transaction_date": "2024-03-05", "transaction_amount": "-2500.00",
     "balance": "34191.50", "counterparty": "房东李明", "description": "房租"},
    {"transaction_date": "2024-03-10", "transaction_amount": "-156.80",
     "balance": "34034.70", "counterparty": "京东商城", "description": "网购"},
    {"transaction_date": "2024-03-20", "transaction_amount": "3000.00",
     "balance": "37034.70", "counterparty": "王五", "description": "还款"},
]

VAT_INVOICE_DATA = [
    {"invoice_number": "31200024501234", "invoice_date": "2024-03-15",
     "seller": "北京科技有限公司", "buyer": "上海贸易有限公司",
     "amount": "10000.00", "tax_rate": "13%", "total_amount": "11300.00"},
    {"invoice_number": "44000018760056", "invoice_date": "2024-02-20",
     "seller": "广州电子设备有限公司", "buyer": "深圳创新科技股份有限公司",
     "amount": "25000.00", "tax_rate": "13%", "total_amount": "28250.00"},
    {"invoice_number": "31100025890123", "invoice_date": "2024-01-08",
     "seller": "杭州软件技术有限公司", "buyer": "北京数字科技有限公司",
     "amount": "50000.00", "tax_rate": "6%", "total_amount": "53000.00"},
    {"invoice_number": "51000032145678", "invoice_date": "2024-03-28",
     "seller": "成都制造业有限公司", "buyer": "重庆工贸股份有限公司",
     "amount": "75000.00", "tax_rate": "13%", "total_amount": "84750.00"},
    {"invoice_number": "33000021987654", "invoice_date": "2024-02-05",
     "seller": "宁波进出口有限公司", "buyer": "温州小商品贸易有限公司",
     "amount": "3600.00", "tax_rate": "9%", "total_amount": "3924.00"},
    {"invoice_number": "11000019234567", "invoice_date": "2024-03-01",
     "seller": "北京咨询顾问有限公司", "buyer": "天津金融服务有限公司",
     "amount": "15000.00", "tax_rate": "6%", "total_amount": "15900.00"},
    {"invoice_number": "37000028765432", "invoice_date": "2024-01-25",
     "seller": "济南食品配送有限公司", "buyer": "青岛连锁超市有限公司",
     "amount": "8800.00", "tax_rate": "9%", "total_amount": "9592.00"},
    {"invoice_number": "42000031456789", "invoice_date": "2024-02-28",
     "seller": "武汉建材有限公司", "buyer": "长沙房地产开发有限公司",
     "amount": "120000.00", "tax_rate": "13%", "total_amount": "135600.00"},
]

RECEIPT_DATA = [
    {"receipt_number": "REC-20240315-001", "date": "2024-03-15",
     "merchant": "星巴克咖啡", "amount": "68.00", "payment_method": "微信支付"},
    {"receipt_number": "REC-20240218-042", "date": "2024-02-18",
     "merchant": "麦当劳", "amount": "45.50", "payment_method": "支付宝"},
    {"receipt_number": "REC-20240301-128", "date": "2024-03-01",
     "merchant": "全家便利店", "amount": "23.80", "payment_method": "云闪付"},
    {"receipt_number": "REC-20240110-007", "date": "2024-01-10",
     "merchant": "沃尔玛超市", "amount": "356.90", "payment_method": "银行卡"},
    {"receipt_number": "REC-20240225-033", "date": "2024-02-25",
     "merchant": "肯德基", "amount": "89.00", "payment_method": "微信支付"},
    {"receipt_number": "REC-20240308-156", "date": "2024-03-08",
     "merchant": "永辉超市", "amount": "128.60", "payment_method": "支付宝"},
    {"receipt_number": "REC-20240320-089", "date": "2024-03-20",
     "merchant": "罗森便利店", "amount": "19.50", "payment_method": "现金"},
]

BANK_TRANSFER_DATA = [
    {"transfer_date": "2024-03-15", "sender_account": "622202****7890",
     "receiver_name": "李四", "receiver_account": "622848****7891",
     "amount": "50000.00", "bank_name": "中国工商银行"},
    {"transfer_date": "2024-02-20", "sender_account": "621226****3456",
     "receiver_name": "赵云", "receiver_account": "620058****9012",
     "amount": "8800.00", "bank_name": "中国农业银行"},
    {"transfer_date": "2024-01-05", "sender_account": "622588****2345",
     "receiver_name": "孙红雷", "receiver_account": "621700****6789",
     "amount": "25000.00", "bank_name": "中国建设银行"},
    {"transfer_date": "2024-03-10", "sender_account": "622262****5678",
     "receiver_name": "刘德华", "receiver_account": "622312****3456",
     "amount": "1500.00", "bank_name": "中国银行"},
    {"transfer_date": "2024-02-14", "sender_account": "621661****8901",
     "receiver_name": "周杰伦", "receiver_account": "621480****4567",
     "amount": "100000.00", "bank_name": "招商银行"},
    {"transfer_date": "2024-01-20", "sender_account": "622669****1234",
     "receiver_name": "王菲", "receiver_account": "621483****7890",
     "amount": "3600.00", "bank_name": "交通银行"},
    {"transfer_date": "2024-03-25", "sender_account": "623095****4567",
     "receiver_name": "林俊杰", "receiver_account": "621226****0123",
     "amount": "12000.00", "bank_name": "中国邮政储蓄银行"},
]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    for subdir in ["bank_statement", "vat_invoice", "receipt", "bank_transfer"]:
        (IMG_BASE / subdir).mkdir(parents=True, exist_ok=True)
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    case_id = 1

    # bank_statement (c01–c08)
    for d in BANK_STATEMENT_DATA:
        cid = f"c{case_id:02d}"
        img_rel = f"fixtures/scenarios/vlm_document_extraction/bank_statement/{cid}.png"
        gen_bank_statement(d, ROOT / img_rel)
        cases.append({"id": cid, "provenance": "synthetic", "payload": {
            "document_type": "bank_statement",
            "image_path": img_rel,
            "fields": _BS_FIELDS,
            "golden": {k: d[k] for k in _BS_FIELDS},
        }})
        print(f"  [{cid}] bank_statement → {img_rel}")
        case_id += 1

    # vat_invoice (c09–c16)
    for d in VAT_INVOICE_DATA:
        cid = f"c{case_id:02d}"
        img_rel = f"fixtures/scenarios/vlm_document_extraction/vat_invoice/{cid}.png"
        gen_vat_invoice(d, ROOT / img_rel)
        cases.append({"id": cid, "provenance": "synthetic", "payload": {
            "document_type": "vat_invoice",
            "image_path": img_rel,
            "fields": _VI_FIELDS,
            "golden": {k: d[k] for k in _VI_FIELDS},
        }})
        print(f"  [{cid}] vat_invoice    → {img_rel}")
        case_id += 1

    # receipt (c17–c23)
    for d in RECEIPT_DATA:
        cid = f"c{case_id:02d}"
        img_rel = f"fixtures/scenarios/vlm_document_extraction/receipt/{cid}.png"
        gen_receipt(d, ROOT / img_rel)
        cases.append({"id": cid, "provenance": "synthetic", "payload": {
            "document_type": "receipt",
            "image_path": img_rel,
            "fields": _RC_FIELDS,
            "golden": {k: d[k] for k in _RC_FIELDS},
        }})
        print(f"  [{cid}] receipt        → {img_rel}")
        case_id += 1

    # bank_transfer (c24–c30)
    for d in BANK_TRANSFER_DATA:
        cid = f"c{case_id:02d}"
        img_rel = f"fixtures/scenarios/vlm_document_extraction/bank_transfer/{cid}.png"
        gen_bank_transfer(d, ROOT / img_rel)
        cases.append({"id": cid, "provenance": "synthetic", "payload": {
            "document_type": "bank_transfer",
            "image_path": img_rel,
            "fields": _BT_FIELDS,
            "golden": {k: d[k] for k in _BT_FIELDS},
        }})
        print(f"  [{cid}] bank_transfer  → {img_rel}")
        case_id += 1

    with open(CASES_PATH, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n✓ Generated {len(cases)} cases → {CASES_PATH}")
    print(f"✓ Images in {IMG_BASE}")


if __name__ == "__main__":
    main()
