# Rockchip RKNN3 Model Cache

This project tracks RK3588/RK182X model coverage in `models.yaml`, but the
large `.rknn` artifacts stay outside git under `drivers/`.

## Scope

The RKNN3 v1.0.4 coverage includes:

- LLM: Qwen2.5 0.5B/1.5B/3B/7B, Qwen3 0.6B/1.7B/4B/8B, CoPaw 4B 4k/8k/32k.
- VLM: FastVLM, InternVL, MiniCPM, Qwen2.5-VL, Qwen3-VL, SmolVLM, UI-TARS, Gemma4.
- OCR/VLM: PaddleOCR-VL.

Current status on 2026-07-07:

- Manifest generation works from `models.yaml`.
- Full RKNN3 v1.0.4 scope resolves 46 RKNN artifacts.
- Local cache is complete: 46/46 artifacts cached under
  `drivers/rockchip-rknn3-models/`.
- Cache size is about 1.2GB on disk; `cache-index.tsv` contains 46 SHA256 rows
  plus the header.
- Scope checks are complete: LLM 11/11, VLM 33/33, others/OCR 2/2.

## Commands

Print the manifest:

```bash
python3 scripts/cache_rockchip_rknn3_models.py --manifest
```

Populate from a mounted Model Zoo directory:

```bash
ROCKCHIP_RKNN3_LOCAL_ROOT=/mnt/RKNN3_SDK/rknn3_models \
python3 scripts/cache_rockchip_rknn3_models.py --download-missing
```

Populate from an internal HTTP mirror:

```bash
ROCKCHIP_RKNN3_BASE_URL=https://example.internal/rknn3_models \
python3 scripts/cache_rockchip_rknn3_models.py --download-missing
```

Populate from a Lenovo Filez share after loading the share URL and extraction
code into the local secure environment:

```bash
python3 scripts/cache_rockchip_rknn3_models.py --download-missing
```

Populate from a reachable RK target after loading connection values into the
local secure environment:

```bash
python3 scripts/cache_rockchip_rknn3_models.py --download-missing
```

Sync cached artifacts back to the target after loading connection values into
the local secure environment:

```bash
python3 scripts/cache_rockchip_rknn3_models.py --sync-to-device
```

The local cache root defaults to:

```text
drivers/rockchip-rknn3-models/
```

`cache-index.tsv` records size, SHA256, and relative path for cached files.

The download command is intentionally source-driven: it copies from
`ROCKCHIP_RKNN3_LOCAL_ROOT`, downloads from `ROCKCHIP_RKNN3_BASE_URL`, downloads
from a Filez share through `ROCKCHIP_RKNN3_LENOVO_SHARE_URL`, or pulls from
`RK3588_HOST`/`RK3588_USER`. It does not embed private URLs or credentials.

## Policy

Do not write real hostnames, IPs, usernames, passwords, or device serials into
reports, scripts, or docs. Use environment variables for all device access.
