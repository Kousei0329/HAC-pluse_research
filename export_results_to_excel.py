"""
Export HAC experiment results to Excel.

Usage:
    python export_results_to_excel.py <experiment_dir> [output.xlsx]

If output path is omitted, saves as <experiment_dir>/results_summary.xlsx
"""

from __future__ import annotations

import sys
import json
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side, numbers
)
from openpyxl.utils import get_column_letter

# ── Style constants ──────────────────────────────────────────────────────────
SCENE_FILL   = PatternFill("solid", fgColor="4472C4")   # blue  – scene header
LAMBDA_FILL  = PatternFill("solid", fgColor="D9E1F2")   # light – λ row
SIZE_FILL    = PatternFill("solid", fgColor="FFFFFF")   # white – size rows
METRIC_FILL  = PatternFill("solid", fgColor="E2EFDA")   # green – metric rows
AVG_FILL     = PatternFill("solid", fgColor="FCE4D6")   # orange– average block

WHITE_FONT   = Font(bold=True, color="FFFFFF")
BOLD_FONT    = Font(bold=True)
THIN         = Side(style="thin")
FULL_BORDER  = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

SIZE_KEYS    = ["anchor", "feat", "scaling", "offsets", "hash", "masks", "MLPs", "Total"]
SIZE_LABELS  = ["anchor", "feat", "scaling", "offset", "hash", "masks", "MLP", "total"]
METRIC_KEYS  = ["PSNR", "SSIM", "LPIPS"]
ANCHOR_LABEL = "anchor num"

ROW_LABELS   = SIZE_LABELS + METRIC_KEYS + [ANCHOR_LABEL]   # 13 rows per block


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_outputs_log(log_path: Path) -> dict | None:
    """Extract encoded sizes (MB) and final anchor_num from outputs.log."""
    if not log_path.exists():
        return None

    text = log_path.read_text(errors="replace")

    # Last "Encoded sizes in MB:" line
    enc_matches = re.findall(
        r"Encoded sizes in MB:\s*"
        r"anchor ([\d.]+),\s*feat ([\d.]+),\s*scaling ([\d.]+),\s*"
        r"offsets ([\d.]+),\s*hash ([\d.]+),\s*masks ([\d.]+),\s*"
        r"MLPs ([\d.]+),\s*Total ([\d.]+)",
        text,
    )
    if not enc_matches:
        return None
    anchor, feat, scaling, offsets, hash_, masks, mlps, total = (
        float(v) for v in enc_matches[-1]
    )

    # Last anchor_num= line
    anchor_nums = re.findall(r"bits info: anchor_num=(\d+)", text)
    anchor_num = int(anchor_nums[-1]) if anchor_nums else None

    return {
        "anchor": anchor, "feat": feat, "scaling": scaling,
        "offsets": offsets, "hash": hash_, "masks": masks,
        "MLPs": mlps, "Total": total,
        "anchor_num": anchor_num,
    }


def parse_results_json(json_path: Path) -> dict | None:
    """Extract PSNR/SSIM/LPIPS from results.json."""
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    # Key is usually "ours_30000"
    entry = next(iter(data.values()))
    return {k: entry[k] for k in ("PSNR", "SSIM", "LPIPS") if k in entry}


def _has_lambda_dirs(d: Path) -> bool:
    return any(p.is_dir() and p.name.startswith("lambda_") for p in d.iterdir())


def _collect_scene(scene_dir: Path) -> dict:
    result = {}
    for lam_dir in sorted(
        (p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith("lambda_")),
        key=lambda p: float(p.name.replace("lambda_", "")),
    ):
        try:
            lmbda = float(lam_dir.name.replace("lambda_", ""))
        except ValueError:
            continue
        row = parse_outputs_log(lam_dir / "outputs.log") or {}
        metrics = parse_results_json(lam_dir / "results.json") or {}
        row.update(metrics)
        result[lmbda] = row
    return result


def collect_data(exp_dir: Path) -> dict:
    """
    Returns nested dict:
        data[dataset][scene][lambda_val] = {size+metric+anchor_num dict}

    Handles both 4-level (exp_dir/dataset/scene/lambda_X/) and
    3-level (exp_dir/scene/lambda_X/) directory structures.
    """
    top_dirs = sorted(p for p in exp_dir.iterdir() if p.is_dir())
    if not top_dirs:
        return {}

    # Detect structure: if any top-level dir directly contains lambda_* dirs,
    # this is a 3-level layout (scene dirs at top level, no dataset subdirectory).
    is_3level = any(_has_lambda_dirs(d) for d in top_dirs)

    data = {}
    if is_3level:
        dataset = exp_dir.name
        data[dataset] = {}
        for scene_dir in top_dirs:
            scene_data = _collect_scene(scene_dir)
            if scene_data:
                data[dataset][scene_dir.name] = scene_data
    else:
        for dataset_dir in top_dirs:
            scene_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
            dataset_data = {}
            for scene_dir in scene_dirs:
                scene_data = _collect_scene(scene_dir)
                if scene_data:
                    dataset_data[scene_dir.name] = scene_data
            if dataset_data:
                data[dataset_dir.name] = dataset_data

    return data


# ── Excel writing ─────────────────────────────────────────────────────────────

def _cell(ws, row, col, value=None, *, font=None, fill=None, align=None, border=None, num_fmt=None):
    c = ws.cell(row=row, column=col, value=value)
    if font:   c.font      = font
    if fill:   c.fill      = fill
    if align:  c.alignment = align
    if border: c.border    = border
    if num_fmt: c.number_format = num_fmt
    return c


def write_scene_block(ws, start_row: int, scene: str, lambdas: list[float],
                      scene_data: dict, fill_override=None) -> int:
    """
    Write one scene block starting at start_row.
    Returns the next available row after the block.
    """
    scene_fill = fill_override or SCENE_FILL
    n_cols = len(lambdas)

    # ── Row 1: scene name header ──
    ws.merge_cells(start_row=start_row, start_column=1,
                   end_row=start_row, end_column=1 + n_cols)
    _cell(ws, start_row, 1, scene,
          font=WHITE_FONT, fill=scene_fill,
          align=Alignment(horizontal="center"))

    # ── Row 2: λ header ──
    r = start_row + 1
    _cell(ws, r, 1, "λ", font=BOLD_FONT, fill=LAMBDA_FILL, border=FULL_BORDER)
    for j, lam in enumerate(lambdas, start=2):
        _cell(ws, r, j, lam, font=BOLD_FONT, fill=LAMBDA_FILL,
              border=FULL_BORDER, num_fmt="0.000")

    # ── Data rows ──
    for i, (size_key, label) in enumerate(zip(SIZE_KEYS, SIZE_LABELS)):
        r = start_row + 2 + i
        _cell(ws, r, 1, label, font=BOLD_FONT, fill=SIZE_FILL, border=FULL_BORDER)
        for j, lam in enumerate(lambdas, start=2):
            val = scene_data.get(lam, {}).get(size_key)
            _cell(ws, r, j, val, fill=SIZE_FILL, border=FULL_BORDER,
                  num_fmt="0.0000")

    metric_row_start = start_row + 2 + len(SIZE_KEYS)
    for i, key in enumerate(METRIC_KEYS):
        r = metric_row_start + i
        _cell(ws, r, 1, key, font=BOLD_FONT, fill=METRIC_FILL, border=FULL_BORDER)
        for j, lam in enumerate(lambdas, start=2):
            val = scene_data.get(lam, {}).get(key)
            _cell(ws, r, j, val, fill=METRIC_FILL, border=FULL_BORDER,
                  num_fmt="0.0000000")

    # anchor num row
    r = metric_row_start + len(METRIC_KEYS)
    _cell(ws, r, 1, ANCHOR_LABEL, font=BOLD_FONT, fill=SIZE_FILL, border=FULL_BORDER)
    for j, lam in enumerate(lambdas, start=2):
        val = scene_data.get(lam, {}).get("anchor_num")
        _cell(ws, r, j, val, fill=SIZE_FILL, border=FULL_BORDER, num_fmt="#,##0")

    return r + 2  # leave one blank row


def write_average_block(ws, start_row: int, lambdas: list[float],
                        dataset_data: dict) -> int:
    """Write an average block across all scenes for each lambda."""
    scenes = list(dataset_data.keys())
    avg_data = {}
    for lam in lambdas:
        avg_data[lam] = {}
        all_keys = SIZE_KEYS + METRIC_KEYS + ["anchor_num"]
        for key in all_keys:
            vals = [
                dataset_data[sc].get(lam, {}).get(key)
                for sc in scenes
                if dataset_data[sc].get(lam, {}).get(key) is not None
            ]
            avg_data[lam][key] = sum(vals) / len(vals) if vals else None

    return write_scene_block(ws, start_row, "Average", lambdas, avg_data,
                             fill_override=PatternFill("solid", fgColor="C55A11"))


def build_excel(data: dict, out_path: Path):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # remove default sheet

    for dataset, scenes in data.items():
        ws = wb.create_sheet(title=dataset[:31])  # sheet name max 31 chars

        lambdas = sorted({
            lam
            for scene_data in scenes.values()
            for lam in scene_data
        })
        n_cols = len(lambdas)

        # Column widths
        ws.column_dimensions["A"].width = 14
        for j in range(2, n_cols + 3):
            ws.column_dimensions[get_column_letter(j)].width = 14

        current_row = 1
        for scene, scene_data in scenes.items():
            current_row = write_scene_block(ws, current_row, scene, lambdas, scene_data)

        # Average block (only if >1 scene)
        if len(scenes) > 1:
            current_row = write_average_block(ws, current_row, lambdas, scenes)

    wb.save(out_path)
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    exp_dir = Path(sys.argv[1]).resolve()
    if not exp_dir.is_dir():
        print(f"Error: {exp_dir} is not a directory")
        sys.exit(1)

    out_path = Path(sys.argv[2]).resolve() if len(sys.argv) >= 3 else exp_dir / "results_summary.xlsx"

    print(f"Collecting data from: {exp_dir}")
    data = collect_data(exp_dir)

    datasets = list(data.keys())
    print(f"Found datasets: {datasets}")
    for ds, scenes in data.items():
        for sc, lams in scenes.items():
            print(f"  {ds}/{sc}: {sorted(lams.keys())}")

    build_excel(data, out_path)


if __name__ == "__main__":
    main()
