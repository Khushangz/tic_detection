"""
summarize_experiments.py
------------------------
Collects results from all experiments and produces a summary table.

Reads:
    outputs/runs/{exp_name}/{eval_dir}/results.json
    outputs/runs/{exp_name}/config.json

Outputs:
    outputs/experiment_summary_{eval_dir}.csv
    outputs/experiment_summary_{eval_dir}.html

Usage:
    python summarize_experiments.py                        # reads from eval/
    python summarize_experiments.py --eval-dir eval_no_vote
"""

import os
import json
import sys
import argparse
from pathlib import Path
import pandas as pd
import yaml

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)

parser = argparse.ArgumentParser()
parser.add_argument("--eval-dir", type=str, default="eval", dest="eval_dir",
                    help="eval subdirectory to read results from (default: eval)")
args = parser.parse_args()

with open(os.path.join(HOME_DIR, "configs", "paths.yaml")) as f:
    paths_cfg = yaml.safe_load(f)

output_dir = Path(paths_cfg["output_dir"])
runs_dir   = output_dir / "runs"

print(f"[summary] Reading from: {args.eval_dir}/")

rows = []

for exp_dir in sorted(runs_dir.iterdir()):
    if not exp_dir.is_dir():
        continue

    exp_name = exp_dir.name

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        print(f"[summary] ⚠️  No config.json for {exp_name}, skipping")
        continue

    with open(config_path) as f:
        cfg = json.load(f)

    results_path = exp_dir / args.eval_dir / "results.json"
    if not results_path.exists():
        print(f"[summary] ⚠️  No {args.eval_dir}/results.json for {exp_name}, skipping")
        continue

    with open(results_path) as f:
        results = json.load(f)

    metrics_path = exp_dir / "metrics.json"
    train_metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            train_metrics = json.load(f)

    m   = results["metrics"]
    row = {
        "experiment":       exp_name,
        "model":            cfg.get("model_type", "bilstm"),
        "loss":             cfg.get("loss", "multiclass"),
        "num_layers":       cfg.get("bilstm", {}).get("num_layers", "?"),
        "hidden_size":      cfg.get("bilstm", {}).get("hidden_size", "?"),
        "dropout":          cfg.get("bilstm", {}).get("dropout", "?"),
        "lr":               cfg.get("lr", "?"),
        "sampler":          cfg.get("imbalance_strategy", "?"),
        "epochs_trained":   train_metrics.get("best_epoch", "?"),
        "val_auroc_best":   train_metrics.get("best_val_auroc", "?"),
        "test_bin_auroc":   m.get("binary_auroc", "?"),
        "test_bin_f1":      m.get("binary_f1", "?"),
        "test_bin_prec":    m.get("binary_precision", "?"),
        "test_bin_recall":  m.get("binary_recall", "?"),
        "test_mc_auroc":    m.get("mc_auroc", "?"),
        "test_mc_f1":       m.get("mc_f1", "?"),
    }
    rows.append(row)

if not rows:
    print("[summary] No completed experiments found.")
    sys.exit(0)

df = pd.DataFrame(rows)
df = df.sort_values("test_bin_auroc", ascending=False).reset_index(drop=True)

# save CSV
csv_path = output_dir / f"experiment_summary_{args.eval_dir}.csv"
df.to_csv(csv_path, index=False)
print(f"[summary] CSV saved to: {csv_path}")

# save HTML
html_path = output_dir / f"experiment_summary_{args.eval_dir}.html"

def highlight_best(df):
    styled = df.style\
        .highlight_max(subset=["test_bin_auroc","test_bin_f1","test_mc_auroc","test_mc_f1","val_auroc_best"],
                       color="#d1fae5")\
        .highlight_min(subset=["test_bin_auroc","test_bin_f1","test_mc_auroc","test_mc_f1"],
                       color="#fee2e2")\
        .format({
            "lr":              lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            "test_bin_auroc":  lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            "test_bin_f1":     lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            "test_bin_prec":   lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            "test_bin_recall": lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            "test_mc_auroc":   lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            "test_mc_f1":      lambda x: f"{x:.4f}" if isinstance(x, float) else x,
            "val_auroc_best":  lambda x: f"{x:.4f}" if isinstance(x, float) else x,
        })\
        .set_table_styles([
            {"selector": "thead th", "props": [
                ("background-color", "#1a1916"),
                ("color", "#ffffff"),
                ("font-family", "IBM Plex Mono, monospace"),
                ("font-size", "11px"),
                ("padding", "8px 12px"),
                ("text-align", "left"),
            ]},
            {"selector": "tbody td", "props": [
                ("font-family", "IBM Plex Mono, monospace"),
                ("font-size", "11px"),
                ("padding", "6px 12px"),
                ("border-bottom", "1px solid #e5e7eb"),
            ]},
            {"selector": "tbody tr:hover td", "props": [
                ("background-color", "#f9fafb"),
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("width", "100%"),
            ]},
        ])
    return styled

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Experiment Summary — {args.eval_dir}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;600&display=swap');
body {{
    font-family: 'IBM Plex Sans', sans-serif;
    background: #f7f6f3;
    padding: 2rem 3rem;
    max-width: 1400px;
    margin: 0 auto;
}}
h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 0.25rem; color: #1a1916; }}
.meta {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #7a776e; margin-bottom: 2rem; }}
.table-wrap {{ background: white; border: 1px solid #dddbd5; border-radius: 4px; overflow-x: auto; }}
.legend {{ margin-top: 1rem; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #7a776e; display: flex; gap: 1.5rem; }}
.legend span {{ display: flex; align-items: center; gap: 6px; }}
.dot {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
</style>
</head>
<body>
<h1>Experiment Summary — {args.eval_dir}</h1>
<div class="meta">tic_detection · {len(df)} experiments · sorted by test binary AUROC · voting={'enabled' if args.eval_dir == 'eval' else 'disabled'}</div>
<div class="table-wrap">
{highlight_best(df).to_html()}
</div>
<div class="legend">
    <span><span class="dot" style="background:#d1fae5"></span> Best in column</span>
    <span><span class="dot" style="background:#fee2e2"></span> Worst in column</span>
</div>
</body>
</html>"""

html_path.write_text(html_content)
print(f"[summary] HTML saved to: {html_path}")
print(f"\n{'═'*80}")
print(f"EXPERIMENT SUMMARY — {args.eval_dir} ({len(df)} experiments)")
print(f"{'═'*80}")
print(df.to_string(index=False))