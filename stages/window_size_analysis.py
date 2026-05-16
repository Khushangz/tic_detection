"""
stages/window_size_perf.py
--------------------------
Per-group AUROC across sequence lengths from seq experiments.

Reads per_group_metrics.csv from each cnn_2l_bin_seq* experiment
and produces:
    outputs/viz/window_size_perf_{eval_dir}.csv
    outputs/viz/window_size_perf_{eval_dir}.html
    outputs/viz/window_size_perf_{eval_dir}.png
    outputs/viz/window_size_trends_{eval_dir}.csv

Usage:
    python window_size_perf.py                        # reads from eval/
    python window_size_perf.py --eval-dir eval_no_vote
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)

parser = argparse.ArgumentParser()
parser.add_argument("--eval-dir", type=str, default="eval", dest="eval_dir",
                    help="eval subdirectory to read per_group_metrics.csv from")
args = parser.parse_args()

with open(f'{HOME_DIR}/configs/paths.yaml') as f:
    paths_cfg = yaml.safe_load(f)

output_dir = Path(paths_cfg['output_dir'])
viz_dir    = output_dir / 'viz'
viz_dir.mkdir(parents=True, exist_ok=True)

print(f"[window] Reading from: {args.eval_dir}/")

# -- collect data --
runs_dir = output_dir / 'runs'
records  = []

for exp_dir in sorted(runs_dir.iterdir()):
    if not exp_dir.name.startswith('cnn_2l_bin_seq'):
        continue

    metrics_path = exp_dir / args.eval_dir / 'per_group_metrics.csv'
    config_path  = exp_dir / 'config.json'

    if not metrics_path.exists() or not config_path.exists():
        continue

    with open(config_path) as f:
        cfg = json.load(f)

    seq_len = cfg.get('sequence_length', None)
    if seq_len is None:
        continue

    df = pd.read_csv(metrics_path)
    for _, row in df.iterrows():
        records.append({
            'exp':        exp_dir.name,
            'seq_len':    int(seq_len),
            'group':      row['group'],
            'auroc':      row['auroc'],
            'f1':         row['f1'],
            'n_frames':   row['n_frames'],
            'n_positive': row['n_positive'],
        })

if not records:
    print(f"[window] No data found in {args.eval_dir}/per_group_metrics.csv")
    sys.exit(0)

df = pd.DataFrame(records).sort_values(['group', 'seq_len'])

# save CSV
csv_path = viz_dir / f'window_size_perf_{args.eval_dir}.csv'
df.to_csv(csv_path, index=False)
print(f"[window] Data saved: {csv_path}")
print(f"[window] {df['group'].nunique()} groups, {df['seq_len'].nunique()} seq lengths")

seq_lens = [int(s) for s in sorted(df['seq_len'].unique())]
groups   = sorted(df['group'].unique())

# -- static matplotlib plot --
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})

colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))

fig, ax = plt.subplots(figsize=(12, 7))

for i, group in enumerate(groups):
    gdf = df[df['group'] == group].sort_values('seq_len')
    if len(gdf) < 2:
        continue
    ax.plot(
        gdf['seq_len'], gdf['auroc'],
        marker='o', markersize=5,
        linewidth=1.5, color=colors[i],
        label=group, alpha=0.85
    )

ax.axhline(0.5, color='#9e9b93', linewidth=1, linestyle=':', label='random (0.5)')
ax.set_xlabel('Sequence Length (frames)', fontsize=11)
ax.set_ylabel('Test AUROC', fontsize=11)
ax.set_title(
    f'Per-Group AUROC vs Sequence Length\nCNN+BiLSTM, binary loss, undersample · {args.eval_dir}',
    fontsize=12
)
ax.set_xscale('log')
ax.set_xticks(seq_lens)
ax.set_xticklabels([str(s) for s in seq_lens])
ax.set_ylim(0.0, 1.0)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8, frameon=False)
fig.tight_layout()

png_path = viz_dir / f'window_size_perf_{args.eval_dir}.png'
fig.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"[window] PNG saved: {png_path}")

# -- group data for HTML --
group_data = {}
for group in groups:
    gdf = df[df['group'] == group].sort_values('seq_len')
    group_data[group] = {
        'seq_lens':   gdf['seq_len'].tolist(),
        'auroc':      gdf['auroc'].tolist(),
        'f1':         gdf['f1'].tolist(),
        'n_positive': gdf['n_positive'].tolist(),
    }

# -- compute trends --
trends = []
for group in groups:
    gdf = df[df['group'] == group].sort_values('seq_len')
    if len(gdf) >= 3:
        x     = np.log(gdf['seq_len'].values.astype(float))
        y     = gdf['auroc'].values.astype(float)
        slope = float(np.polyfit(x, y, 1)[0])
        trends.append({
            'group':        group,
            'slope':        round(slope, 4),
            'mean_auroc':   round(float(y.mean()), 4),
            'max_auroc':    round(float(y.max()), 4),
            'best_seq_len': int(gdf.loc[gdf['auroc'].idxmax(), 'seq_len']),
        })

trends_df   = pd.DataFrame(trends).sort_values('mean_auroc', ascending=False)
trends_path = viz_dir / f'window_size_trends_{args.eval_dir}.csv'
trends_df.to_csv(trends_path, index=False)
print(f"[window] Trends saved: {trends_path}")
print(f"\n[window] Group trends (sorted by mean AUROC):")
print(trends_df.to_string(index=False))

# -- interactive HTML --
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Window Size Performance — {args.eval_dir}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{{--bg:#f7f6f3;--surface:#fff;--surface2:#f2f0ec;--border:#dddbd5;--border2:#c8c5bc;--text:#1a1916;--muted:#7a776e;--faint:#9e9b93}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);padding:2.5rem 3rem;max-width:1300px;margin:0 auto}}
header{{margin-bottom:2rem;padding-bottom:1rem;border-bottom:2px solid var(--text)}}
h1{{font-family:'Source Serif 4',serif;font-size:24px;font-weight:600}}
.meta{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-top:4px}}
.section-label{{font-size:10px;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;gap:8px}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1.5rem;margin-bottom:1.5rem}}
.chart-wrap{{position:relative;height:480px}}
.controls{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1rem}}
.btn{{padding:4px 12px;font-size:11px;font-family:'IBM Plex Mono',monospace;border:1px solid var(--border2);border-radius:3px;cursor:pointer;background:var(--surface);color:var(--muted)}}
.btn:hover{{background:var(--surface2)}}
.btn.active{{background:var(--text);color:#fff;border-color:var(--text)}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace}}
thead tr{{border-bottom:2px solid var(--text)}}
th{{text-align:left;font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);padding:6px 10px 8px}}
td{{padding:5px 10px;border-bottom:1px solid var(--border);color:var(--text)}}
tbody tr:hover td{{background:var(--surface2)}}
.trend-pos{{color:#276749;font-weight:500}}
.trend-neg{{color:#dc2626;font-weight:500}}
.trend-flat{{color:#7a776e}}
</style>
</head>
<body>
<header>
  <h1>Per-Group AUROC vs Sequence Length</h1>
  <div class="meta">CNN+BiLSTM · 2 layers · binary loss · lr=0.0001 · dropout=0.5 · undersample · {args.eval_dir}</div>
</header>

<div class="section-label">auroc by sequence length</div>
<div class="card">
  <div class="controls">
    <button class="btn active" onclick="toggleAll(true)">Show all</button>
    <button class="btn" onclick="toggleAll(false)">Hide all</button>
  </div>
  <div class="chart-wrap"><canvas id="main-chart"></canvas></div>
</div>

<div class="section-label">group trends (slope of auroc vs log seq_len)</div>
<div class="card">
  <table>
    <thead><tr>
      <th>group</th><th>mean auroc</th><th>max auroc</th>
      <th>best seq len</th><th>trend (slope)</th><th>interpretation</th>
    </tr></thead>
    <tbody id="trends-tbody"></tbody>
  </table>
</div>

<script>
const GROUP_DATA = {json.dumps(group_data)};
const SEQ_LENS   = {json.dumps(seq_lens)};
const TRENDS     = {json.dumps(trends_df.to_dict(orient='records'))};

const PALETTE = [
  '#2563a8','#c05621','#276749','#7c3aed','#db2777',
  '#0891b2','#ea580c','#16a34a','#dc2626','#9333ea',
  '#b45309','#0284c7','#15803d','#e11d48','#0e7490',
  '#65a30d','#d97706','#6366f1',
];

const groups   = Object.keys(GROUP_DATA);
const datasets = groups.map((g,i) => ({{
  label: g,
  data: SEQ_LENS.map(s => {{
    const idx = GROUP_DATA[g].seq_lens.indexOf(s);
    return idx >= 0 ? GROUP_DATA[g].auroc[idx] : null;
  }}),
  borderColor:     PALETTE[i % PALETTE.length],
  backgroundColor: PALETTE[i % PALETTE.length] + '22',
  borderWidth: 2, pointRadius: 5, pointHoverRadius: 7,
  tension: 0.3, fill: false,
}}));

const chart = new Chart(document.getElementById('main-chart'), {{
  type: 'line',
  data: {{ labels: SEQ_LENS, datasets }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{
        position: 'right',
        labels: {{color:'#7a776e',font:{{family:'IBM Plex Mono',size:10}},padding:8,usePointStyle:true}}
      }},
      tooltip: {{
        backgroundColor:'#fff',borderColor:'#dddbd5',borderWidth:1,
        titleColor:'#1a1916',bodyColor:'#7a776e',padding:10,
        titleFont:{{family:'IBM Plex Mono',size:11}},
        bodyFont:{{family:'IBM Plex Mono',size:11}},
      }}
    }},
    scales: {{
      x: {{
        type: 'logarithmic',
        title: {{display:true,text:'Sequence Length (frames)',color:'#7a776e',font:{{family:'IBM Plex Mono',size:11}}}},
        ticks: {{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:10}},
          callback: v => SEQ_LENS.includes(Number(v)) ? v : ''}},
        grid: {{color:'#f2f0ec'}}, border: {{color:'#dddbd5'}},
      }},
      y: {{
        min: 0.0, max: 1.0,
        title: {{display:true,text:'Test AUROC',color:'#7a776e',font:{{family:'IBM Plex Mono',size:11}}}},
        ticks: {{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:10}}}},
        grid: {{color:'#f2f0ec'}}, border: {{color:'#dddbd5'}},
      }}
    }}
  }}
}});

function toggleAll(show) {{
  chart.data.datasets.forEach(ds => ds.hidden = !show);
  chart.update();
  document.querySelectorAll('.btn').forEach((b,i) => {{
    b.classList.toggle('active', show ? i===0 : i===1);
  }});
}}

document.getElementById('trends-tbody').innerHTML = TRENDS.map(t => {{
  const sc = t.slope > 0.01 ? 'trend-pos' : t.slope < -0.01 ? 'trend-neg' : 'trend-flat';
  const interp = t.slope > 0.01 ? 'needs more context' : t.slope < -0.01 ? 'shorter is better' : 'context-independent';
  return `<tr>
    <td>${{t.group}}</td>
    <td>${{t.mean_auroc.toFixed(4)}}</td>
    <td>${{t.max_auroc.toFixed(4)}}</td>
    <td>${{t.best_seq_len}}</td>
    <td class="${{sc}}">${{t.slope >= 0 ? '+' : ''}}${{t.slope.toFixed(4)}}</td>
    <td class="${{sc}}">${{interp}}</td>
  </tr>`;
}}).join('');
</script>
</body>
</html>"""

html_path = viz_dir / f'window_size_perf_{args.eval_dir}.html'
html_path.write_text(html)
print(f"[window] HTML saved: {html_path}")
print(f"[window] Done.")