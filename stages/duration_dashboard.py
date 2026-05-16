"""
stages/duration_dashboard.py
-----------------------------
Generates duration statistics per tic type and group,
saves CSVs and an interactive HTML dashboard.

Outputs:
    outputs/duration_stats_per_group.csv
    outputs/duration_stats_per_type.csv
    outputs/duration_dashboard.html
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
import yaml

HOME_DIR = os.environ.get("HOME_DIR", "/home/kzaveri1/codes/modular_pipline_package/tic_detection")

with open(f'{HOME_DIR}/configs/paths.yaml') as f:
    paths_cfg = yaml.safe_load(f)

output_dir = Path(paths_cfg['output_dir'])

# -- load data --
df = pd.read_csv('/home/kzaveri1/codes/Vocal_tics/data/filtered_ticList_metadata.csv')
tic_groups = pd.read_csv(f'{HOME_DIR}/configs/tic_groups.csv')
df = df.merge(tic_groups, on='Type', how='left')

def compute_stats(grouped_df, groupby_col):
    summary = grouped_df.groupby(groupby_col)['Duration'].agg(
        count    = 'count',
        mean_s   = 'mean',
        std_s    = 'std',
        min_s    = 'min',
        q25_s    = lambda x: x.quantile(0.25),
        median_s = 'median',
        q75_s    = lambda x: x.quantile(0.75),
        max_s    = 'max',
    ).round(3)

    for col in ['mean_s','std_s','min_s','q25_s','median_s','q75_s','max_s']:
        summary[col.replace('_s','_frames')] = (
            summary[col].fillna(0) / 0.02
        ).round(0).astype(int)

    summary['pct_under_500ms'] = grouped_df.groupby(groupby_col).apply(
        lambda x: (x['Duration'] < 0.5).mean() * 100,
        include_groups=False
    ).round(1)

    return summary.sort_values('median_s').reset_index()

group_stats = compute_stats(df, 'group')
type_stats  = compute_stats(df, 'Type')

# add group name to type stats
type_to_group = dict(zip(tic_groups['Type'], tic_groups['group']))
type_stats['group'] = type_stats['Type'].map(type_to_group)

# save CSVs
group_stats.to_csv(output_dir / 'duration_stats_per_group.csv', index=False)
type_stats.to_csv(output_dir / 'duration_stats_per_type.csv', index=False)
print(f"[duration] CSVs saved to: {output_dir}")

# -- build dashboard --
group_json = group_stats.to_json(orient='records')
type_json  = type_stats.to_json(orient='records')

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tic Duration Statistics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {{
  --bg:#f7f6f3; --surface:#fff; --surface2:#f2f0ec;
  --border:#dddbd5; --border2:#c8c5bc;
  --text:#1a1916; --muted:#7a776e; --faint:#9e9b93;
  --blue:#2563a8; --orange:#c05621; --green:#276749;
  --blue-l:#dbeafe; --orange-l:#fde8d4; --green-l:#d1fae5;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);padding:2.5rem 3rem;max-width:1300px;margin:0 auto}}
header{{margin-bottom:2rem;padding-bottom:1rem;border-bottom:2px solid var(--text)}}
h1{{font-family:'Source Serif 4',serif;font-size:24px;font-weight:600}}
.meta{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-top:4px}}
.tabs{{display:flex;gap:0;border:1px solid var(--border2);border-radius:4px;overflow:hidden;margin-bottom:2rem;width:fit-content}}
.tab{{padding:7px 18px;font-size:12px;font-weight:500;border:none;border-right:1px solid var(--border2);cursor:pointer;background:var(--surface);color:var(--muted);transition:all 0.1s}}
.tab:last-child{{border-right:none}}
.tab.active{{background:var(--text);color:#fff}}
.tab:hover:not(.active){{background:var(--surface2)}}
.section-label{{font-size:10px;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;gap:8px}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1.5rem;margin-bottom:1.5rem}}
.chart-wrap{{position:relative;height:320px}}
.search-bar{{width:100%;background:var(--surface);border:1px solid var(--border2);border-radius:3px;padding:7px 12px;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:12px;margin-bottom:12px;outline:none}}
.search-bar:focus{{border-color:var(--text)}}
.search-bar::placeholder{{color:var(--faint)}}
table{{width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace}}
thead tr{{border-bottom:2px solid var(--text)}}
th{{text-align:left;font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);padding:6px 10px 8px}}
td{{padding:5px 10px;border-bottom:1px solid var(--border);color:var(--text);vertical-align:middle}}
tbody tr:hover td{{background:var(--surface2)}}
.bar-cell{{display:flex;align-items:center;gap:6px}}
.bar-bg{{flex:1;height:5px;background:var(--surface2)}}
.bar-fill{{height:100%;background:var(--blue)}}
.pct-high{{color:#dc2626;font-weight:500}}
.pct-low{{color:#276749}}
.badge{{display:inline-block;padding:1px 7px;border-radius:2px;font-size:10px;font-weight:500;border:1px solid;font-family:'IBM Plex Mono',monospace}}
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem}}
</style>
</head>
<body>
<header>
  <h1>Tic Duration Statistics</h1>
  <div class="meta">tic_detection · duration analysis by group and type · units: seconds</div>
</header>

<div class="tabs">
  <button class="tab active" onclick="switchTab('group')">By Group</button>
  <button class="tab" onclick="switchTab('type')">By Type</button>
</div>

<!-- GROUP VIEW -->
<div id="group-view">
  <div class="section-label">median duration by group</div>
  <div class="card">
    <div class="chart-wrap"><canvas id="group-chart"></canvas></div>
  </div>

  <div class="section-label">% events under 500ms by group</div>
  <div class="card">
    <div class="chart-wrap"><canvas id="short-chart"></canvas></div>
  </div>

  <div class="section-label">full group statistics</div>
  <div class="card">
    <input type="text" class="search-bar" placeholder="filter by group name..." oninput="filterTable('group-tbody', this.value)">
    <table>
      <thead><tr>
        <th>group</th><th>count</th>
        <th>median (s)</th><th>mean (s)</th><th>std (s)</th>
        <th>min (s)</th><th>max (s)</th>
        <th>median (frames)</th><th>% &lt;500ms</th>
      </tr></thead>
      <tbody id="group-tbody"></tbody>
    </table>
  </div>
</div>

<!-- TYPE VIEW -->
<div id="type-view" style="display:none">
  <div class="section-label">median duration by type (top 30)</div>
  <div class="card">
    <div class="chart-wrap"><canvas id="type-chart"></canvas></div>
  </div>

  <div class="section-label">full type statistics</div>
  <div class="card">
    <input type="text" class="search-bar" placeholder="filter by type or group..." oninput="filterTable('type-tbody', this.value)">
    <table>
      <thead><tr>
        <th>type</th><th>group</th><th>count</th>
        <th>median (s)</th><th>mean (s)</th><th>std (s)</th>
        <th>min (s)</th><th>max (s)</th>
        <th>median (frames)</th><th>% &lt;500ms</th>
      </tr></thead>
      <tbody id="type-tbody"></tbody>
    </table>
  </div>
</div>

<script>
const GROUP_DATA = {group_json};
const TYPE_DATA  = {type_json};

const COLORS = {{
  train:'#2563a8', val:'#c05621', test:'#276749',
}};

function switchTab(tab) {{
  document.querySelectorAll('.tab').forEach((t,i) => {{
    t.classList.toggle('active', (tab==='group'&&i===0)||(tab==='type'&&i===1));
  }});
  document.getElementById('group-view').style.display = tab==='group' ? '' : 'none';
  document.getElementById('type-view').style.display  = tab==='type'  ? '' : 'none';
}}

function fmt(n) {{ return typeof n === 'number' ? n.toFixed(3) : n; }}

function pctClass(v) {{
  if (v >= 15) return 'pct-high';
  if (v <= 5)  return 'pct-low';
  return '';
}}

function filterTable(tbodyId, q) {{
  q = q.toLowerCase();
  document.querySelectorAll(`#${{tbodyId}} tr`).forEach(tr => {{
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
  }});
}}

// -- group table --
const maxMedian = Math.max(...GROUP_DATA.map(d => d.median_s));
document.getElementById('group-tbody').innerHTML = GROUP_DATA.map(d => `
  <tr>
    <td>${{d.group}}</td>
    <td>${{d.count}}</td>
    <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${{(d.median_s/maxMedian*100).toFixed(0)}}%"></div></div>${{fmt(d.median_s)}}</div></td>
    <td>${{fmt(d.mean_s)}}</td>
    <td>${{fmt(d.std_s)}}</td>
    <td>${{fmt(d.min_s)}}</td>
    <td>${{fmt(d.max_s)}}</td>
    <td>${{d.median_frames}}</td>
    <td class="${{pctClass(d.pct_under_500ms)}}">${{d.pct_under_500ms}}%</td>
  </tr>
`).join('');

// -- type table --
const maxTypeMedian = Math.max(...TYPE_DATA.map(d => d.median_s));
document.getElementById('type-tbody').innerHTML = TYPE_DATA.map(d => `
  <tr>
    <td>${{d.Type}}</td>
    <td>${{d.group || ''}}</td>
    <td>${{d.count}}</td>
    <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${{(d.median_s/maxTypeMedian*100).toFixed(0)}}%"></div></div>${{fmt(d.median_s)}}</div></td>
    <td>${{fmt(d.mean_s)}}</td>
    <td>${{fmt(d.std_s)}}</td>
    <td>${{fmt(d.min_s)}}</td>
    <td>${{fmt(d.max_s)}}</td>
    <td>${{d.median_frames}}</td>
    <td class="${{pctClass(d.pct_under_500ms)}}">${{d.pct_under_500ms}}%</td>
  </tr>
`).join('');

// -- group median chart --
const groupLabels  = GROUP_DATA.map(d => d.group);
const groupMedians = GROUP_DATA.map(d => d.median_s);
const groupQ25     = GROUP_DATA.map(d => d.q25_s);
const groupQ75     = GROUP_DATA.map(d => d.q75_s);

new Chart(document.getElementById('group-chart'), {{
  type: 'bar',
  data: {{
    labels: groupLabels,
    datasets: [
      {{
        label: 'Median duration (s)',
        data: groupMedians,
        backgroundColor: 'rgba(37,99,168,0.7)',
        borderColor: '#2563a8',
        borderWidth: 1,
      }},
      {{
        label: 'Q25 duration (s)',
        data: groupQ25,
        backgroundColor: 'rgba(39,103,73,0.5)',
        borderColor: '#276749',
        borderWidth: 1,
      }},
      {{
        label: 'Q75 duration (s)',
        data: groupQ75,
        backgroundColor: 'rgba(192,86,33,0.5)',
        borderColor: '#c05621',
        borderWidth: 1,
      }},
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{labels: {{color:'#7a776e', font:{{family:'IBM Plex Mono',size:11}}}}}},
      tooltip: {{backgroundColor:'#fff',borderColor:'#dddbd5',borderWidth:1,titleColor:'#1a1916',bodyColor:'#7a776e',padding:10,titleFont:{{family:'IBM Plex Mono',size:11}},bodyFont:{{family:'IBM Plex Mono',size:11}}}}
    }},
    scales: {{
      x: {{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:9}},maxRotation:45}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}}}},
      y: {{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:10}}}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}},title:{{display:true,text:'Duration (s)',color:'#7a776e',font:{{family:'IBM Plex Mono',size:10}}}}}}
    }}
  }}
}});

// -- short tic % chart --
new Chart(document.getElementById('short-chart'), {{
  type: 'bar',
  data: {{
    labels: groupLabels,
    datasets: [{{
      label: '% events under 500ms',
      data: GROUP_DATA.map(d => d.pct_under_500ms),
      backgroundColor: GROUP_DATA.map(d => d.pct_under_500ms >= 15 ? 'rgba(220,38,38,0.7)' : d.pct_under_500ms <= 5 ? 'rgba(39,103,73,0.7)' : 'rgba(192,86,33,0.7)'),
      borderColor: GROUP_DATA.map(d => d.pct_under_500ms >= 15 ? '#dc2626' : d.pct_under_500ms <= 5 ? '#276749' : '#c05621'),
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{labels: {{color:'#7a776e', font:{{family:'IBM Plex Mono',size:11}}}}}},
      tooltip: {{backgroundColor:'#fff',borderColor:'#dddbd5',borderWidth:1,titleColor:'#1a1916',bodyColor:'#7a776e',padding:10,titleFont:{{family:'IBM Plex Mono',size:11}},bodyFont:{{family:'IBM Plex Mono',size:11}}}}
    }},
    scales: {{
      x: {{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:9}},maxRotation:45}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}}}},
      y: {{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:10}}}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}},title:{{display:true,text:'% events under 500ms',color:'#7a776e',font:{{family:'IBM Plex Mono',size:10}}}}}}
    }}
  }}
}});

// -- type chart (top 30 by count) --
const top30 = [...TYPE_DATA].sort((a,b) => b.count - a.count).slice(0, 30);
new Chart(document.getElementById('type-chart'), {{
  type: 'bar',
  data: {{
    labels: top30.map(d => `${{d.Type}} (${{d.group||''}})`),
    datasets: [{{
      label: 'Median duration (s)',
      data: top30.map(d => d.median_s),
      backgroundColor: 'rgba(37,99,168,0.7)',
      borderColor: '#2563a8',
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{
      legend: {{labels: {{color:'#7a776e', font:{{family:'IBM Plex Mono',size:11}}}}}},
      tooltip: {{backgroundColor:'#fff',borderColor:'#dddbd5',borderWidth:1,titleColor:'#1a1916',bodyColor:'#7a776e',padding:10,titleFont:{{family:'IBM Plex Mono',size:11}},bodyFont:{{family:'IBM Plex Mono',size:11}}}}
    }},
    scales: {{
      x: {{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:9}},maxRotation:45}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}}}},
      y: {{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:10}}}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}},title:{{display:true,text:'Duration (s)',color:'#7a776e',font:{{family:'IBM Plex Mono',size:10}}}}}}
    }}
  }}
}});
</script>
</body>
</html>"""

html_path = output_dir / 'duration_dashboard.html'
html_path.write_text(html)
print(f"[duration] Dashboard saved to: {html_path}")
