import json
import os
import sys
from pathlib import Path
import yaml

HOME_DIR = os.environ.get(
    "HOME_DIR",
    "/home/kzaveri1/codes/modular_pipline_package/tic_detection"
)

def _generate_dashboard(file_split_path, patient_split_path, output_path):
    with open(file_split_path) as f:
        file_data = json.load(f)
    with open(patient_split_path) as f:
        patient_data = json.load(f)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Split Analysis</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,wght@0,300;0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
:root {{
  --bg:#f7f6f3;
  --surface:#ffffff;
  --surface2:#f2f0ec;
  --border:#dddbd5;
  --border2:#c8c5bc;
  --text:#1a1916;
  --muted:#7a776e;
  --faint:#9e9b93;
  --train:#2563a8;
  --val:#c05621;
  --test:#276749;
  --train-light:#dbeafe;
  --val-light:#fde8d4;
  --test-light:#d1fae5;
  --accent:#2563a8;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'IBM Plex Sans',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;padding:2.5rem 3rem;max-width:1200px;margin:0 auto}}
header{{margin-bottom:2.5rem;padding-bottom:1.25rem;border-bottom:2px solid var(--text)}}
.header-top{{display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:0.5rem}}
h1{{font-family:'Source Serif 4',serif;font-size:26px;font-weight:600;letter-spacing:-0.01em;color:var(--text)}}
.header-meta{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--muted);margin-top:4px}}
.split-tabs{{display:flex;gap:0;border:1px solid var(--border2);border-radius:4px;overflow:hidden}}
.tab{{padding:7px 18px;font-size:12px;font-weight:500;font-family:'IBM Plex Sans',sans-serif;border:none;border-right:1px solid var(--border2);cursor:pointer;background:var(--surface);color:var(--muted);transition:all 0.1s;letter-spacing:0.02em}}
.tab:last-child{{border-right:none}}
.tab.active{{background:var(--text);color:#fff}}
.tab:hover:not(.active){{background:var(--surface2)}}

.metrics-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--border);border:1px solid var(--border);border-radius:4px;overflow:hidden;margin-bottom:2rem}}
.metric-card{{background:var(--surface);padding:1.25rem 1.5rem}}
.metric-label{{font-size:10px;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:10px;font-family:'IBM Plex Mono',monospace}}
.metric-value{{font-size:26px;font-weight:300;font-family:'Source Serif 4',serif;letter-spacing:-0.02em;color:var(--text)}}
.metric-card.train .metric-value{{color:var(--train)}}
.metric-card.val   .metric-value{{color:var(--val)}}
.metric-card.test  .metric-value{{color:var(--test)}}
.metric-sub{{font-size:11px;color:var(--muted);margin-top:6px;font-family:'IBM Plex Mono',monospace}}
.metric-rule{{width:24px;height:2px;margin-bottom:10px}}
.metric-card.train .metric-rule{{background:var(--train)}}
.metric-card.val   .metric-rule{{background:var(--val)}}
.metric-card.test  .metric-rule{{background:var(--test)}}
.metric-card.total .metric-rule{{background:var(--text)}}

.section-label{{font-size:10px;font-weight:500;letter-spacing:0.1em;text-transform:uppercase;color:var(--muted);margin-bottom:1rem;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;gap:8px}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--border)}}

.grid-2{{display:grid;grid-template-columns:1fr 1.4fr;gap:1.5rem;margin-bottom:2rem}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1.5rem}}
.chart-wrap{{position:relative;height:240px}}

.group-table{{width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace}}
.group-table thead tr{{border-bottom:2px solid var(--text)}}
.group-table th{{text-align:left;font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);padding:6px 10px 8px}}
.group-table th.train{{color:var(--train)}}
.group-table th.val  {{color:var(--val)}}
.group-table th.test {{color:var(--test)}}
.group-table td{{padding:5px 10px;border-bottom:1px solid var(--border);color:var(--text);vertical-align:middle}}
.group-table tr:last-child td{{border-bottom:none}}
.group-table tbody tr:hover td{{background:var(--surface2)}}
.group-name{{font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:400}}
.bar-cell{{display:flex;align-items:center;gap:6px}}
.bar-bg{{flex:1;height:5px;background:var(--surface2);border-radius:0}}
.bar-fill{{height:100%}}
.bar-fill.train{{background:var(--train)}}
.bar-fill.val  {{background:var(--val)}}
.bar-fill.test {{background:var(--test)}}
.bar-num{{font-size:11px;color:var(--muted);min-width:52px;text-align:right}}

.full-card{{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1.5rem;margin-bottom:2rem}}
.table-scroll{{overflow-x:auto}}
.file-table{{width:100%;border-collapse:collapse;font-size:12px;font-family:'IBM Plex Mono',monospace}}
.file-table thead tr{{border-bottom:2px solid var(--text)}}
.file-table th{{text-align:left;font-size:10px;font-weight:500;letter-spacing:0.08em;text-transform:uppercase;color:var(--muted);padding:6px 12px 8px;white-space:nowrap}}
.file-table td{{padding:5px 12px;border-bottom:1px solid var(--border);color:var(--text);white-space:nowrap}}
.file-table tr:last-child td{{border-bottom:none}}
.file-table tbody tr:hover td{{background:var(--surface2)}}
.badge{{display:inline-block;padding:1px 7px;border-radius:2px;font-size:10px;font-weight:500;letter-spacing:0.05em;font-family:'IBM Plex Mono',monospace;border:1px solid}}
.badge.train{{background:var(--train-light);color:var(--train);border-color:var(--train)}}
.badge.val  {{background:var(--val-light);  color:var(--val);  border-color:var(--val)}}
.badge.test {{background:var(--test-light); color:var(--test); border-color:var(--test)}}
.search-bar{{width:100%;background:var(--surface);border:1px solid var(--border2);border-radius:3px;padding:7px 12px;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:12px;margin-bottom:12px;outline:none;transition:border-color 0.1s}}
.search-bar:focus{{border-color:var(--text)}}
.search-bar::placeholder{{color:var(--faint)}}
</style>
</head>
<body>
<header>
  <div class="header-top">
    <div>
      <h1>Train / Val / Test Split Analysis</h1>
      <div class="header-meta">tic_detection · vocal tics dataset · jsd-optimized greedy search</div>
    </div>
    <div class="split-tabs">
      <button class="tab active" onclick="switchSplit('file')">File split</button>
      <button class="tab" onclick="switchSplit('patient')">Patient split</button>
    </div>
  </div>
</header>

<div class="metrics-row">
  <div class="metric-card train">
    <div class="metric-rule"></div>
    <div class="metric-label">train</div>
    <div class="metric-value" id="train-files">—</div>
    <div class="metric-sub" id="train-frames">—</div>
    <div class="metric-sub" id="train-pct">—</div>
  </div>
  <div class="metric-card val">
    <div class="metric-rule"></div>
    <div class="metric-label">validation</div>
    <div class="metric-value" id="val-files">—</div>
    <div class="metric-sub" id="val-frames">—</div>
  </div>
  <div class="metric-card test">
    <div class="metric-rule"></div>
    <div class="metric-label">test</div>
    <div class="metric-value" id="test-files">—</div>
    <div class="metric-sub" id="test-frames">—</div>
  </div>
  <div class="metric-card total">
    <div class="metric-rule"></div>
    <div class="metric-label">avg jsd</div>
    <div class="metric-value" id="jsd-val">—</div>
    <div class="metric-sub">lower is better</div>
  </div>
</div>

<div class="section-label">distribution overview</div>
<div class="grid-2">
  <div class="card">
    <div class="chart-wrap"><canvas id="pie-chart"></canvas></div>
  </div>
  <div class="card">
    <div class="chart-wrap"><canvas id="bar-chart"></canvas></div>
  </div>
</div>

<div class="section-label">tic group breakdown</div>
<div class="full-card">
  <div id="group-table-wrap"></div>
</div>

<div class="section-label">file assignments</div>
<div class="full-card">
  <input type="text" class="search-bar" placeholder="filter by filename, ID, or split..." oninput="filterFiles(this.value)">
  <div class="table-scroll">
    <table class="file-table">
      <thead><tr><th>filename</th><th>split</th><th>patient ID</th><th>session</th><th>phase</th></tr></thead>
      <tbody id="file-tbody"></tbody>
    </table>
  </div>
</div>

<script>
const SPLITS = {{
  file:    {json.dumps(file_data)},
  patient: {json.dumps(patient_data)}
}};

const COLORS = {{
  train:'#2563a8', val:'#c05621', test:'#276749',
  trainA:'rgba(37,99,168,0.75)', valA:'rgba(192,86,33,0.75)', testA:'rgba(39,103,73,0.75)',
  trainL:'rgba(37,99,168,0.12)', valL:'rgba(192,86,33,0.12)', testL:'rgba(39,103,73,0.12)',
}};

let pieChart=null, barChart=null, allFileRows=[];

function switchSplit(type) {{
  document.querySelectorAll('.tab').forEach((t,i)=>{{
    t.classList.toggle('active',(type==='file'&&i===0)||(type==='patient'&&i===1));
  }});
  renderDashboard(SPLITS[type]);
}}

function fmt(n){{ return n.toLocaleString(); }}

function renderDashboard(data) {{
  document.getElementById('train-files').textContent  = data.train_files.length + ' files';
  document.getElementById('val-files').textContent    = data.val_files.length   + ' files';
  document.getElementById('test-files').textContent   = data.test_files.length  + ' files';
  document.getElementById('train-frames').textContent = fmt(data.train_frames)  + ' frames';
  document.getElementById('val-frames').textContent   = fmt(data.val_frames)    + ' frames';
  document.getElementById('test-frames').textContent  = fmt(data.test_frames)   + ' frames';
  document.getElementById('train-pct').textContent    = data.train_pct          + '% of tic frames';
  document.getElementById('jsd-val').textContent      = data.avg_jsd.toFixed(4);

  const total = data.train_frames + data.val_frames + data.test_frames;

  if(pieChart) pieChart.destroy();
  pieChart = new Chart(document.getElementById('pie-chart'),{{
    type:'doughnut',
    data:{{
      labels:['Train','Validation','Test'],
      datasets:[{{
        data:[data.train_frames,data.val_frames,data.test_frames],
        backgroundColor:[COLORS.train,COLORS.val,COLORS.test],
        borderColor:'#ffffff', borderWidth:3, hoverOffset:4,
      }}]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,cutout:'62%',
      plugins:{{
        legend:{{position:'bottom',labels:{{color:'#7a776e',font:{{family:'IBM Plex Mono',size:11}},padding:16,usePointStyle:true,pointStyleWidth:8}}}},
        tooltip:{{
          callbacks:{{label:ctx=>` ${{ctx.label}}: ${{fmt(ctx.raw)}} (${{(ctx.raw/total*100).toFixed(1)}}%)`}},
          backgroundColor:'#fff',borderColor:'#dddbd5',borderWidth:1,
          titleColor:'#1a1916',bodyColor:'#7a776e',padding:10,
          titleFont:{{family:'IBM Plex Mono',size:11}},bodyFont:{{family:'IBM Plex Mono',size:11}},
        }}
      }}
    }}
  }});

  const groups = Object.keys(data.group_distribution).sort();
  const tVals  = groups.map(g=>data.group_distribution[g].train);
  const vVals  = groups.map(g=>data.group_distribution[g].val);
  const xVals  = groups.map(g=>data.group_distribution[g].test);

  if(barChart) barChart.destroy();
  barChart = new Chart(document.getElementById('bar-chart'),{{
    type:'bar',
    data:{{
      labels:groups.map(g=>g.length>16?g.slice(0,14)+'…':g),
      datasets:[
        {{label:'Train',     data:tVals,backgroundColor:COLORS.trainA,borderColor:COLORS.train,borderWidth:1}},
        {{label:'Validation',data:vVals,backgroundColor:COLORS.valA,  borderColor:COLORS.val,  borderWidth:1}},
        {{label:'Test',      data:xVals,backgroundColor:COLORS.testA, borderColor:COLORS.test, borderWidth:1}},
      ]
    }},
    options:{{
      responsive:true,maintainAspectRatio:false,
      plugins:{{
        legend:{{labels:{{color:'#7a776e',font:{{family:'IBM Plex Mono',size:11}},usePointStyle:true,pointStyleWidth:8}}}},
        tooltip:{{backgroundColor:'#fff',borderColor:'#dddbd5',borderWidth:1,titleColor:'#1a1916',bodyColor:'#7a776e',padding:10,titleFont:{{family:'IBM Plex Mono',size:11}},bodyFont:{{family:'IBM Plex Mono',size:11}}}}
      }},
      scales:{{
        x:{{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:9}},maxRotation:45}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}}}},
        y:{{ticks:{{color:'#9e9b93',font:{{family:'IBM Plex Mono',size:10}}}},grid:{{color:'#f2f0ec'}},border:{{color:'#dddbd5'}}}}
      }}
    }}
  }});

  const maxTotal = Math.max(...groups.map(g=>
    data.group_distribution[g].train+data.group_distribution[g].val+data.group_distribution[g].test
  ));

  let t=`<table class="group-table"><thead><tr>
    <th>tic group</th>
    <th class="train">train</th>
    <th class="val">validation</th>
    <th class="test">test</th>
    <th>total</th>
  </tr></thead><tbody>`;

  groups.forEach(g=>{{
    const d=data.group_distribution[g];
    const tot=d.train+d.val+d.test;
    const pct=v=>maxTotal>0?(v/maxTotal*100).toFixed(0):0;
    t+=`<tr>
      <td><span class="group-name">${{g}}</span></td>
      <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill train" style="width:${{pct(d.train)}}%"></div></div><span class="bar-num">${{fmt(d.train)}}</span></div></td>
      <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill val"   style="width:${{pct(d.val)}}%"></div></div><span class="bar-num">${{fmt(d.val)}}</span></div></td>
      <td><div class="bar-cell"><div class="bar-bg"><div class="bar-fill test"  style="width:${{pct(d.test)}}%"></div></div><span class="bar-num">${{fmt(d.test)}}</span></div></td>
      <td style="color:var(--muted)">${{fmt(tot)}}</td>
    </tr>`;
  }});
  t+='</tbody></table>';
  document.getElementById('group-table-wrap').innerHTML=t;

  allFileRows=[];
  const add=(files,split)=>files.forEach(f=>{{
    const p=f.replace('.wav','').split('_');
    allFileRows.push({{filename:f,split,id:p[0],sess:p[1]?p[1].replace('V',''):'',phase:p[2]||''}});
  }});
  add(data.train_files,'train');
  add(data.val_files,'val');
  add(data.test_files,'test');
  allFileRows.sort((a,b)=>a.filename.localeCompare(b.filename));
  renderFileTable(allFileRows);
}}

function renderFileTable(rows){{
  document.getElementById('file-tbody').innerHTML=rows.map(r=>`
    <tr><td>${{r.filename}}</td><td><span class="badge ${{r.split}}">${{r.split}}</span></td><td>${{r.id}}</td><td>${{r.sess}}</td><td>${{r.phase}}</td></tr>
  `).join('');
}}

function filterFiles(q){{
  q=q.toLowerCase();
  renderFileTable(allFileRows.filter(r=>
    r.filename.toLowerCase().includes(q)||r.split.includes(q)||r.id.toLowerCase().includes(q)
  ));
}}

renderDashboard(SPLITS['file']);
</script>
</body>
</html>"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    print(f"[dashboard] Saved to: {output_path}")


if __name__ == "__main__":
    with open(os.path.join(HOME_DIR, "configs", "paths.yaml"), "r") as f:
        paths_cfg = yaml.safe_load(f)

    output_dir = Path(paths_cfg["output_dir"])
    _generate_dashboard(
        file_split_path    = output_dir / "splits" / "file_split"    / "split_summary.json",
        patient_split_path = output_dir / "splits" / "patient_split" / "split_summary.json",
        output_path        = output_dir / "splits" / "dashboard.html",
    )