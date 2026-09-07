"""Render the completed decoder experiment into CSV, figures and a notebook."""
import csv
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lightstim-decoder-review')
os.environ.setdefault('IPYTHONDIR', '/tmp/lightstim-decoder-review-ipython')
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from scipy.stats import binomtest
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from nbconvert import HTMLExporter
from playground.subsystem.bacon_shor_schedule.decoder_review import OUT
from playground.subsystem.subsystem_crosscheck.common import save_json

jobs = [json.loads(p.read_text()) for p in sorted(OUT.glob('d*.json')) if not p.name.endswith('_config.json')]
assert len(jobs) == 18 and all(j['status'] in {'error_target', 'shot_cap', 'time_cap'} for j in jobs)
rows, pairs = [], []
for j in jobs:
    c = j['config']
    for decoder, result in j['results'].items():
        seconds = sum(b['decode_seconds'][decoder] for b in j['batches'])
        rows.append(dict(d=c['d'], p=c['p'], decoder=decoder, shots=j['shots'],
                         errors=result['errors'], ler=result['ler'], ci_low=result['ci95'][0], ci_high=result['ci95'][1],
                         decode_seconds=seconds, decode_shots_per_second=j['shots']/seconds,
                         status=j['status'], detectors=j['num_detectors'], mechanisms=j['num_errors']))
    if c['profile'] == 'z_pair':
        totals = {k: sum(b['paired'][k] for b in j['batches']) for k in ['both', 'mwpm_only', 'bposd_only']}
        discordant = totals['mwpm_only'] + totals['bposd_only']
        pairs.append(dict(d=c['d'], p=c['p'], shots=j['shots'], **totals,
                          nominal_pvalue=binomtest(totals['mwpm_only'], discordant).pvalue if discordant else 1.))
with (OUT / 'decoders.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
unique = {}
for j in jobs:
    key = (j['config']['d'], j['config']['p'])
    unique[key] = max(unique.get(key, 0), j['shots'])
save_json(OUT / 'summary.json', dict(results=rows, paired=pairs, unique_seeded_shots=sum(unique.values()),
                                    decoder_shots=sum(r['shots'] for r in rows)))

colors = {'mwpm_z': '#247aa8', 'bposd_z_serial': '#d26c24',
          'bposd_full_serial': '#268354', 'bposd_full_parallel': '#9365b8'}
labels = {'mwpm_z': 'MWPM / Z detectors', 'bposd_z_serial': 'BPOSD serial / Z detectors',
          'bposd_full_serial': 'BPOSD serial / full XZ', 'bposd_full_parallel': 'BPOSD parallel / full XZ'}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout='constrained')
for ax, p in zip(axes, [.0005, .001, .002]):
    for index, decoder in enumerate(['mwpm_z', 'bposd_z_serial']):
        selected = sorted([r for r in rows if r['p'] == p and r['decoder'] == decoder], key=lambda r:r['d'])
        x = np.array([r['d'] for r in selected]) + (index-.5)*.1
        y = np.array([r['ler'] for r in selected])
        lo, hi = np.array([[r['ci_low'],r['ci_high']] for r in selected]).T
        ax.errorbar(x,y,yerr=[y-lo,hi-y],capsize=3,marker='o',color=colors[decoder],label=labels[decoder])
        for xx, r in zip(x, selected):
            if r['errors'] < 100:
                ax.scatter(xx, r['ler'], s=80, facecolors='none', edgecolors='black', zorder=4)
    ax.set(title=f'p = {p:g}', xlabel='d (also number of XZ pairs)', ylabel='Z-memory failure probability per shot',
           xticks=[3,5,7,9], yscale='log')
    ax.grid(which='both', alpha=.2)
axes[0].legend(fontsize=8)
fig.suptitle('Dedicated Bacon-Shor SE: CPU decoders on identical samples\n95% binomial intervals; black rings mark fewer than 100 failures')
for ext in ['png', 'svg']:
    fig.savefig(OUT / f'decoder_scaling.{ext}', dpi=160, bbox_inches='tight')
plt.close(fig)

def table(selected):
    lines = ['| d | p | Decoder / detectors | Errors / shots | LER [95% CI] | Status |',
             '|---|---|---|---:|---|---|']
    for r in selected:
        lines.append(f"| {r['d']} | {r['p']:g} | {r['decoder']} | {r['errors']}/{r['shots']} | {r['ler']:.3g} [{r['ci_low']:.3g}, {r['ci_high']:.3g}] | {r['status']} |")
    return '\n'.join(lines)

report = '''# Dedicated Bacon–Shor decoder review — 2026-09-06

本次全部使用 CPU：LightStim registry 的 PyMatching 与 stimbposd/ldpc BP+OSD。
默认 memory baseline 为 MWPM + 同 basis detectors；BPOSD 是明确配置的对照。
所有 circuit、detector、observable 来自 LightStim 的 BaconShorCode / dedicated SE / MemoryExperiment / tracker。

## 实验口径

Z memory，d=3,5,7,9，各执行 d 个 XZ pairs。Reset 和 ancilla measurement flips p，CX 后 DEPOLARIZE2(p)，无 idle noise，理想 final data readout。
LER 是完整 memory shot 失败概率，不是 per-round LER，也不是 code-capacity LER。
BPOSD：max BP iterations 100，min-sum，dynamic scaling (0)，OSD-CS order 10。分别记录 serial/parallel BP update schedule；两者都是 CPU。
Z-only projection 保留原自动 detector 行，完整 projection DEM 无 hyperedges，不拆分、不忽略 errors。Full-XZ BPOSD 使用全部原始 detectors。

每个配置目标至少 100 errors，最多 2M shots 或 180 秒。表中的置信区间为 nominal pointwise Clopper–Pearson intervals；error-target stopping、多个点的比较不提供 simultaneous coverage。少量 failure 的点仅是初步估计。
样本从完整 noisy circuit 生成；同一 d,p 的 seed/batch size 不包含 decoder，因此相同 batch 在各 profile 是相同输入。Z-only 两 decoder 的停止点完全一致，可直接配对。Full-XZ profiles 的停止点不同，不应称为全程相同 shot count 的比较。

## MWPM 与 BPOSD 的配对实验

![CPU decoder scaling](decoder_scaling.png)

'''
report += table([r for r in rows if r['decoder'] in ['mwpm_z','bposd_z_serial']])
report += '\n\n## Full-XZ BPOSD 对照\n\n' + table([r for r in rows if 'full' in r['decoder']])
report += '''

## 解读

在同 Z-detector 模型下，两 decoder 接近；多个点逐 shot 相同。较大 d 的 BP+OSD 出现少量额外失败，不能凭少数差异宣称普遍最优性。`summary.json` 保存 matched discordant counts 及未作多重比较校正的 exact binomial p-values。
固定 d 降低 p 可明显降低 LER。p=0.001 的 d=3→9 存在 suppression；p=0.002 的曲线则开始变平，不能宣称这整个 p 区间都随 d 单调改善或存在无限 distance 的指数 suppression。
常规全 X / 全 Z Bacon–Shor 无 asymptotic threshold；有限窗口下降不构成反例。完整 XZ 信息也不保证启发式 BPOSD 更好，BP schedule 和 factor graph 会影响结果。

另有前一轮 dedicated MWPM 每个 d 一百万独立 shots 的高统计 baseline，保存在 `../2026-09-06/mwpm.csv`，p=0.001 的 counts 分别为 532、198、99、53。

## 验证和资产

`single_fault_audit.json`：对 d=3,5,7,9 的每个完整 DEM error mechanism 单独构造 syndrome/observable，并测试四个配置。当前 dedicated circuit 全部通过。这只证明这些单机制被正确解码，不替代对所有多 fault 的最优解码证明。
此前 generic circuit 上的 full-DEM parallel-BPOSD single-fault failure 是另一 circuit/DEM 的结果，不应归到当前 dedicated circuit。
每个 job 保存 `.stim`、实际 decoder 的 `.dem`、配置、逐 batch seed/count/decode time，以及 first-batch packed raw data 和 predictions。
`manifest.json` 保存 source hashes 和版本；`validation.json` 保存输出复核。`literature/` 保存调研原文快照和来源 hash。
CPU decode throughput 在 CSV 中逐配置列出，包含包装与 batch 开销；并发负载随时间变化，不是严格独立的硬件性能 benchmark。
'''
report += f"\n18 jobs 完成；按 d,p 与 seed 去重后的物理 shots：{sum(unique.values()):,}；各 decoder workload shots 总和：{sum(r['shots'] for r in rows):,}。\n"
report += '\n正式 memory demo：`notebooks/Memory/memory_bacon_shor.ipynb`。\n'
(OUT / 'REPORT.md').write_text(report)

nb = nbformat.v4.new_notebook()
md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
nb.cells = [
    md('''# Bacon–Shor CPU decoder review

**[DEMO]** dedicated SE，LightStim 自动 detectors。CPU BPOSD 与 MWPM 的同样本比较。
Z memory，d 个 XZ pairs，reset/ancilla readout flips p，CX depolarization p，无 idle noise，final data readout 理想。
默认 baseline：MWPM + Z-record detectors；完整 XZ DEM 含 hyperedges。
'''),
    code('''from pathlib import Path
import sys, json
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lightstim").is_dir())
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from playground.subsystem.bacon_shor_schedule.decoder_review import build, detector_rows, OUT, PARAMS
from playground.subsystem.bacon_shor_schedule.run import select_basis_detectors
from lightstim.simulation.decoder_backend import get_decoder
from IPython.display import Image, display
import numpy as np
import pandas as pd
full = build(3, .001)
selected = select_basis_detectors(full, "Z")
print("full / selected detectors:", full.num_detectors, selected.num_detectors)
print("BPOSD CPU parameters:", dict(PARAMS, schedule="serial"))
print("Selected detector indices:", detector_rows(full))'''),
    md('## 已完成的 scaling\n\n黑圈表示不足 100 个 errors；低 p、大 d 的误差条较宽。LER 为完整 shot。'),
    code('''data = pd.read_csv(OUT / "decoders.csv")
display(data[["d","p","decoder","errors","shots","ler","ci_low","ci_high","status"]])
display(Image(filename=str(OUT / "decoder_scaling.png")))'''),
    md('## 验证一次真实的 decoder 调用\n\n复用保存的 first batch，通过 LightStim decoder registry 重算 MWPM 和 BPOSD。'),
    code('''saved = np.load(OUT / "d3_p0.001_z_pair_first_batch.npz")
for name, params in [("pymatching", {}), ("bposd", dict(PARAMS, schedule="serial"))]:
    decoder = get_decoder(name, **params).compile_decoder_for_dem(dem=selected.detector_error_model())
    pred = decoder.decode_shots_bit_packed(bit_packed_detection_event_data=saved["decoder_detectors"])
    key = "mwpm_z" if name == "pymatching" else "bposd_z_serial"
    assert np.array_equal(pred, saved[key])
    print(name, "reproduced saved predictions; failures:", int(((pred ^ saved["observables"]) & 1).any(axis=1).sum()))'''),
    md('## 配对差异与 decoder CPU 用时\n\nBP parallel 指消息更新顺序，不是 GPU。用时是共享机器上的实际 decode time，不能视为独立硬件 benchmark。'),
    code('''summary = json.loads((OUT / "summary.json").read_text())
display(pd.DataFrame(summary["paired"]))
display(data[["d","p","decoder","decode_seconds","decode_shots_per_second"]])
audit = json.loads((OUT / "single_fault_audit.json").read_text())
assert all(v["bad_count"] == 0 for r in audit for v in r["profiles"].values())
print("Single-mechanism audit:", [(r["d"],r["error_mechanisms"]) for r in audit])'''),
    md('''## 继续运行的入口

`playground/subsystem/bacon_shor_schedule/decoder_review.py` 可按 checkpoint 续跑，`--max-seconds` 增加每 job 总时间预算；每个 job 的 max_shots 默认 2M。
配对 benchmark 直接使用 LightStim registry，以保存同样本的多 decoder predictions。常规非配对采样也可将 `selected` 交给 `SimulationPipeline(DecoderConfig("pymatching"), ...)`。
正式 dedicated SE 与原生 noise / MWPM demo 见 [memory_bacon_shor.ipynb](../../notebooks/Memory/memory_bacon_shor.ipynb)。
'''),
]
nb.metadata['kernelspec'] = dict(display_name='Python 3', language='python', name='python3')
shell = InteractiveShell.instance()
os.chdir(ROOT)
count = 0
for cell in nb.cells:
    if cell.cell_type != 'code': continue
    count += 1
    with capture_output() as captured:
        result = shell.run_cell(cell.source)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}')
    outputs = []
    if captured.stdout: outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
    if captured.stderr: outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
    for output in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data',data=output.data,metadata=output.metadata))
    cell.outputs, cell.execution_count = outputs, count
nbformat.write(nb, ROOT / 'playground/subsystem/bacon_shor_decoder_review.ipynb')
html, _ = HTMLExporter().from_notebook_node(nb)
(OUT / 'review.html').write_text(html)
print('Executed notebook cells:', count, 'unique shots:', sum(unique.values()))
