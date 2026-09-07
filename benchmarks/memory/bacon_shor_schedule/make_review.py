"""Build standalone schedule figures and an executed, small review notebook."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lightstim-dedicated-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock

OUT = Path(__file__).resolve().parent / "results/2026-09-06"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"svg.fonttype":"none", "font.family":"DejaVu Sans"})
patch = BaconShorCode(distance=3)
system = QECSystem()
system.add_patch(patch, name="bs")
blocks = {"Dedicated (default)": BaconShorCodeExtractionBlock(system),
          "Generic edge coloring": GenericCSSGaugeExtractionBlock(system)}
colors = {"X":"#c9691b", "Z":"#277bb5"}
fig, axes = plt.subplots(2, 4, figsize=(13, 7.5), layout="constrained")
for row, (label, block) in enumerate(blocks.items()):
    for col, (basis, layer) in enumerate(zip(["X","X","Z","Z"], block.x_layers+block.z_layers)):
        ax = axes[row,col]
        coords = patch.qubit_coords
        for g in patch.gauges:
            for q in g["data_indices"]:
                a = g["syn_idx"]
                ax.plot([coords[a][0],coords[q][0]], [coords[a][1],coords[q][1]], color="#e0e4e8", zorder=0)
        for a,q in layer:
            control,target = (a,q) if basis=="X" else (q,a)
            ax.annotate("", xy=coords[target], xytext=coords[control],
                        arrowprops=dict(arrowstyle="->",color=colors[basis],lw=2.6,shrinkA=10,shrinkB=10))
        for q,xy in coords.items():
            b = "X" if q in patch.syndrome_indices_x else "Z"
            data = q in patch.data_indices
            ax.scatter(*xy,s=260 if data else 210,marker="o" if data else "s",
                       color="#273644" if data else colors[b],zorder=3)
            ax.text(*xy,str(q),ha="center",va="center",color="white",fontsize=9,zorder=4)
        ax.set(xlim=(-.5,4.5),ylim=(4.5,-.5),aspect="equal")
        ax.axis("off")
        ax.set_title(f"{basis}: CNOT layer {col%2+1}",fontsize=11)
        if col==0:
            ax.text(-.13,.5,label,transform=ax.transAxes,rotation=90,ha="center",va="center",weight="bold")
fig.suptitle("Bacon-Shor d=3: same gauges and depth, different Z-layer order\nArrows indicate control to target",fontsize=14)
for ext in ["svg","png"]:
    fig.savefig(OUT/f"schedule_comparison.{ext}",dpi=160,bbox_inches="tight")
plt.close(fig)

summary = json.loads((OUT/"summary.json").read_text())
if summary["mwpm"]:
    fig,ax = plt.subplots(figsize=(7,4.5),layout="constrained")
    for label,color in [("dedicated","#277bb5"),("generic","#c9691b")]:
        rows = [r for r in summary["mwpm"] if r["schedule"]==label]
        ys = np.array([r["ler"] for r in rows])
        ci = np.array([r["ci95"] for r in rows])
        ax.errorbar([r["d"] for r in rows],ys,yerr=[ys-ci[:,0],ci[:,1]-ys],marker="o",capsize=3,label=label,color=color)
    ax.set(yscale="log",xticks=[3,5,7,9],xlabel="d (= number of XZ pairs)",ylabel="Z-memory LER per shot",
           title="MWPM, p=0.001; independent 1M-shot runs")
    ax.grid(alpha=.2,which="both")
    ax.legend()
    for ext in ["svg","png"]:
        fig.savefig(OUT/f"mwpm_comparison.{ext}",dpi=160,bbox_inches="tight")
    plt.close(fig)

for label,cls in [("dedicated",BaconShorCodeExtractionBlock),("generic",GenericCSSGaugeExtractionBlock)]:
    circuit=MemoryExperiment(qec_patch=BaconShorCode(distance=3),extraction_block_class=cls,basis="Z",rounds=2).build()
    circuit.to_file(OUT/f"d3_two_rounds_{label}.stim")
    (OUT/f"d3_two_rounds_{label}_detslice.svg").write_text(str(circuit.diagram("detslice-with-ops-svg")))

nb=nbformat.v4.new_notebook()
md=nbformat.v4.new_markdown_cell
code=nbformat.v4.new_code_cell
nb.cells=[
md("""# Bacon–Shor dedicated SE / generic SE 对照

**[DEMO]** — 使用已封装的 BaconShorCode、两种 extraction blocks 和 MemoryExperiment。

默认版本固定 X 左→右、Z 上→下（row-index 图）。Generic 版本保留为显式选项。
二者都是四层 CNOT，signed measurement instrument 相同，但实际 Z 层顺序不同。
所有 detectors 和 observables 均由 LightStim Builder/Tracker 生成。
"""),
code('''import sys, json
from pathlib import Path
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "lightstim").is_dir())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lightstim.ir.qec_system import QECSystem
from lightstim.protocols.memory import MemoryExperiment
from lightstim.qec_code.bacon_shor import BaconShorCode, BaconShorCodeExtractionBlock
from lightstim.qec_code.generic_css import GenericCSSGaugeExtractionBlock
from IPython.display import Image, SVG, display
import pandas as pd
ASSETS = ROOT / "benchmarks/memory/bacon_shor_schedule/results/2026-09-06"
system = QECSystem()
system.add_patch(BaconShorCode(distance=3), name="bs")
dedicated = BaconShorCodeExtractionBlock(system)
generic = GenericCSSGaugeExtractionBlock(system)
print("CNOT depth:", dedicated.cnot_depth, generic.cnot_depth)
assert all(generic.circuit.has_flow(f, unsigned=False) for f in dedicated.circuit.flow_generators())
assert all(dedicated.circuit.has_flow(f, unsigned=False) for f in generic.circuit.flow_generators())'''),
md("## 四层配对：看实际的全局 qubit indices\n\n上排为新的默认版本，下排为 generic；箭头为 CNOT control → target。"),
code('display(Image(filename=str(ASSETS / "schedule_comparison.png")))'),
md("## 默认 SE 的实际 Stim 指令\n\n不含 noise 和 detectors；每个 basis block 有独立 reset 和 terminal measurement。"),
code('print(dedicated.circuit)'),
md("## 用原生 MemoryExperiment 生成 detectors\n\n完整的两轮 Z-memory，包含初始化和最终 data readout。"),
code('''memory = MemoryExperiment(qec_patch=BaconShorCode(distance=3), basis="Z", rounds=2)
assert memory.block_class is BaconShorCodeExtractionBlock
circuit = memory.build()
assert not circuit.compile_detector_sampler(seed=70).sample(256, append_observables=True).any()
print("qubits / measurements / detectors / observables:", circuit.num_qubits, circuit.num_measurements, circuit.num_detectors, circuit.num_observables)
display(SVG(filename=str(ASSETS / "d3_two_rounds_dedicated_detslice.svg")))'''),
md("""## 已完成的 distance / detector / decoding 核对

d=3,5,7,9；X/Z memories；每次 d 个 XZ pairs。兩种调度的 affine detector spaces 相同，logical observables 模 detectors 相同。
32 个 distance cases（两种调度、两种 memory basis、两种 readout fault models）都有相等的上下界 d。

下表的 MWPM 使用 Z detectors。p=0.001：reset/ancilla readout flips、CX depolarization，无 idle noise，末端 data readout 理想。
每点固定 100 万 shots，计数是整个 memory shot 的 LER。测量 instrument 相同不代表 noisy LER 必须相同。
"""),
code('''results = json.loads((ASSETS / "summary.json").read_text())
assert all(r["lower_bound"] == r["upper_bound"] == r["d"] for r in results["distances"])
display(pd.read_csv(ASSETS / "mwpm.csv"))
display(Image(filename=str(ASSETS / "mwpm_comparison.png")))'''),
md("""## 显式选择 generic

```python
generic_memory = MemoryExperiment(
    qec_patch=BaconShorCode(distance=3),
    extraction_block_class=GenericCSSGaugeExtractionBlock,
    basis="Z", rounds=3,
).build()
```

Source: `lightstim/qec_code/bacon_shor/SE_block.py`。完整实验脚本和原始数据在 `benchmarks/memory/bacon_shor_schedule/`。
"""),
]
nb.metadata["kernelspec"]={"display_name":"Python 3", "language":"python", "name":"python3"}
shell=InteractiveShell.instance()
os.chdir(ROOT)
count=0
for cell in nb.cells:
    if cell.cell_type!="code":continue
    count+=1
    with capture_output() as captured:
        result=shell.run_cell(cell.source)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f"Notebook cell {count} failed: {result.error_before_exec or result.error_in_exec}")
    outputs=[]
    if captured.stdout:outputs.append(nbformat.v4.new_output("stream",name="stdout",text=captured.stdout))
    if captured.stderr:outputs.append(nbformat.v4.new_output("stream",name="stderr",text=captured.stderr))
    for output in captured.outputs:
        outputs.append(nbformat.v4.new_output("display_data",data=output.data,metadata=output.metadata))
    cell.outputs=outputs
    cell.execution_count=count
path=ROOT/"playground/subsystem/bacon_shor_se_review.ipynb"
nbformat.write(nb,path)
html,_=HTMLExporter().from_notebook_node(nb)
(OUT/"review.html").write_text(html)
print("Executed",count,"cells; wrote",path)
