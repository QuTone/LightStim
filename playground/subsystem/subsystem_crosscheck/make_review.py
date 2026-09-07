"""Build and execute a lightweight review notebook from already collected assets."""
import datetime
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from nbconvert import HTMLExporter

from common import RESULTS, ROOT, bs_theory


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |',
                     '| '+' | '.join(['---']*len(headers))+' |']+
                    ['| '+' | '.join(map(str,row))+' |' for row in rows])


def estimate(row):
    if not row.errors:
        return f"<{row.ci95_high:.3g}（95% 上限）"
    return f"{row.ler:.3g} [{row.ci95_low:.3g}, {row.ci95_high:.3g}]"


def report(df,paper):
    summary=json.loads((RESULTS/'run_summary.json').read_text())
    exact=json.loads((RESULTS/'bacon_capacity_exact_validation.json').read_text())
    c=df[(df.experiment=='bacon_capacity')&(df.decoder=='bposd')].copy()
    cap=[]
    for p in [.005,.01]:
        for _,r in c[np.isclose(c.p,p)].sort_values('size').iterrows():
            cap.append([int(r['size']),f'{p:g}',f'{int(r.errors)}/{int(r.shots):,}',estimate(r),f'{r.theory_ler:.3g}'])
    circuit=[]
    for decoder,variant in [('bposd','default'),('bposd','serial'),('mwpf','default')]:
        f=df[(df.experiment=='bacon_circuit')&(df.decoder==decoder)&(df.variant==variant)&np.isclose(df.p,.001)]
        for _,r in f.sort_values('size').iterrows():
            circuit.append([int(r['size']),f'{decoder}/{variant}',f'{int(r.errors)}/{int(r.shots):,}',estimate(r),r.statuses])
    comparison=[]
    for r in [3,4]:
        ref=paper[(paper['size']==r)&np.isclose(paper.p,.001)].iloc[0]
        comparison.append([r,'作者公开 CSV','SW BP+LSD',f'{int(ref.errors)}/{int(ref.shots):,}',estimate(ref),'1.00'])
        subset=df[(df['size']==r)&np.isclose(df.p,.001)&df.experiment.isin(['shyps_reference_z','shyps_auto_z','shyps_generic_z'])]
        for _,row in subset.iterrows():
            if row.decoder=='mle-ilp':continue
            comparison.append([r,row.experiment.replace('shyps_',''),f'{row.decoder}/{row.variant}',
                               f'{int(row.errors)}/{int(row.shots):,}',estimate(row),f'{row.ler/ref.ler:.2f}'])
    spaces=[]
    for r in [3,4]:
        meta=json.loads((RESULTS/f'shyps_r{r}_equivalence.json').read_text())
        spaces.append([r,meta['physical_round_pairs'],meta['reference_z_detectors'],meta['z_detector_space']['rank_a'],
                       meta['auto_z_detectors'],meta['z_detector_space']['rank_b'],r*r])
    p4=paper[(paper['size']==4)&np.isclose(paper.p,.001)].iloc[0]
    low=c[np.isclose(c.p,.005)].sort_values('size')
    suppression=low.iloc[0].ler/low.iloc[-1].ler
    n_nominal=int(c.theory_in_ci95.astype(bool).sum())
    n_family=int(c.theory_in_family95.astype(bool).sum())
    text=f"""# Bacon–Shor / SHYPS cross-check：实验记录

生成时间：{datetime.datetime.now(datetime.timezone.utc).isoformat()}。
本轮保存了 **{summary['aggregate_points']} 个聚合实验点**；总 decoding 工作量
**{summary['total_decoded_shots']:,} shots**。不同 decoder 会复用样本，这个总数不是独立物理样本数。
当前未结束的聚合组：**{summary['unfinished_groups']}**。

## 结论

**Bacon–Shor 的文献噪声模型交叉检查通过。SHYPS 的 code/gauge 图和物理电路核对通过，
但 detector 集不完全相同，而且本轮没有复现论文的 decoder 和完整性能曲线。**
不能从 signed flow、理想采样或 DEM 提取成功，直接推出 noisy logical error rate 已经正确。
本轮确实发现了 BPOSD 参数与自动 detector 表达之间的性能问题。

已有实现仍在此 worktree 的 `lightstim/qec_code/bacon_shor/`、
`lightstim/qec_code/shyps/`，共同 SE block 在
`lightstim/qec_code/generic_css/gauge_SE_block.py`。
这一轮没有修改生产库代码，也没有新建或推进 PR。
新实验由原生 `MemoryExperiment` / `CircuitBuilder` / `SyndromeTracker` 构建，
使用 LightStim decoder registry 解码；只有作者基准直接读取其公开 Stim 文件。

## 1. Bacon–Shor：严格匹配 publication 的比较

基准是 Napp–Preskill，*Optimal Bacon-Shor codes*，QIC 13, 490–510 (2013)。
采用其 perfect-syndrome 模型：每个 data qubit **一次独立 X(p) 和一次独立 Z(p)**，
SE、初始化和末端读出完全理想。这里比较的是 Z-memory 的一个 logical observable，
即 X logical error sector。它既不是总概率 p 的 depolarizing channel，也不是 X/Z 两类
logical failure 的合并概率。[论文](https://preskill.caltech.edu/pubs/preskill-2013-optimal-codes.pdf)

对奇数正方形 distance d，论文的多数判决结果为

\\[
q=\\frac{{1-(1-2p)^d}}{{2}},\\qquad
P_L=\\sum_{{j=(d+1)/2}}^d {{d\\choose j}} q^j(1-q)^{{d-j}}.
\\]

这里并非只拿模拟曲线“看起来接近”作判断：对 d=3,5,7,9、p=0.005,0.01,0.02,0.04，
枚举实际 LightStim 电路 DEM 的全部 fault patterns，把同 syndrome 的两个 logical class
概率分别求和，再算真正 degenerate-ML 的错误概率。**16/16 个点与上式在数值精度内一致**。
这验证了电路、detectors 和 observable 对应的统计模型。BPOSD 的 Monte Carlo 独立结果如下；
完整 16 点在 [results.csv](results.csv)。当前 {n_nominal}/16 点的理论值落在逐点 95% 区间，
{n_family}/16 落在对这 16 点作 Bonferroni 调整后的 family 95% 区间。

{table(['d','p','errors / shots','BPOSD LER [95% CI]','论文精确公式'],cap)}

论文还给出 p=0.001、d=173 的数值 2.638×10⁻²⁸；重算为
**{exact['published_anchor']['recomputed_ler']:.10g}**。
这是解析数值核对，**不是** d=173 的 Monte Carlo 结果，也没有构建那个尺寸的 LightStim 电路。

![Bacon–Shor capacity](figures/bacon_capacity.png)

在 p=0.005、d=3→9 的已测范围，BPOSD 错误率下降约 **{suppression:.1f} 倍**，
呈现有限尺寸范围内接近指数的抑制；随着 p 降低，错误率也降低。
但固定 p 下 Bacon–Shor 有最佳尺寸：例如 p=0.04 时 d 继续增大反而更差。
图中 d>9 的线是文献精确公式，不是额外模拟点；不能把局部指数趋势解释为无限尺寸的 threshold。

## 2. Bacon–Shor：原生 SE 电路与 decoder 对照

这里使用原生 patch、默认 X→Z gauges、d 个 XZ pairs。每次 reset 和 ancilla measurement
以 p 翻转，CNOT 后 DEPOLARIZE2(p)，末端 data readout 理想，无 idle noise。
这是单独的 circuit-noise 实验，不拿上一节的 capacity 公式来比较。
下表固定 p=0.001，计数仍是整个 memory shot 的错误率，没有除以 d。

{table(['d','decoder / variant','errors / shots','LER [95% CI]','停止原因'],circuit)}

![Circuit noise](figures/bacon_circuit_decoders.png)

circuit-noise 的部分 d=7 点因时间上限而计数较少；serial BP 在低 p 下有尺寸抑制趋势，
但这些点不足以单独支持精确的指数拟合或 threshold 声明。capacity 的高统计量结论与此分开。

发现了可重复的 decoder 问题：d=3、p=0.0005，BPOSD parallel/dynamic scaling 在 107 个
DEM 单机制症状中解错两个，其 priors 之和约 2.667×10⁻⁴。增加 BP iterations 从 100 到
1000 没有消除；serial BP 或 MWPF 能解对所有这些单机制症状。
所以默认配置下低 p 的异常 scaling 有明确的 decoder 原因，不能归咎于 gauge tracking。
所有原始差结果均保留在表和图中，见 `bacon_circuit_single_fault_decoder_audit.json`。
两个机制都能由 data qubit 4 上单次 Y error 触发，分别位于第 31 和 68 条 noise instruction；
完整 CNOT / target 定位在 `bacon_bposd_single_fault_locations.txt`。

## 3. SHYPS：code、物理门和 detectors 分开核对

作者仓库固定在 `df815862d22dd02d3c3a7f7266639ac5b7840a91`。
r=3、r=4 的经典 gauge 图均与现有 `SHYPSCode` 同构；保存了每个 qubit 的置换表。
将作者的 reset/CNOT/measurement blocks 重编号后交给 LightStim，
**没有把作者的 DETECTOR 或 OBSERVABLE_INCLUDE 交给 Tracker**。
LightStim 重新产生 annotations；忽略坐标和 TICK 后，所有物理门、噪声位置、概率、
measurement 顺序均逐项相同。每条生成的 detector 和 observable 都通过 exact signed flow。
这个结论只对参考 schedule 适配实验成立；原生 generic XZ 的顺序/edge coloring 并未声称与作者一致。

{table(['r','实际 SE pairs','作者 Z detector 数','作者 rank','自动 Z detector 数','自动 rank','logical 数'],spaces)}

作者 Z detector span 是自动结果的**严格子空间**；modulo 自动 detectors 后，
两套 logical observable span 完全相同。比较的是 affine measurement-record GF(2) 空间，
不仅仅数 detector 行数。各 logical 的逐个编号/基可以不同，但 any-logical block failure 的定义不变。

![Detector spaces](figures/shyps_detector_spaces.png)

额外约束有明确来源。r=3：第一次 Z round 多 28 个独立约束，以后 4 轮每轮多 12 个，
总共 76。r=4：初始多 165 个，以后 8 轮每轮多 44 个，总共 517。
它们包括初始 gauge fixing 可知的信息以及同一轮冗余 gauge measurements 的乘积关系。
例如 r=3 的第二个 Z round 中，absolute record indices 98、99、101、102 的 gauge
Pauli 乘积为 identity；其 parity 是有效的 measurement-consistency detector。
具体 supports、门数和置换见 `shyps_r*_physical_audit.json`、`shyps_r*_qubit_mapping.csv`。

Display layout 是代数连接图的摆放，尚未加入 nearest-neighbor、routing 或额外 idle 时间成本。
这轮验证的是 memory circuit；没有复现论文的 Clifford compilation 或全部 logical-gate 实验。

## 4. SHYPS：与公开错误率数据相差多少

论文 v3 使用 **proprietary sliding-window BP+LSD**，不是本轮全图 BPOSD。
内层参数：r=3 是 BP=100、scaling=0.1、LSD order=1；r=4 是 BP=2000、scaling=0.85、
LSD order=4；window/commit 均为 (2,1)。仓库没有公开完整 sliding-window 实现。
[论文及 Table VI](https://arxiv.org/pdf/2502.07150v3)、
[公开数据仓库](https://github.com/PhotonicInc/ComputingEfficientlyInQLDPCCodes)。

下表固定 p=0.001。`reference_z` 是作者电路和作者 detectors；`auto_z` 是同一物理电路，
detectors 来自 LightStim；`generic_z` 是现有原生 generic XZ schedule，取自动 Z detectors。
每一项都以 9 或 16 个 logical observables 中任一个错误作为 block failure。
比值是相对作者公开 CSV 的 **shot error rate**，而非论文图的 per-round rate。

{table(['r','电路 / detector 来源','decoder / variant','errors / shots','LER [95% CI]','对作者比值'],comparison)}

![Reference decoding](figures/shyps_reference_decoders.png)

![Native integration](figures/shyps_integration_comparison.png)

![r=4 native integration](figures/shyps_r4_integration.png)

更多有效 detectors 并不保证近似 decoder 表现更好。r=3、p=0.0005，自动 Z detector 集
配 BPOSD 默认 scaling 时，在完整单机制审计中有 6 个解错；使用 scaling=0.1 后为 0。
这并不证明 0.1 是最优参数，也不证明 decoder 在多故障时最优。
r=4 的大 detector 集在部分测试配置下也出现了明显性能下降和计算开销。
另存了 r=4 的 scaling=0.1、BP=1/100 iterations 的小样本诊断结果；它们也保留在图表中，
不是经过独立验证的最优参数。128 个抽样单机制症状通过不能代替全部 fault-pattern 或 LER 检查。
scaling=0.1、BP=100 时，p=0.001 的诊断得到 1/500，统计区间很宽；p=0.002 为 52/480，
仍明显高于作者公开的 500/21240。降低 scaling 能改善某些点，但没有得到全曲线的性能匹配。
应把 **detector 表达/选择与 decoder 的兼容性** 列为 integration 尚未解决的一项，
不能只看 CI、signed flows 和 DEM 成功就宣布论文性能已复现。

MLE-ILP 额外跑了小样本：它求最可能的 fault pattern，不是按 logical class 求和的
degenerate ML。SHYPS r=3 的 0.2 s/shot 预算产生了大量超时；这些点计入 operational
failure 并单独记录 `decode_failures`，不能当作其不限时 logical error rate。
MWPF 使用 cluster-node limit=50，也不作最优 ML 声明。

## 5. 公开材料中需要保留的一个不一致

论文 Table II 的 r=4、p=0.001 列写的是 **7×10⁻⁴ per round**。
公开 figure-2 CSV 是 **97 / 174245 = {p4.ler:.8g} per shot**；按论文 Eq. (52)
及 s=8 换算为 **{p4.eq52_s_d:.8g} per round**，s=9 时为
**{p4.eq52_s_d_plus_1:.8g}**。这些数与 Table II 不一致。
表格原页已保存为 `assets/shyps_paper_page7.png`。
本轮统一以公开 CSV 的原始 errors/shots 为主基准，没有擅自修改论文数值。

参考文件实际含 1 个 preparation pair 加 d 个重复 pairs；论文主文称 d rounds。
因此 `paper_reference_rates.csv` 同时给出 s=d、s=d+1 的 Eq. (52) 换算，
主要比较直接使用 per-shot 计数，避免把 round convention 混进 decoder 差异。

## 6. 资产、复现与下一步

- [所有聚合计数和区间](results.csv)、[与作者逐点对比](comparison_to_paper.csv)、
  [Bacon–Shor scaling](bacon_scaling.csv)。
- `jobs/`：每个 batch 的 seed、errors、shots、decoder failures 和耗时；
  `configs/`：实际参数；零错误点有 95% 上限，没有画成 LER=0。
- `assets/circuits/`：生成和参考 Stim circuits、DEMs；`assets/samples/`：每个任务的
  第一批 packed detector/observable samples 和 predictions；其余 batches 可按记录重采样。
- `reference_snapshot/`：作者数据、r=3/4 memory circuits、README、MIT license；
  `papers/`：论文 PDF 和提取文本；`manifest.json`：revision、依赖版本和来源 hashes。
  `implementation_at_dc1fec1.patch` 保存本轮所用生产实现相对其 base revision 的变更，便于重建环境。
- `figures/`：PNG / SVG / PDF；本目录的 `review.html` 是自包含的图表浏览页。
- 本轮重新运行 Bacon–Shor / SHYPS 的 45 个现有 tests，全部通过。
  严格 MWPF replay 的状态在 `mwpf_audits/`；命令和完整方法见上级 [README](../../README.md)。
  每个已完成的 replay 都重新生成全部 batches、要求 native panic 抛出异常，并核对逐批计数。

建议明天先审查 native patch 的 supports / logical basis、generic SE 的 CNOT ordering，
再决定给 BPOSD 使用 centre-based detector 表达，还是保留全部约束并配合适当的 decoder。
任何 detector 投影或基变换都应保留自动生成的 provenance，并明确它是否丢弃了有效信息。
下一阶段若要声称 paper reproduction，还需要复现其 sliding-window BP+LSD 或拿到作者实现，
再做相同噪声、相同 round/observable 定义下的对照。
"""
    (RESULTS/'REPORT.md').write_text(text)
    return text


def notebook():
    intro='''# Subsystem integration cross-check

本 notebook 展示已保存的实验资产。报告位于 `playground/subsystem/subsystem_crosscheck/results/2026-09-06/REPORT.md`。

Bacon–Shor：匹配 publication 的 capacity 检查 + native SE circuit noise。
SHYPS：reference / regenerated / generic 分开比较。更多 detectors 不保证 BPOSD 更好。
全部纵轴默认是 **logical block failure per shot**，不自动除以 rounds 或 logical 数。

正式 Bacon–Shor demo 见 `notebooks/Memory/memory_bacon_shor.ipynb`；早期 SHYPS visualization 本地存档于 `archive/memory_subsystem.ipynb`。
'''
    setup='''from pathlib import Path
import pandas as pd
from IPython.display import display, Image, Markdown

root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "lightstim").is_dir())
assets = root / "playground/subsystem/subsystem_crosscheck/results/2026-09-06"
rates = pd.read_csv(assets / "results.csv")
paper_comparison = pd.read_csv(assets / "comparison_to_paper.csv")
display(Markdown(f"已保存 {len(rates)} 个聚合点。完整方法、失败案例和限制见 REPORT.md。"))
'''
    cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_code_cell(setup)]
    for title,name in [('Bacon–Shor: exact noise-model comparison','bacon_capacity'),
                       ('Bacon–Shor: native circuit and decoders','bacon_circuit_decoders'),
                       ('SHYPS: detector subspaces','shyps_detector_spaces'),
                       ('SHYPS: authors’ exact circuit / different decoders','shyps_reference_decoders'),
                       ('SHYPS: LightStim integration / decoder sensitivity','shyps_integration_comparison'),
                       ('SHYPS r=4: unresolved detector / decoder performance','shyps_r4_integration')]:
        cells.extend([nbformat.v4.new_markdown_cell('## '+title),
                      nbformat.v4.new_code_cell(f'display(Image(filename=str(assets / "figures/{name}.png")))')])
    cells.extend([nbformat.v4.new_markdown_cell('## 原始计数、置信区间与失败状态'),
                  nbformat.v4.new_code_cell('''columns = ["experiment", "size", "p", "decoder", "variant", "errors", "shots",
           "ler", "ci95_low", "ci95_high", "decode_failures", "statuses"]
display(rates[columns])'''),
                  nbformat.v4.new_markdown_cell('## 与公开 CSV 的对照（p=0.001）'),
                  nbformat.v4.new_code_cell('''display(paper_comparison.loc[paper_comparison.p == 0.001,
    ["experiment", "size", "decoder", "variant", "errors", "shots", "ler",
     "paper_errors", "paper_shots", "paper_ler", "ratio_to_paper", "decode_failures"]])''')])
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
    shell=InteractiveShell.instance();count=0
    for cell in nb.cells:
        if cell.cell_type!='code':continue
        count+=1
        with capture_output() as captured:
            result=shell.run_cell(cell.source)
        if result.error_before_exec or result.error_in_exec:
            raise RuntimeError(result.error_before_exec or result.error_in_exec)
        cell.execution_count=count;cell.outputs=[]
        if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
        if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
        for out in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data',data=out.data,metadata=out.metadata))
    path=ROOT/'playground/subsystem/subsystem_crosscheck.ipynb'
    nbformat.write(nb,path)
    html,_=HTMLExporter().from_notebook_node(nb)
    (RESULTS/'review.html').write_text(html)
    print('Executed review notebook:',path,flush=True)


if __name__=='__main__':
    df=pd.read_csv(RESULTS/'results.csv');paper=pd.read_csv(RESULTS/'paper_reference_rates.csv')
    report(df,paper);notebook()
