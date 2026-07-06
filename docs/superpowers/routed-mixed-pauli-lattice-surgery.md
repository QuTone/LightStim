# Routed Mixed-Pauli Lattice Surgery 设计记录

这份文档记录这次讨论得到的正确思路，目标是支持 `ZZZX`、`X1Z2`
这类 mixed X/Z logical product measurement。

## 核心规则

每个 data patch 上需要先回答两个问题：

1. 目标想测的是 `X` 还是 `Z`？
2. 真正接入 ancillary path / bus 的那条边，本征 interface basis 是
   `X` 还是 `Z`？

是否需要 logical H 由这两个 basis 的比较决定：

| 接入 ancilla 的 interface basis | 目标 Pauli | 是否需要 logical H |
| --- | --- | --- |
| X | X | 不需要 |
| X | Z | 需要 |
| Z | Z | 不需要 |
| Z | X | 需要 |

所以不能简单地认为“目标里出现 `X` 就先转成 `Z`”。如果某个 patch
本来就是通过 `X` 边接入 ancilla，那么目标也要测 `X` 时就应该保持原
frame，不应该额外做 H。

实现上应当按这个顺序处理：

1. 对每个参与 patch 的选中边，推断或显式传入 native interface Pauli。
2. 将 native interface Pauli 与目标 Pauli product 逐项比较。
3. 只在不匹配的 patch 上做 logical H。
4. 如果测量后还希望回到原 logical frame，就在 readout 前再 H 回来。

## 几何规则

ancilla 区域不是只能是单纯竖直或单纯水平 corridor。参考
Litinski/von Oppen 的 long-range multi-target CNOT 图，更合适的解释是：
这条会拐弯的 bus 本身是一个完整的 long ancillary surface-code patch。

因此 full-width route 的首要几何规则不是“给每个坐标按最近 interface
染成 X/Z 区域”，而是：

1. ancilla bus 具有 code-distance 级别的固定宽度。
2. data / syndrome checkerboard 在整条 bent patch 上连续延伸。
3. 90 度拐角按普通 surface-code patch corner 处理，自然允许 weight-2 /
   weight-3 边界 stabilizer。
4. 相邻 patch-sized route blocks 之间必须包含 seam 行/列，使整条 bus 等价
   于一个标准 unrotated rectangular / bent patch，而不是一串互相重启相位
   的小 patch。

data patch 的 selected interface window 与这个 long ancilla patch 做
merge。merge-check product 可以留下一个 long ancilla 的 logical boundary
factor；这个 factor 由 ancilla 的初始化/测量 frame 给出，而不是逐个读出
ancilla data qubit。

## Stabilizer 模板规则

paper-style long ancilla 的 stabilizer 设计按 unrotated surface-code patch
生成，但 X/Z sheet 交界处必须使用 mixed/domain-wall stabilizer：

- bulk plaquette 仍是标准 weight-4 X/Z checkerboard。
- X/Z route label 改变的 seam/corner 附近使用 mixed stabilizer；不能把
  这些位置强行保持成纯 CSS X/Z check。
- 外边界、内边界、接口窗口边界和拐角处自然出现 weight-2 / weight-3
  stabilizer。
- 拐弯处不重新初始化 checkerboard phase；它只是同一个 patch 的 corner。
- 参与 logical product 的是 data-ancilla 接触窗口上的 merge-check
  syndrome product。ancilla 内部 stabilizer 负责保持 bus 是一个 code patch，
  不应被解释成独立的 data readout。

如果确实需要在同一个局部位置把 X 边界和 Z 边界通过 domain wall/twist
相连，才需要 mixed X/Z stabilizer 模板。当前代码仍保留
`mixed_stabilizers=True` 作为实验性 scaffold，但参考 Figure 10 的 bent
long ancilla 首选普通 CSS patch stabilizer。

## 例子

对于 `ZZZX`：

- 如果四个接入 interface 被解释为 `Z, Z, Z, Z`，那么只需要在最后一
  个目标 `X` patch 上做 logical H，执行 routed `ZZZZ` 测量，之后根据
  需要再 H 回原 frame。
- 如果四个接入 interface 本来就是 `Z, Z, Z, X`，那么不需要任何 H。
  此时真正需要的是 route 本身支持 mixed X/Z local stabilizer，在 `X`
  区域和 `Z` 区域交界处生成对应的 mixed check。

对于 `X1Z2`：

- 如果 patch 1 通过 `X` interface 接入，patch 2 通过 `Z` interface
  接入，那么这就是 native mixed X/Z measurement。
- 这时默认把 `X1` 转成 `Z1` 反而是不正确的。正确方向是 mixed-boundary
  stabilizer generation，再配套实现对应 logical product 的 outcome /
  tracker 支持。

## 当前实现状态

已经实现：

- `UnrotatedRoutedMultiPatchCoupler` 支持显式选择每个 patch 的接入边，
  并用 Manhattan routing 连接任意布局下的这些边。
- direct routed coupler 默认使用
  `route_width = 2 * code_distance - 1` 的 full ancillary-patch route。
  对 d=3 的 unrotated patch，这个跨度是 5 个整数 lattice 坐标，而不是
  逻辑距离本身的 3。
- full ancillary-patch route 不再允许 data patch 放在任意物理坐标上。
  所有参与 routing 的 data/obstacle patch 必须落在同一个 coarse grid 上：
  每个 patch 本身必须刚好占据一个 `route_width × route_width` 坐标块，
  但相邻 coarse cell 的 origin pitch 是 `route_pitch = route_width + 1 = 2d`，
  不是 `route_width`。多出来的 1 个整数坐标就是两个标准 patch block
  拼接时共享的 seam 行/列；没有这条 seam，边界 weight-3/weight-4
  stabilizer 就会错位。对 d=3 来说，`route_width=5`、`route_pitch=6`，
  所以同方向 patch 的常用安全间距是 12、24、36、... 个整数坐标。
- routed coupler 会先在这个 coarse grid 上做 Manhattan BFS，避开被 data
  patch 占据的 coarse cells；然后把每个 route cell 展开成完整
  `route_width × route_width` ancillary block，并在相邻 route cells
  之间显式加入 seam 行/列。因此 selected interface 的 terminal region
  和中间拐弯/走廊都是由标准 patch blocks 拼成的，不再是 thin skeleton
  膨胀出来的非规整形状。两个水平相邻的 d=3 blocks 在偶数 data 行上会连成
  6 个 data qubit，对应一个标准 `distance_z=6` 的 unrotated rectangular patch。
- route cell 是几何 coarse block，不是独立重新初始化 checkerboard phase
  的小 patch。full ancillary region 的 data/X-syndrome/Z-syndrome parity
  必须在整个 merged lattice 上连续；如果每个 5×5 cell 内重新起相位，
  图形上仍是方块，但边界 merge 的 Pauli product algebra 会断掉。
- interface basis 可以由选中边自动推断，也可以显式传入。
- routed Pauli-product helper 会比较 target Pauli 和 native interface
  Pauli，只在不匹配的位置插入 logical H。
- `mixed_stabilizers=True` 的 full-width routed coupler 是 notebook 当前的
  paper-style long ancillary patch 路径：它在 bulk 中生成 X/Z checks，
  在 X/Z sheet seam 上生成 mixed checks，拐角自然出现 low-weight boundary
  stabilizer。
- `solve_routed_pauli_product_long_ancilla` 会把 syndrome product 里剩下的
  ancillary boundary support 压缩成一个 `AncillaLogicalTerm`。这表示
  long ancilla patch 的已知 logical boundary factor，而不是独立的
  ancillary data readout terms。
- long-ancilla helper 现在使用 paper-style 几何 product：它选中整条
  connected ancillary bus 上属于目标 product sheet 的 stabilizer，而不是
  用最小权重线性代数解只在 data-patch 接口附近挑少数 stabilizer。notebook
  里高亮的 X/Z plaquette 就是实际进入 syndrome product 的 stabilizer。
- `mixed_stabilizers=False` 只适合纯 CSS / 同 basis 的 routed check；对
  `ZZZX` 这种 mixed-interface bus 会在 seam/corner 处产生错误的纯 X/Z
  stabilizer。
- mixed-template 路径会用代数方式暂停与 routed check 反对易的原 patch
  stabilizer，使测试过的 active stabilizer set 保持对易。
- mixed-template 路径还会在 coupler 内部做本地 pruning：X/Z seam 上与 mixed
  check 反对易的纯 coupler check 会被 mixed/twist template 替换；拐弯处若
  naive CSS checkerboard 生成一对反对易的 pure X/Z corner checks，也会按
  固定局部规则删去其中一个。这个 pruning 不引入高权重 closure。
- `solve_routed_pauli_product_syndromes` 可以自动求解并验证目标 logical
  product 的 outcome 分解。正常的 routed/mixed lattice surgery 应该在
  `include_ancilla_readout_terms=False` 时成功；如果失败，则说明当前局部
  stabilizer template 没有让 ancillary Pauli 在 product 中全部抵消。
- 之前尝试过的 high-weight closure syndrome 已经移除。它只能把 residual
  代数上补掉，不是合理的局部 lattice-surgery stabilizer 设计。
- mixed-interface boundary 现在不会再把 data patch 自己的 boundary syndrome
  任意重新涂色。只有当原生 boundary syndrome 类型是该 logical interface
  的互补 stabilizer 类型时，才允许复用这个边界 syndrome；例如 Z interface
  上复用 X boundary checks，X interface 上复用 Z boundary checks。对 d=3
  的一条 selected edge，这样的 boundary checks 数量是 `d-1=2`，不是把整条
  几何边界上所有 syndrome 都拿进来。
- `routed_coupler_data_basis` 会根据 routed ancillary region 的局部
  `route_coord_basis` 生成物理 basis map：
  - `mode="opposite"` 用于初始化：`Z` route 区域准备在 `X`，`X` route
    区域准备在 `Z`。
  - `mode="same"` 用于读出：`Z` route 区域读 `Z`，`X` route 区域读 `X`。
- `multi_patch_LS_straight_unrotated.ipynb`（原 `multi_patch_LS.ipynb`）的 `build_zz_circuit` 默认把非 coupler/data patch
  qubit 初始化在 `Z` basis；只有 coupler ancillary data 会为 ZZ surgery
  单独初始化在 `X` basis。full ancillary patch 可视化（原 `routed_ZZZX_LS.ipynb`
  的角色，该 notebook 已随分支清理移除）也应沿用这个约定，而不是把普通 data patch
  默认初始化成 `X`。
- `solve_routed_pauli_product_syndromes` 仍把两类 ancillary diagnostic 项分开：
  - `selected_ancilla_known_terms` 是由 chosen ancillary initialization
    basis 提供的确定性 +1 本征值，不是 readout。
  - `selected_ancilla_terms` 才是 residual ancillary data readout terms；
    对目标 native full-width mixed measurement，应当为 0。
- 真正对应“绿色点 stabilizer 乘积”的接口是
  `solve_routed_pauli_product_merge_checks`。它现在使用 basis-aware no-trim
  diagnostic：按几何选择完整 coupler stabilizer，但还要按 local route label
  选择 product sheet。在 `Z`-labeled 区域只乘 `Z` checks，在 `X`-labeled
  区域只乘 `X` checks，`MIXED` seam checks 保留；反过来的 interleaved
  checks 不属于该 logical-product sheet，不能乘进去。只允许原始 data patch
  stabilizer 作为 code-space equivalence 项。若 product 仍留下 ancillary/data
  residual Pauli，就返回 `verified=False` 并列出 `residual_terms`。这避免把
  本来没有自然抵消的 boundary Pauli 人为截断后误报成功，也避免把不属于当前
  sheet 的 X/Z checks 误乘进去。
- product 分解器仍可以传入 `ancilla_readout_bases` 做 residual diagnostic：
  如果 syndrome-only 失败，它能告诉我们还剩哪些 ancillary Pauli 没有被
  局部 checks 抵消。但这只是诊断，不是最终实现路径。

已经验证：

- paper-style long ancillary patch 的 `ZZZX` product algebra 已经通过：
  d=3 示例中，mixed-domain-wall bent ancilla 的 syndrome product 加 1 个
  `AncillaLogicalTerm` 后验证为目标 `ZZZX`。这个 logical factor 是 long
  ancilla 的已知边界逻辑量，不是一组独立 ancillary data readout。
- Z-normalized 的 `ZZZX` final-readout 路径已经通过 detector error model
  验证。在这个模式下，helper 等价于测 routed `ZZZZ`，并且只在目标为
  `X` 的 patch 上做 H。这个 legacy validated helper 当前显式使用
  `route_width=1`，以保持 tracker/DEM 验证闭合。
- mixed-template scaffold 已经测试了 mixed check 的生成、非 weight-4
  check 的存在，以及 active stabilizer set 的两两对易。
- mixed-check extraction 现在按 compatible stabilizer batch 执行；每个
  mixed syndrome 使用 X-basis ancilla，`Z` 项用 `CZ(data, syndrome)`，
  `X` 项用 `CNOT(syndrome, data)`，并对 batch 内 entangling edges 做
  layer coloring，避免 `detslice-with-ops-svg` 退化成一长串单 check 的细线图。
- native `X1Z2` 的 product algebra 已经测试通过：coupler checks 加 active
  patch stabilizer correction 后，求解器可以 syndrome-only 验证最终乘积等于
  `X1Z2`。
- 四 patch native-interface `ZZZX` mixed full ancillary patch 之前的
  “20 个 measured local merge checks 验证通过”结论是错误的：那条路径把
  一些 ancillary boundary Pauli 从 local check 中 trim 掉了。后来“全选所有
  coupler checks”的 strict diagnostic 也不正确，因为它会把 Z-side 上的
  interleaved X checks、X-side 上的 interleaved Z checks 一起乘进去。当前
  paper-style long-ancilla path 会先过滤到正确 product sheet，并用原始
  patch stabilizer 扣除 code-space equivalence；在 d=3 示例中剩下的是
  long ancillary patch 的一个已知 logical boundary factor，而不是额外 data
  readout。

尚未宣称完全完成：

- patch-span 的 routed ancillary region 已经用于 direct mixed coupler
  geometry；但 patch-span native mixed tracker/observable 与 detector error
  model schedule 还没有完成闭合验证，所以 notebook 里不再宣称 full mixed
  `detslice-with-ops-svg` 是最终 DEM 图。
- 当前 unrotated SE block 中的 mixed-check extraction 已经不是单 check
  串行；它会 batch compatible mixed checks 并 layer CNOT edges。不过这仍
  不是最终宣称 fault-tolerant 的 mixed-boundary lattice-surgery schedule。
- 当前已经验证的是这些具体 routed mixed 几何和 unrotated d=3 示例；更复杂
  的多分支、不同距离、不同 route order 仍应逐个用 product decomposition、
  tracker observable 和 detector error model 验证。
