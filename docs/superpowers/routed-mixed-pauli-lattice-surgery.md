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

ancilla 区域不是只能是单纯竖直或单纯水平 corridor。更一般地，它应当
是一个 routed、basis-labeled 的 ancillary region。

参与测量的 data patch 可以放在任意位置，只要被选中的 boundary strip
能够通过一条 Manhattan route 连通，并且这条 route 不穿过已有 patch
占据的坐标。

route 上的每个坐标会被标记成离它最近的 interface basis。这样同一个
ancilla region 内会自然出现 `X` 区域、`Z` 区域，以及二者交界处的
mixed X/Z seam。

## Stabilizer 模板规则

stabilizer 必须根据局部 labeled geometry 生成，而不是整条 bus 共用同
一种模板：

- 纯 `X` 邻域生成普通 X check。
- 纯 `Z` 邻域生成普通 Z check。
- `X/Z` seam 处生成 mixed XZ check。
- endpoint、拐弯、被裁剪的局部邻域可能自然产生 weight-2 或 weight-3
  check，不能假设所有 ancilla stabilizer 都是 weight 4。

因此，像 `X1` 和 `Z2` 中间有多个 ancilla patch，或者 route 发生拐弯
的情况，stabilizer 类型和 weight 都会依赖局部几何；不能把整条 bus
粗暴地看成同一种 `XXXX` 或 `ZZZZ` 模板。

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
  patch origin 之间的差必须是 `route_width` 的整数倍，并且每个 patch 本身
  必须刚好占据一个 `route_width × route_width` 坐标块。此外现有
  unrotated lattice 还要求 syndrome parity 对齐；对 d=3 来说
  `route_width=5` 是奇数，所以同方向 patch 的常用安全间距是 10、20、30、
  ... 个整数坐标，中间空出的 5×5 coarse cell 才是 ancillary patch block。
- routed coupler 会先在这个 coarse grid 上做 Manhattan BFS，避开被 data
  patch 占据的 coarse cells；然后把每个 route cell 展开成完整
  `route_width × route_width` ancillary block。因此 selected interface 的
  terminal region 和中间拐弯/走廊都是由完整 patch-sized blocks 拼成的，
  不再是 thin skeleton 膨胀出来的非规整形状。
- interface basis 可以由选中边自动推断，也可以显式传入。
- routed Pauli-product helper 会比较 target Pauli 和 native interface
  Pauli，只在不匹配的位置插入 logical H。
- `mixed_stabilizers=True` 时会启用实验性的 mixed X/Z local stabilizer
  模板生成。
- mixed-template 路径会用代数方式暂停与 routed check 反对易的原 patch
  stabilizer，使测试过的 active stabilizer set 保持对易。
- `solve_routed_pauli_product_syndromes` 可以自动求解并验证目标 logical
  product 的 outcome 分解。它会先尝试纯 syndrome outcomes；如果 full
  ancillary patch 几何留下 ancilla 自身的 boundary/logical 因子，则会额外
  返回 `selected_ancilla_terms`，表示需要乘入的 ancilla data readout 或
  known-boundary terms。对于当前测试的 full patch-span `X1Z2` 几何，它会
  返回 coupler stabilizer outcomes、active patch stabilizer correction
  outcomes，以及必要的 ancilla terms。
- `routed_coupler_data_basis` 会根据 routed ancillary region 的局部
  `route_coord_basis` 生成物理 basis map：
  - `mode="opposite"` 用于初始化：`Z` route 区域准备在 `X`，`X` route
    区域准备在 `Z`。
  - `mode="same"` 用于读出：`Z` route 区域读 `Z`，`X` route 区域读 `X`。
- product 分解器现在可以传入 `ancilla_readout_bases`，只允许使用实际会被
  读出的 ancillary data Pauli，避免数学上同时使用同一 qubit 的 `X` 和 `Z`
  readout。

已经验证：

- Z-normalized 的 `ZZZX` final-readout 路径已经通过 detector error model
  验证。在这个模式下，helper 等价于测 routed `ZZZZ`，并且只在目标为
  `X` 的 patch 上做 H。这个 legacy validated helper 当前显式使用
  `route_width=1`，以保持 tracker/DEM 验证闭合。
- mixed-template scaffold 已经测试了 mixed check 的生成、非 weight-4
  check 的存在，以及 active stabilizer set 的两两对易。
- mixed-check extraction 现在按 compatible stabilizer batch 执行，并对
  batch 内 CNOT edges 做 layer coloring，避免 `detslice-with-ops-svg`
  退化成一长串单 check 的细线图。
- native `X1Z2` 的 product algebra 已经测试通过：单独使用 coupler checks
  不足以生成目标 product；加入 active patch stabilizer correction 后，
  求解器会在限制到实际 route-basis readout 的条件下验证最终乘积等于
  `X1Z2`。是否还需要额外 ancilla readout terms 取决于具体 routed geometry；
  当前规整 terminal-block 几何下不再强制需要这些项。
- 四 patch native-interface `ZZZX` mixed full ancillary patch 已经能走完整
  tracker/observable 流程：ancilla 用 conjugate basis 初始化、用 route basis
  读出，`if_detector=True` 时可生成 1 个 mixed-surgery observable，并通过
  detector error model。`routed_ZZZX_LS.ipynb` 的 Exp 2 会显示完整
  `detslice-with-ops-svg` 彩色图。

尚未宣称完全完成：

- patch-span 的 routed ancillary region 已经用于 direct mixed coupler
  geometry；但 patch-span Z-normalized final-readout helper 还没有完成
  tracker/DEM 验证，所以 validated helper 暂时保留 thin route 选项。
- 当前 unrotated SE block 中的 mixed-check extraction 已经不是单 check
  串行；它会 batch compatible mixed checks 并 layer CNOT edges。不过这仍
  不是最终宣称 fault-tolerant 的 mixed-boundary lattice-surgery schedule。
- 当前已经验证的是这些具体 routed mixed 几何和 unrotated d=3 示例；更复杂
  的多分支、不同距离、不同 route order 仍应逐个用 product decomposition、
  tracker observable 和 detector error model 验证。
