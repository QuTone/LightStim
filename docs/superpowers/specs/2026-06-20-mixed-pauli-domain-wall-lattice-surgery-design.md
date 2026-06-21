# Mixed-Pauli Domain-Wall Lattice Surgery — 设计文档

- 日期: 2026-06-20
- 范围: **静态 stabilizer 构造的正确性**(不含电路 / DEM / fault-tolerant schedule)
- 先做: **X1Z2 两-patch 最小情形**,做对后再推广到 ZZZX 与 full-width
- 取代: `docs/superpowers/routed-mixed-pauli-lattice-surgery.md` 中关于 mixed
  product 验证的部分(那份记录里的 `AncillaLogicalTerm` 路径被本设计判定为错误)

---

## 1. 目标

支持在**不同位置的 data patch** 之间做混合类型(X/Z 任意组合)的 logical
product measurement,典型例子 `X1Z2`、`ZZZX`。本阶段只要求**静态构造**正确,
即:ancilla bus 作为一个 stabilizer code,其 check 集合在代数上严格满足下面四条
硬性要求,而不依赖任何"已知逻辑因子"之类的事后补救。

## 2. 背景:当前实现为什么是错的

当前 notebook (`routed_ZZZX_LS.ipynb`) 走 `mixed_stabilizers=True` +
`solve_routed_pauli_product_long_ancilla`,做法是逐 syndrome qubit 猜类型、加
boundary candidate、再 ad-hoc 剪枝反对易对,最后把乘不掉的残余打包成一个
`AncillaLogicalTerm` 声明为"已知边界逻辑量"。实测:

| 情形 | ancilla data | ancilla stab | 期望 stab (#data−1) | 残余支撑 |
| --- | --- | --- | --- | --- |
| ZZZX | 206 | 212 | 205 | 69 qubit (55 Z + 14 X) |
| X1Z2 | 48 | 52 | 47 | 20 qubit |

两个核心错误:

1. **自由度不是 1**:stab 比 data 还多,根本不是一个干净的单逻辑比特 code。
2. **假阳性验证**:`solve_routed_pauli_product_long_ancilla` 里
   `verified = all(term.owner == coupler_name for term in residual_terms)` ——
   只要残余 Pauli 落在 coupler 自己 qubit 上就报通过,**完全不检查残余是否真的
   等于目标 logical**。实际残余是 69 个 qubit 的一坨,被当成"逻辑因子"扫到地毯下。

## 3. 物理核心:domain wall 为什么必需

Lattice surgery **从不单独测出 data patch 的逻辑算符**。沿一条边界 merge,测到的是
`(data 逻辑) × (ancilla 在该边界上的算符)`:

- p1 端(X merge): `X1 · X_anc^left`
- p2 端(Z merge): `Z2 · Z_anc^top`

目标 `X1Z2` = 两者相乘 = `X1 Z2 · (X_anc^left · Z_anc^top)`。括号内 ancilla 残留
必须确定地 = +1。

- **同 basis(ZZ)**:两端是同一条 ancilla Z-string 的两半,乘积 = ancilla 的一个
  stabilizer (+1) → 自动抵消。这就是普通 surgery 简单的原因。
- **混 basis(XZ)**:`X_anc^left` 与 `Z_anc^top` 是 ancilla 自己**反对易**的两个逻辑
  (`X̄_anc`、`Z̄_anc`),乘积 = `Ȳ_anc ≠ I`,非确定值 → 残留消不掉。这就是当前"剩一坨"
  的根源。

**domain wall 把这两个反对易的端算符焊成同一条折线 string**,于是它们的乘积变成
`(同一算符)² = I` → 自动抵消 → 干净的 `X1Z2`。wall 上的 mixed check 就是"焊点"。
因此"中间出现 mixed stabilizer"不是可选项,而是物理必需。

被否决的替代方案:

- **corner 构造**(把 X 边界、Z 边界在一个凸角相接):角上 `X̄/Z̄` 仍是反对易两逻辑,
  需要把 ancilla 制备成 Y 本征态,昂贵,不采用。
- **H-trick**(对 X-target patch 做 logical H → 全 Z 测量):已被 DEM 验证、最简单,
  但**违反要求③**(全 Z),不是 native 混合,本阶段不采用(保留为 legacy 对照)。
- **twist defect**:wall 端点落在 bulk 时退化为 twist(weight-5),通用但过重;本设计
  通过让 wall 两端落在 bus 边界上来**避免** twist。

## 4. 硬性要求(形式化为断言)

针对 ancilla bus(作为独立 patch 计):

1. **自由度 = 1**
   - `len(bus.data) - 1 == len(bus.stabilizers)`
   - `gf2_rank(symplectic(bus.stabilizers)) == len(bus.data) - 1`
   - 所有 check 两两对易
2. **乘积 = 被测 logical**
   - 存在一组 check 子集,其乘积 ×(原始 data-patch stabilizer,作 code-space 等价)
     **精确等于** 目标 product 的 symplectic 向量,**residual == 0**
   - 不接受任何 "已知逻辑因子 / ancilla readout 残留" 作为通过条件
3. **类型混合**:`{'X','Z','MIXED'} ⊆ set(check types)`(不能全 X 或全 Z)
4. **wall 连通**:MIXED check 构成一条连通的 domain wall(不是散点)

## 5. 几何

因为 unrotated 约定下 **left/right = X 边界、top/bottom = Z 边界**(见
`code_patch.py`:logical X 是 x=0 竖直 X-string,logical Z 是 y=0 水平 Z-string),
X-target 接入边是竖直边、Z-target 接入边是水平边,所以 X1Z2 的 bus 天然是 **L 形**
(横臂贴 p1 的竖直 X 边界,竖臂贴 p2 的水平 Z 边界)。

```
   p1 ──X接口──►[ X-sector ]══ WALL(mixed) ══[ Z-sector ]
                                                   │
                                                 Z接口
                                                   ▼
                                                  p2
```

- X-sector:bus 长边是 Z 型,X-string 自 p1 走入;
- Z-sector:bus 长边是 X 型,Z-string 走向 p2;
- WALL:横贯一条直臂、**两端落在 bus 边界上**(→ 无 twist)的一列 mixed check。

最小化:用尽量小的 `route_width` 和靠近的 p1/p2,使 bus 仅几十个 qubit,便于手算与
严格断言。L 形(拐弯)按用户确认采用,不强行改直线。

## 6. 构造算法:Hadamard-region domain wall(已用真实格点验证)

核心原理:**domain wall = 对 bus 的一整片连通子区域 R(即 X-sector)做 transversal
Hadamard 共轭**。对每个 stabilizer,在 R 内的 qubit 上把 X↔Z 互换(R 外不动):

- 完全落在 R 内的纯 check → 纯类型翻转(X↔Z);
- 完全落在 R 外的 check → 不变;
- **横跨 ∂R 的 check → 自动变成 mixed**。

因为 Hadamard 是幺正变换,**共轭保持对易性**,所以这样得到的 mixed check **一定**和所有
邻居对易,且生成元个数、秩、逻辑数都不变 → `#data − 1 = #stab` 自动成立。这把"对易"
从一个需要手工凑的约束变成了**数学保证**。

实现:在 `UnrotatedRoutedMultiPatchCoupler` 中**新增一个干净的 domain-wall 构造方法**,
专供 mixed 情形,不改动 legacy `_init_stabilizers`。步骤:

1. 先把 bus 建成一个**合法的 CSS unrotated patch**(全对易、1 逻辑、`#data−1` 个 check)。
2. 选取 H-region `R` = 想要成为 X-sector 的连通子区域(靠 p1 的那一臂)。`∂R` 即 wall。
3. 对每个 stabilizer 在 R 内 qubit 上 X↔Z 共轭;`∂R` 上的 check 自动成 mixed。
4. **边界 merge check**:在 p1 的 X 接口、p2 的 Z 接口生成连接 bus 与 patch 的 check
   (复用 `_probe_and_create_stabilizer` 思路);类型由该侧 sector 决定。
5. 选 `R` 使 bus 的逻辑 string 在 p1 端呈 X、p2 端呈 Z(已验证:纯 Z 水平 string 经
   左半 R 共轭后 = 左半 X + 右半 Z)。

**验证记录**(`scratch_domain_wall_check.py`,d_z=7/d_x=3 的 33-qubit bus):

| 构造 | data/stab | rank | 反对易对 | 类型 |
| --- | --- | --- | --- | --- |
| baseline CSS | 33/32 | 32 | 0 | X:18, Z:14 |
| 孤立 mixed check | +1 | — | **4(崩)** | — |
| Hadamard-region wall | 33/32 | 32 | **0(全对易)** | X:15, Z:12, MIXED:5 |

mixed check 落在单一 wall 列(连通);逻辑 Z-string → 左 X / 右 Z 混合 string。

> **注**:wall 形状是 `∂R`,可以是**直线(竖/横)**,无需对角;只要 `∂R` 落在 bus 边界上
> 即无 twist。"裁剪"在干净 patch 上**不需要**(计数自动对);仅当把构造塞进现有过宽
> coarse-grid 几何、出现多余 qubit 时才可能需要,且那时也按"同类型边界 check 连 qubit
> 一起删"处理,不影响对易(等价于把 R 选得更贴合)。

## 7. 验证 harness(测试入口)

提供一个 `verify_mixed_bus(system, coupler_name, patch_names, target_paulis)`,返回
结构化结果并被测试断言:

- `dof_ok`: 第 4 节要求①(计数 + 秩 + 对易)
- `product_ok`: 要求②(residual == 0,严格)
- `types_ok`: 要求③
- `wall_connected`: 要求④
- 失败时给出诊断(多/少哪些 check、residual 落在哪些 qubit)

并把这些断言加入 `tests/test_protocols.py`(X1Z2 最小情形先行)。

## 8. 实施顺序

1. **X1Z2 最小 bus**:实现 domain-wall 构造方法,TDD 到四条断言全过。
2. **ZZZX 四-patch**:推广到树状多臂 bus(Z 主干 + 一条 X 臂带 wall)。
3. **full-width(route_width = 2d−1)**:把构造接到现有 coarse-grid + seam 几何上,
   断言保持全过。
4. notebook 改用新构造与严格 `verify_mixed_bus`,删除 `AncillaLogicalTerm` 假阳性路径。

## 9. 非目标(本阶段不做)

- syndrome-extraction 电路、detector error model、fault-tolerant schedule。
- twist-based(weight-5)通用混合测量;Y-type product。
- 对现有 legacy `_init_stabilizers` / H-trick 路径的改动(保留作对照)。

## 10. 风险

- ~~wall 的确切格点形式 / 对易~~ **已验证**:Hadamard-region 构造保证全对易、计数不变;
  wall 可为直线,`∂R` 落在 bus 边界即无 twist。
- **要求②(merge 乘积 = X1Z2 残余 0)尚未端到端验证**:需把 p1/bus/p2 真正接起来、
  加 merge check 后核对。这是实现第一步,用第 4/7 节断言驱动。
- **merge check 与 wall 的相互作用**:H-region 的边界若与 p1/p2 接口太近,merge check
  可能落在被共轭的 qubit 上。缓解:wall(∂R)放在远离两端接口的臂段。
- **推广到 full-width / coarse-grid 后计数漂移**:seam 行/列与 R 的交互、以及可能的
  裁剪需重新核对断言。

## 11. 最终实现(已验证 — 水平 X1Z2,不拐弯)

经过与 rotated(Fowler–Gidney FIG.9)和 unrotated 手绘标准 mixed stabilizer 的对照,
最终落地的构造**比从零造 mixed merge 简单且更稳**:

> **方法**:用代码库**已工作的同 basis `XX` merge**(`UnrotatedMultiPatchCoupler`,
> 两个标准 patch 水平并排),然后对 **p2 的所有 data qubit 做 transversal Hadamard**
> (= 对 p2 做 logical 旋转,X2↔Z2)。

要点:
- Hadamard 是幺正变换 → **共轭保持对易性与逻辑数**,所以 DOF、对易、无 twist 都是
  **数学保证**,不需要手工凑奇偶/计数(这避免了从零造 mixed merge 时反复踩的坑)。
- merge 测的算符 `X1·X2` 在 H 共轭后变成 **`X1·Z2`**;跨两 patch 的 coupler check
  自动变成 **mixed domain-wall check**(p1 侧 qubit 取 X,p2 侧取 Z),正是 unrotated
  标准 mixed stabilizer。
- 这就是论文"rotate the logical qubit"法在 unrotated 上的实现;Hadamard 把那半格
  dislocation 整个吸收掉,seam 是直的、无需显式 step。

**验证(d=3,`routed_ZZZX_LS.ipynb` 端到端跑通)**:

| 要求 | 结果 |
| --- | --- |
| ① `#data − rank = 38 − 37 = 1` | ✓ DOF=1 |
| ① 全对易 | ✓ 0 反对易对 |
| ② joint `X1·Z2` 被测;`X1/X2/Z2` 单独都不被测 | ✓ 真·joint |
| ③ pure-X / pure-Z / MIXED 都在 | ✓ MIXED=5 |
| ④ domain wall、无 twist | ✓ 5 个 MIXED check,0 个含 Y |

被测逻辑:`X̄1 = X on p1 @ {(0,0),(0,2),(0,4)}`,`Z̄2 = Z on p2 @ {(10,0),(10,2),(10,4)}`,
notebook 图中已标注。

**仍未做**:电路 / detector-error-model 层的 FT schedule;弯折 bus;`ZZZX`。前面第 6–10
节关于 domain-wall / Hadamard-region 的分析仍成立,本节是它在两-patch 水平情形的最简落地。
