# AutoDEM ASPLOS Revision Plan (30-Hour Sprint)

**目标:** 把 borderline reject 的版本提升到 solid accept。

**策略:** 优先级驱动,先做高 ROI 改动,后做 polish。每个 Phase 结束后状态都是"可投出"的——这样即便后面 phase 来不及,也不会死。

---

## ⏱️ 时间预算总览

| Phase | 时长 | 性质 | 完成后状态 |
|---|---|---|---|
| Phase 0 | 1h | 准备 + verify | 知道战场实际情况 |
| Phase 1 | 5h | 必须修复的 blocker | 不会被一眼毙掉 |
| Phase 2 | 8h | 核心证据补充 | 审稿人能信任工作 |
| Phase 3 | 8h | Sec.5-6 重构 | 论文有 systems paper 的骨架 |
| Phase 4 | 5h | Quantification + 内容补充 | 论证完整 |
| Phase 5 | 3h | Final polish | 可投出 |
| **Total** | **30h** | | |

---

## 🔴 Phase 0: Preparation & Verification (1h)

**目的:** 在动手前先搞清楚战场实际状况。两件事如果发现问题,后续 plan 要调整。

### Action Items

#### P0-1. 处理 arxiv 2504.05611 (10min) [已确认]
- **已确认情况:** 该 paper(合作者工作)实现了 Bell teleportation 但**没有开源 code**——这正是本领域现状
- **行动:**
  - Table 1 "Bell Teleportation" 行措辞改为:**"Closed-source impl. [2504.05611]"**(而非 "Not available")
  - "AutoDEM's Added Value" 列写:**"First open-source unified TG+LS implementation"**
  - 在 Sec. 5.1 (C) 末尾或 Related Work 中合适位置 cite [2504.05611]
- **关键叙事:** 强调 "open-source + unified",而不是 "first to implement"——这样既尊重合作者工作,又保留 AutoDEM 的贡献定位
- 这一项是**领域现状的有力佐证**:连这种重要工作都没有开源 code,正说明 AutoDEM 这种 unified framework 的价值

#### P0-2. 核实 correctness(差异化对比策略) (35min)
- **背景:** 之前已查过 rotated surface code 的 detector 和 logical observable 数量与 Stim built-in match;现在要把这一点系统化并扩展到 BB code
- **可对比的 reference 现状:**
  - **Stim built-in:** 只有 rotated surface code memory 可以做 bit-exact 对比
  - **BB code:** **使用 [gongaa/SlidingWindowDecoder](https://github.com/gongaa/SlidingWindowDecoder)** 作为对比 reference
    - 该 repo 有 `build_circuit.py` 专门构建 BB codes 的 Stim circuit,复现了 IBM BB code paper [10] 的 Figure 3
    - 用 `IBM.ipynb` 拿 [[72,12,6]] code 的 detector annotations 数量作为 baseline
    - 预期:**AutoDEM 应该多几个 detector per round**(自动发现了 linearly dependent stabilizers 带来的额外 detectors)
  - **其他协议:** 没有可对比的开源 reference,只能依靠 P2-1 的 "correctness invariant argument" + LER 合理性
- **行动:**
  1. 跑 Stim built-in vs AutoDEM 的 rotated surface code memory(d=3,5,7),记录:detector count、observable annotations、noise injection 完全 bit-exact 一致
  2. 用同 random seed 跑 LER,确认差距纯属 Monte Carlo 涨落
  3. clone gongaa/SlidingWindowDecoder,跑 BB code [[72,12,6]] 的 build_circuit,数 DETECTOR annotation 数量
  4. 跑 AutoDEM 同样 [[72,12,6]] BB code memory,对比 detector count
  5. 如果时间允许,扩展到 [[108,8,10]] 和 [[144,12,12]]
- **输出 `correctness_diff_log.md`:** 记录上述对比的具体数字,这些数据会直接喂给:
  - P2-1 的 Correctness Validation 表格
  - P4-1 的 BB code "额外 detectors" 量化
- **重要 framing:** Sec. 6 的 "Correctness Validation" 要诚实地说"我们对 Stim built-in 做 bit-exact 对比;对 BB code 通过 gongaa/SlidingWindowDecoder 对比 detector count;对其他协议依赖 invariant argument"——**别 overclaim**

#### P0-3. 检查现有 codebase 能跑出哪些数据 (15min)
- 列出当前能立刻产出的数据:
  - 各协议的 Stim annotation LoC(从已有 output 文件 counting)
  - AutoDEM 的 atomic operations LoC
  - 已有 BB code 实验的额外 detector 数量
  - 编译时间(如果有 log)
- 输出 `available_data.md`,列出"现有"vs"需要新跑"

---

## 🔴 Phase 1: Pre-Submission Blockers (5h)

**目的:** 修复任何"一眼就被毙"的问题。完成后论文至少不会被 desk reject 或第一轮直接 cut。

### Action Items

#### P1-1. 修复 Sec. 6.2 的 Litinski/Gidney 攻击性措辞 (30min)
- **位置:** Sec. 6.2 "Key Insights" 段落
- **当前:** "an approach widely used in existing literature [31, 56] that can be oversimplified and even misleading"
- **改为:** 大致以下意思(用你自己的话润色):
  > "Memory-baseline approximations [31, 56] provide valuable resource estimates at scale. Our gate-specific measurements complement these by exposing operation-level overhead (up to 11.6× for S_TG), suggesting that incorporating gate-level corrections may further refine end-to-end fidelity estimates."
- **同时检查:** 全文搜索任何对其他工作的 negative 措辞,改成 "complement/extend/refine" 语气

#### P1-2. 修正 Table 1 基于 P0-1 的发现 (30min)
- 根据 `verification_notes.md` 修改 "Bell Teleportation" 行
- 如果该 arxiv paper 确实实现了:改成 "Limited TG-only [arxiv ref]" + "AutoDEM's Added Value" 列改为 "Unified TG+LS comparison + routing analysis"
- 如果只做了 memory 或不相关:保持但加一个脚注说明已检查

#### P1-3. 移除所有 Todo 和 polish 图表 (2h)
- 全文 grep `Todo:` `\todo` `FM:`,逐个解决或删除
- 重新生成字号过小的图:Fig. 6 (CrossLS), Fig. 7(c+d), Fig. 8 caption 二义性, Fig. 9 (Bell tele), Fig. 10 (CrossLS)
- 补全 Fig. 9(b2), 9(b3) — 如果数据没跑完,至少占位说明
- Fig. 1 的 Stim circuit 截图重做或简化(当前不可读)

#### P1-4. 修复 Abstract 和 Conclusion 的 overclaim (15min)
- Abstract grammar: "sets" → "set", "demonstrate" → "demonstrates"
- Conclusion: "long limited" → "limited evaluation beyond basic memory experiments"
- Conclusion 加一句 quantitative summary:"AutoDEM uncovers gate-specific overheads up to 11.6× over memory baseline and enables CrossLS to suppress LER by 2-3 orders of magnitude below physical error rate."

#### P1-5. 补 Fig. 3 的 in-text reference (5min)
- 在 Sec. 2.3 引入 memory experiment 处加 "(Fig. 3)"

#### P1-6. 写 Related Work section (1.5h) ⭐
- 新增 Sec. 2 之前或之后的 Related Work section
- **三个 subsections:**

  **(a) QEC Simulators and DEM Construction.**
  - Stim [30] 做 frame tracking,但需要 user 手工标注 DETECTOR/OBSERVABLE
  - 列出 AutoDEM 的精确 delta:从 physical circuit 自动 derive annotations
  - 提 Stim 的 `with_inlined_feedback` 等周边工具,说明它们解决的是不同问题

  **(b) Automated Detector Construction for Clifford Circuits.** ⭐ 新增,这是最重要的 baseline 对比
  - **Spacetime codes [Delfosse & Paetznick, arXiv:2304.05943]:**
    - 提出 outcome code / spacetime code formalism,从任意 Clifford circuit 自动 derive 一组 checks
    - 与 AutoDEM 共享"自动化方向",但**抽象层级不同**:
      > "Spacetime codes [2304.05943] work at the circuit level: given any Clifford circuit, they derive an abstract set of checks via outcome-code analysis. AutoDEM operates at the protocol level: it consumes high-level atomic operations (initialization, syndrome extraction, lattice surgery couplers, transversal gates, data readout) and produces simulation-ready Stim DEMs with detectors, logical observables, and feed-forward Pauli corrections fully tracked. The two approaches are complementary—spacetime codes provide a circuit-level mathematical foundation, while AutoDEM provides the protocol-level toolkit needed for end-to-end FTQC evaluation."
    - **关键:不要说他们"produce non-local edges"——这不准确**。论文明确指出 D-dim local codes → D+1 dim local generators
  - **Stabilizer circuit verification [Kliuchnikov, Beverland & Paetznick, arXiv:2309.08676]:**
    - 一句话提及作为 broader context:"Related work on stabilizer circuit verification [2309.08676] shares the underlying Pauli tableau formalism but targets equivalence checking between circuits rather than DEM construction."

  **(c) Pauli Tableau Methods.**
  - Aaronson-Gottesman [1] 和后续工作
  - 区别:他们做 state simulation,你做 DEM construction
  - 你的 novel 点:record-augmented tableau

  **(d) QEC Protocol-Specific Implementations.**
  - 列举 Categories A-C 里各 protocol 的零散实现:
    - qLDPC repo [76], chromobius [78] for color codes, gongaa/SlidingWindowDecoder for BB codes
    - Static Stim files [89] for transversal gates
    - Closed-source impl. [2504.05611] for Bell teleportation
  - 说明这些是 isolated efforts, no unifying framework
  - 这正是 AutoDEM 提供 unified codebase 的价值所在

---

## 🔴 Phase 2: Core Evidence (8h)

**目的:** 补上 ASPLOS 最看重的三块证据:correctness, performance, productivity。这是整个 revision 的核心 ROI。

### Action Items

#### P2-1. Differential Testing & Correctness Validation (3h) ⭐⭐
- 新增 Sec. 6 开头一个 "Correctness Validation" 小节(半页)
- 内容:
  - 一张表:协议 / Stim reference exists? / Detector count match? / Observable support match? / Edge weight match? / LER agreement
  - 覆盖至少:rotated/unrotated/toric memory, LS CNOT (用[70]作 reference), TG CNOT (用[89]作 reference)
  - 基于 P0-2 的发现,**清晰解释**任何差异
  - 报告 shot count 和 95% CI,把 "15%" 换成有统计意义的数字
- 加一段 "Correctness Argument":
  > "By construction, the Pauli tracker maintains the invariant that measurement records always reflect the current Pauli frame after every Clifford block (Sec. 3.2). Under this invariant, any Pauli decomposition emitted by Algorithm 1 (Case B) yields a deterministic XOR sum under noiseless execution, and any decomposition in Algorithm 2 yields the correct logical observable value. Differential testing against Stim references confirms this invariant in practice."
- 修复 Sec. 6.1 的 "exact match but 15%" 矛盾,改用 P0-2 得到的清晰说法

#### P2-2. Productivity Evaluation (2h) ⭐⭐
- 新增 Sec. 4 末尾或 Sec. 5 开头一个 "Productivity" 小节(半页)
- **核心:一张表 + 一个 side-by-side code 对比**

  **表格:**

  | Protocol | Stim Annotation LoC | AutoDEM API LoC | Prior Open Impl? |
  |---|---|---|---|
  | Surface memory d=5 | ~? | ~5 | Yes [Stim] |
  | LS CNOT d=5 | ~? | ~8 | Limited [70] |
  | TG CNOT d=5 | ~? | ~10 | Static [89] |
  | Bell Teleport (LS) | N/A | ~12 | None |
  | Steane Distillation | N/A | ~20 | None (LS) / Static [89] (TG) |

  - 数据从已有的 Stim output file 数行(只数 DETECTOR/OBSERVABLE_INCLUDE/REPEAT block 里的 annotation,不数物理操作)

  **Code snippet (Listing 格式,半栏):**
  ```python
  # AutoDEM: full LS CNOT protocol
  sys = QECSystem()
  patch1, patch2 = sys.add_patches([SurfaceCode(d=5)] * 2)
  sys.initialize({patch1: '+', patch2: '0'})
  sys.syndrome_extraction(rounds=d)
  sys.activate_coupler(patch1, patch2, type='ZZ')
  sys.syndrome_extraction(rounds=d)
  sys.deactivate_coupler()
  sys.data_readout({patch1: 'Z', patch2: 'X'})
  # All detectors and observables auto-derived
  ```
  - 配一句话:"Equivalent Stim implementation requires ~N lines of manual DETECTOR/OBSERVABLE annotation, with O(d) detectors per round requiring careful tracking of feed-forward Pauli corrections."

#### P2-3. Compilation Performance (3h)
- 新增 Sec. 6 一个 "Compilation Performance" 小节(半页 + 一张图)
- 跑一组 benchmark:
  - 横轴: code distance d ∈ {3, 5, 7, 9, 11} 或 qubit count
  - 纵轴: AutoDEM compilation time (seconds), 对比 Stim circuit generation time
  - 测试 3-4 个 protocol: surface memory, BB memory, LS CNOT, Steane distillation (TG)
- 报告:
  - Per-measurement 复杂度: O(n²) for tableau update, O(n³) for RREF in Case B
  - 实际数据:"For [[144,12,12]] BB code memory at d=12, AutoDEM compiles the full circuit + DEM in X seconds, comparable to Stim's circuit generation time of Y seconds."
- 一句话 scaling 结论:"AutoDEM's compilation overhead is dominated by O(n³) RREF operations per detector, which remains tractable up to ~200-qubit codes and ~100 rounds based on our measurements."

---

## 🟠 Phase 3: Sec. 5-6 重构 (8h)

**目的:** 把 Sec. 5-6 从 "QEC paper" 风格转成 "systems paper" 风格。压缩物理分析,腾出空间给系统证据。

### Action Items

#### P3-1. 压缩 Sec. 6 (Evaluation) 物理分析 (3h) ⭐
- 当前 Sec. 6 占约 4 页(Sec 6.1-6.4),压缩到 **2.5-3 页**
- **保留:** 每个 protocol family 的核心 LER 曲线(Fig. 7-10) + 1-2 句的 key takeaway
- **删除/移到 Appendix:**
  - Sec. 6.2 关于 S_TG 为何 overhead 高的详细物理分析(CZ propagation argument)→ 移到 Appendix B
  - Sec. 6.3 关于 ZZ-LS vs XX-LS feed-forward correction 的详细机理 → 保留一句结论,具体推导移到 Appendix B
  - Sec. 6.4 关于 PQRM Z-distance 限制的详细分析 → 保留结论,机理移到 Appendix B
- **每个 subsection 末尾用 inline "Key Findings"**——**不用 bullet points**(节省空间),用 LaTeX 格式:
  ```latex
  \noindent\textbf{Key Findings.} (1) Gate-specific overhead can exceed memory baseline by up to 11.6$\times$ (S$_{\text{TG}}$); (2) LS routing distance contributes linearly to LER for state-dependent feed-forward; (3) CrossLS reveals structural Z-distance bottleneck only visible at end-to-end level.
  ```
- 这种 inline 格式比 bullet list 节省至少 30% 空间,且更紧凑,适合 ASPLOS 版面

#### P3-2. 把 Phase 2 的三块新内容融入 Sec. 5-6 框架 (2h)
- Sec. 4 末或 Sec. 5 开头插入 "Productivity" (P2-2)
- Sec. 6 开头插入 "Correctness Validation" (P2-1)
- Sec. 6 中段插入 "Compilation Performance" (P2-3)
- 重排 Sec. 6 结构为:
  ```
  6.1 Correctness Validation (新)
  6.2 Compilation Performance (新)
  6.3 Memory Experiments (压缩版原 6.1)
  6.4 Logical Operations (压缩版原 6.2)
  6.5 Logical Circuits (压缩版原 6.3)
  6.6 CrossLS Case Study (压缩版原 6.4)
  ```

#### P3-3. ActivateCoupler 机制扩展 (1h)
- **位置:** Sec. 4.2 atomic operation #3
- 当前: "AutoDEM handles this automatically via the QEC System"
- 改成 2-3 句具体描述:
  > "When a coupler is activated, the QEC System introduces new boundary stabilizers into the tableau. These stabilizers initially anti-commute with existing rows, triggering Algorithm 1's Case A: pivot rows are replaced and no detector is emitted (matching the physical reality that merge-round outcomes are random). After d_coupler rounds of syndrome extraction, the merged stabilizers stabilize and subsequent measurements transition to Case B, emitting deterministic detectors. This natural reuse of Algorithm 1 across phases—rather than special-case handling—is what makes lattice surgery DEM construction tractable."

#### P3-4. 明确 Protocol-Agnostic 的 scope 边界 (45min)
- **位置:** Sec. 3.1 Overview 末尾,加一个 "Scope and Assumptions" 小段
- 内容:
  > "AutoDEM applies to any protocol expressible as: (i) stabilizer codes (CSS or non-CSS), (ii) Clifford circuits augmented with resource-state injection, (iii) static qubit topology (no mid-circuit code transformation), and (iv) measurement-based feed-forward corrections. Subsystem code gauge fixing, dynamically generated codes (e.g., Floquet codes), and runtime qubit recycling beyond the patterns covered in Sec. 4.2 are out of scope and listed as future extensions in Sec. 7."

#### P3-5. CrossLS 定位降级 (45min)
- 在 Sec. 5.2 CrossLS introduction 加一句:
  > "We present CrossLS as a case study demonstrating AutoDEM's rapid prototyping capability for novel cross-code protocols, rather than as a fully optimized protocol contribution. A complete protocol-level analysis (e.g., resource accounting vs. magic state cultivation [33]) is beyond the scope of this paper."
- 删除或缓和 Sec. 5.2 末尾 "highly space-efficient" 等 unquantified claims

#### P3-6. WriteBack 作为 design choice (30min)
- **位置:** Sec. 3.3 step 3
- 重新 framing,从 "side effect" 改成 "deliberate design":
  > "WriteBack is a deliberate design choice: by re-expressing the tableau in the freshly measured basis after each SE block, we enforce that detectors only reference temporally-local measurement records. Without this constraint, Pauli decompositions could span arbitrarily many rounds, producing non-local DEM edges that severely degrade matching-based decoder performance. This locality enforcement is what makes AutoDEM-generated DEMs compatible with off-the-shelf decoders like PyMatching."

---

## 🟡 Phase 4: Quantification & Content Completion (5h)

**目的:** 把所有"提到但没量化"的 claims 补上具体数字。完成后论文论证完整。

### Action Items

#### P4-1. BB code 额外 detectors 量化 (1h) ⭐
- 这是你最有力的 "automation > manual" 证据,不能只一句话
- 在 Sec. 6.3 (新 Memory Experiments) BB code 部分扩到一段
- 量化:
  - 多了多少个 detector? (e.g., "AutoDEM discovers N additional detectors per round in [[144,12,12]] BB code, a M% increase over manual [10] construction")
  - 这些 detector 对 decoding 有帮助吗?跑一个对照实验:有/无这些 detectors 的 decoder accuracy 差异

#### P4-2. CrossLS 数字锚定 (1h)
- E7: "2-3 orders below PER" → "at d_surf=7, p=10⁻³, Z-LER for PQRM(1,2,4) reaches X, which is Y orders below PER"
- E8: "X-stabilizer post-selection partially mitigates this" → 给出具体 PS rate 和 LER 改善
- S4: "highly space-efficient compared to cultivation" → 给出 qubit count 对比,或缓和措辞为 "compact"

#### P4-3. State Injection 和 Distillation 的数字 (1h)
- E5: 加一句解释为何 "linear with p, independent of d"(injection error happens before QEC kicks in)
- m-2: 1.2× PS improvement → 指明哪个 d, p
- **Distillation 范围调整:只 present injection-only noise 的 case**(full noise 数据不好看)
  - 删除 Fig. 9(b2), 9(b3) 的 full noise 子图(原本就缺失/未跑完,正好直接去掉)
  - 保留 Fig. 9(b1) injection-only noise,展示 LER 沿 7p³ 曲线的良好 fit——这本身就是强 correctness validation
  - 在 Sec. 6.5 (新 Logical Circuits) distillation 部分诚实说明:"We focus on injection-only noise to validate the correctness of our distillation circuit construction (LER closely tracks the theoretical 7p³ curve for both TG and LS, Fig. 9b). Full circuit-level noise evaluation is left to future work, as it requires significantly larger code distances to make distillation gain visible above circuit noise."
  - **这种 framing 把 limitation 转化为 future direction**,不是露怯,而是 scope clarity

#### P4-4. S_TG overhead reference 和 Bell tele depth normalization (1h)
- E4: S_TG 的 11.6× 是你的 original measurement 还是 matching prior work?明确说
- E6: Bell tele overhead 是否需要 depth-normalize?加一句 acknowledge:
  > "These overheads are reported at fixed (d, p) without depth normalization. Different circuits have different SE round counts; depth-normalized comparisons would yield smaller spreads but obscure the per-protocol total cost that matters for end-to-end algorithm execution."

#### P4-5. Intro 的 Stakes Paragraph (1h)
- **位置:** Sec. 1 三个 challenges 之后
- 加一段:
  > "These challenges have concrete consequences for the field. Prior to this work, no open-source tool provides circuit-level evaluation for any protocol in Categories B, C, or D of Table 1. As a result, architectural decisions around lattice surgery vs. transversal gates, choice of distillation scheme, and routing strategies have largely been guided by theoretical bounds [56] or isolated hardcoded data points [89], rather than by systematic, fair circuit-level comparison. The absence of automated DEM construction is the root bottleneck."
- 修改 Contribution list,把至少一条改成 impact statement:
  > "**Architectural insights inaccessible from memory simulation alone:** We show that gate-specific LERs exceed memory baselines by up to 11.6×, that LS routing distance contributes linearly to teleportation LER, and that cross-code protocols can exhibit state-dependent structural bottlenecks—findings that inform near-term FTQC architecture decisions."

---

## 🟢 Phase 5: Final Polish (3h)

**目的:** 收尾工作,投出前的最后 check。

### Action Items

#### P5-1. 全文一致性 Pass (1h)
- 检查所有 section/figure/table/equation/algorithm reference
- 检查所有 citation 格式一致
- 全文一遍 grammar pass(focus on Abstract, Intro, Conclusion——审稿人最仔细读的部分)

#### P5-2. 处理剩余 minor issues (1h)
- m-3, m-8, m-15, m-19, m-20: 如 P3 里没顺手修,这里批量处理
- B3, B4: Background 里的 Pauli rules 加 intuition
- B5: PyMatching/BPOSD/MWPF 在 Background 处第一次提及时加 one-liner gloss
- T4: Algorithm 1 `buf` data structure 加注释或 caption explanation
- T5: Sec. 3.4 加一句 edge case 说明
- T6: Sec. 3 开头加 appendix pointer
- m-13: decoder hyperparameter 选择加一句 justification
- m-12: PQRM transversal T 的 mathematical reason 加一句

#### P5-3. Final Read-Through (1h)
- 从头到尾读一遍,模拟审稿人视角
- 核查 Abstract <-> Contribution <-> Conclusion 三处的 claim 一致
- 核查所有数字和单位
- 准备 supplementary materials (codebase link, reproduction scripts)

---

## 📋 Execution Notes for Claude Code

### 优先级
- 🔴 **Phase 0-2** 是 must-do,占 18 小时,这部分完成就能从"很可能 reject"翻到"借走"
- 🟠 **Phase 3** 是 high-ROI,占 8 小时,完成后变 solid accept candidate
- 🟡🟢 **Phase 4-5** 是锦上添花,时间不够可以适度 cut(P4 里只保留 P4-1, P4-5 最关键)

### Failure Modes / Decision Points
- 如果 P0-1 发现 arxiv 2504.05611 完全做了你的 Bell tele,立刻调整 Sec. 5.1 的 framing,不要硬抵
- 如果 P0-2 发现真 bug,**停下所有其他工作**,先修 bug
- 如果 P2-3 的 compilation performance 数据非常难看(e.g., 比 Stim 慢 100×),不要硬报数字,改成 complexity analysis + qualitative discussion
- 如果 Phase 3 的物理分析删不下来,优先保证 Phase 2 的三块新内容进得去——可以让 paper 超页一点点(ASPLOS 允许 appendix overflow)

### Checkpoint Strategy
- Phase 1 结束: commit + tag `revision-phase1-blockers-fixed`
- Phase 2 结束: commit + tag `revision-phase2-evidence-added`
- Phase 3 结束: commit + tag `revision-phase3-restructured`
- Phase 5 结束: commit + tag `revision-final-submission`

每个 checkpoint 状态都是"可投出"的——这是为了如果时间用完,至少有最近的 checkpoint 可以提交。
