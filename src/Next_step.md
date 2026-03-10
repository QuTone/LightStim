# Next Steps

1. **Fix BB code memory experiment [[144,12,12]] 卡住问题**
  已诊断：不是 Logical operator 计算的问题（非 bottleneck）。问题在 `tracker.py` 的 `process_final_measurement` 的 Step 3 最后。
2. **DETECTOR 的 "postselect" tag 功能** [done]
  实现并封装，考虑如何方便控制。
3. **测试带 Postselect 的 Injection Decoding** [done]
  Injection 场景下的 post-select decoding。
4. **Injection / Unencode 写成 Logical Operation** [done]
  把 Injection 及其逆过程 (Unencode) 封装成 Logical Operation，便于后续使用（尤其是 Distillation 实验）。
5. **BPOSD GPU 版本集成与 Decoder 输出优化** [done]
  - 在本框架下测试 BPOSD GPU 是否可跑通  
  - 优化 Decoder 的 data output  
  - 测试 [[72,12,6]] BB code 和 Surface code
6. **Rotated Surface code two-patch coupler**
  参照 Unrotated Patch Coupler，实现 ZZ/XX merge、Lattice surgery CNOT。  
   Paper: Austin Fowler, Craig Gidney — Lattice surgery paper.
7. **Rotated/Unrotated surface code enlarging/shrinking**
  水平、垂直、对角线方向的 enlarging/shrinking，封装成 Logical op。
8. **Y state Injection 与 Measurement 实验** (Skip)
  先用 Injection 的逆过程 Unencode，再对单 qubit 做 Y basis 测量。
9. **Distillation circuit**
  基于 [[8,3,2]] 与 [[15,1,3]]，用 Y state 替代原有 magic state，实现 distillation circuit。
10. **Code-deformation 方式的 Y basis Measurement 实验** (skip)
  基于 XZ boundary 的变形；可能需要新的 syndrome extraction block。
11. **Multi-qubit surface code coupler**
  实现 PPM（Parallel Patch Merge）的实验。
12. **Color code 三种 scheduling 与 decoder 测试**
  用 MWPF 和 BPOSD decoder 测试 color code 的三种 scheduling。
13. **完善decoding Output的Visualization**
14. **Surface Code Transversal Gate Set**
15. **用tag控制selective Noise Injection**
16. **Fold-transversal Gates** Logical H, S

