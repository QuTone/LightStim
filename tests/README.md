# LightStim Test Suite

## 快速上手

```bash
# 安装测试依赖
pip install pytest pytest-timeout

# 运行所有 CI 测试（89个，约30秒）
pytest tests/ -m "not slow" --timeout=90 -q

# 只跑最核心的 smoke 测试
pytest tests/ -m smoke --timeout=60 -q
```

## 测试分层

| 标记 | 说明 | 运行时机 |
|---|---|---|
| `smoke` | 最核心的不变量检查，< 30s | 每次 commit |
| *(无标记)* | 中等速度的集成测试 | PR merge 前 |
| `slow` | 耗时较长（> 1min） | 手动运行或发版前 |

CI 只跑 `not slow` 的部分（目前 89 个 tests，约 30 秒）。

## 文件说明

| 文件 | 内容 | 速度 |
|---|---|---|
| `test_protocols.py` | 全部 17 个 protocol 的 noiseless build；验证 `num_detectors > 0`、零 detection events、DEM 可构建；包含两个数学不变量（两 patch 系统有 2 个逻辑比特；S⁴ = I） | 快 |
| `test_pipeline.py` | noisy circuit → SimulationPipeline → LER > 0；验证噪声注入和解码链路没有断 | 快 |
| `test_export.py` | `lightstim.frontend.export_all()` 输出的 schema 验证；保证前后端 JSON 接口稳定 | 快 |
| `test_api.py` | FastAPI endpoint smoke test（TestClient，无需启动真实 server） | 快 |
| `test_back_propagated_pauli.py` | Tracker Clifford 追踪的单元测试（`SyndromeTracker` 核心逻辑） | 快 |
| `test_simulation_backend_quality.py` | Decoder backend 边界行为：未知 decoder 报错、post-selection、`list_decoders()` 去重 | 快 |
| `color_code/test_color_code.py` | 色码数学性质：稳定子对易性、逻辑算符权重、qubit 数量 | 快 |
| `test_run_memory.py` | `benchmarks/memory/run_memory.py` 的 CLI 端到端测试；大部分标记为 `slow` | 慢 |

## 最核心的不变量

LightStim 的核心价值是**自动生成正确的 detector**（SyndromeTracker）。最重要的不变量是：

> 一个正确构造的无噪声 circuit，noiseless sampler 跑出来必须产生零 detection events 和零 observable errors。

`test_protocols.py` 里的每个 protocol 都验证这一点。如果有人改了 tracker 逻辑、coupler 几何、或者 builder 的 SE 构造方式，只要破坏了这个不变量，测试就会抓住。

## 为什么不测 LER 精度？

- LER 的具体数值是**实验结果**，不是代码合约。更好的 detector 构造方式理应使 LER 降低，这种改动应该允许 merge。
- 跑足够多 shots 验证 LER 精度需要分钟级时间，不适合 CI。
- LER 的阈值行为检查（d=5 < d=3）和 raw/decomposed 一致性检查更适合作为**发版前的人工验收清单**，见 `skills/gotchas/` 或 `front-end/dev_log.md`。

## 添加新协议时

每当新增一个 protocol，在 `test_protocols.py` 对应类里加一个 noiseless build 测试：

```python
def test_my_new_protocol(self):
    from lightstim.protocols.my_protocol import MyProtocol
    exp = MyProtocol(distance=3, rounds=2, noise_params=None)
    c = build_quiet(exp.build)
    assert_valid_circuit(c)
    assert_noiseless(c)
    assert_dem_valid(c)
```
