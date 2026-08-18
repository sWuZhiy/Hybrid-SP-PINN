# Stage 10：PINN 训练策略消融（from-scratch vs fine-tune）

## 10.1 定位与科学问题

**本阶段不是「谁训练更快」的速度对比，而是一个消融实验**：验证
「warm-start 的 G-钉住作用是不是把 Hybrid 循环在强反型区救回来的**必要条件**」。

背景（Stage 9 已证实的约束，见 [stage9.md](stage9.md) §9.4.3 / `src/sp_solver.py`
`_make_pinn_poisson_step` 文档串）：

- 外层 Gummel+Anderson 固定点迭代的全部收敛理论都要求内层 G(n) 是**静态映射**
  （给定 n 唯一决定 φ_new）。
- 续训轮数不足（scf_epochs=500）时 G 随权重每轮漂移，Anderson 在「由不同映射
  产生的残差」上外推失效，强反型漂移到伪不动点却报收敛。
- 训足轮数（scf_epochs=3000）后 G 近似静态，Anderson 才合法。

本阶段要回答三个问题：

1. **from-scratch 在 SP 循环里是否可行**（每轮随机重训、无跨轮记忆）？
2. **fine-tune 的真正收益在哪**——是「跨轮钉住」（压低 G 漂移）而非「省训练步」？
3. **续训轮数 scf_epochs 能否在保证 G 静态的前提下安全下调**（配合降 lr 实现加速）？

## 10.2 两种策略

### A. from-scratch

每一次 SCF 内层都随机初始化 PINN，训练固定 epochs 轮，**不继承**上一轮权重。

### B. fine-tune

第一轮完整训练；后续 SCF 从上一轮权重 warm_start 续训（Stage 9 已默认）。

## 10.3 关键实现约束（Stage 9 审查发现，必须遵守）

### C1. from-scratch 每轮都要重做「n=0 经典课程」

`poisson_pinn.py` `solve_poisson_pinn` 文档串：强反型冻结 n 直接训练 `max|Δφ|≈1.1 V`
失败，两阶段（先 n=0 建势阱、再满 n）才 2.7 mV。from-scratch 每轮 `_reset_model()`
抹掉势阱记忆 → **每轮都必须重做 n=0 预热**，否则第 2 轮起「random init + 满 n」撞回
1.1 V 发散。故 A 组不能复用 Stage 9 的 `initialized` 布尔结构（它只在首轮做两阶段），
而应**每轮走 `solve_poisson_pinn` 的两阶段路径**。

### C2. `_reset_model()` 只重置权重、不清 Adam 状态

`poisson_pinn.py` `_reset_model` 只调 `module.reset_parameters()`，而 Adam optimizer
在 `__init__` 建一次，其 `exp_avg/exp_avg_sq` 绑在同一批 Parameter 上。权重值重置后
动量仍残留 → A 组混入「权重重置但动量续传」，污染 A/B 对照。实现 A 组时须：

- 每轮**新建** `PoissonPINNSolver`（clean），或
- 显式重建 optimizer（`self.optimizer.state.clear()` + 重新 `Adam(...)`）。

本阶段选「每轮新建 solver」，并显式传 seed（见 C3）。

### C3. seed 语义与随机性

`PoissonPINNSolver.__init__` 里 `if seed is None: seed = int(p.get('seed', 0))`：

- config `seed: 0` → 每轮新建 solver 时 `torch.manual_seed(0)` → 每轮**相同**权重，
  A 组退化成确定性重训，测不出 seed 鲁棒性；
- 写 `seed: null` → `int(None)` 崩溃（必须修）。

本阶段修复：`__init__` 支持 `seed is None` → 不调 `manual_seed`（走全局 RNG），并给
`train(strategy='from_scratch')` 传每轮不同 seed。A 组至少跑 3 个 seed 取散布。

### C4. `_check_physical` 会把 A 组失败变成崩溃

from-scratch 在强反型的伪不动点会触发 `_check_physical` 的 `RuntimeError`。实验脚本
须**捕获**并记为「A 组 Vg=1.5 第 k 轮中止（G 漂移→伪不动点）」，而非 crash（类比
`experiments/08_hybrid_sp_pinn.py` 的收敛断言）。A 组「中止」本身就是结论之一。

## 10.4 评价指标（Stage 9 审查修正）

原设计列「training steps / wall-clock / SCF 轮数 / 最终误差」，但前三者在
`scf_epochs == epochs == 3000` 且无早停下被构造性拉平，最终误差又落在噪声地板
（~0.5 mV）内。改用：

1. **是否收敛 / 中止轮次**（A 组强反型是否 abort）——核心区分量；
2. **SCF 迭代轮数**（G 漂移会拖慢 Anderson）；
3. **G 漂移量**（解耦测量，见 10.5）——fine-tune「跨轮钉住」的直接证据；
4. 最终 φ_s / max|Δφ| / rel-L2 / Ns 偏差（作背景校验，非区分量）。

## 10.5 G 漂移的解耦测量（核心新实验）

为把「G 是否静态」从外层 Anderson 里剥离出来，固定 n = n_FDM(Vg)（Stage 9 的
FDM 收敛电子密度），只测内层映射的确定性：

- **A 组**：固定 n，用 K≥3 个不同 seed 各从 scratch 训 epochs=3000，记录每次
  φ_new 的 φ_s；以 φ_s 跨 seed 标准差 / 最大差作为「from-scratch 的 G 漂移」。
- **B 组**：固定 n，从同一 warm-start 分别续训 scf_epochs ∈ {500, 1000, 2000, 3000}，
  记录 φ_new 相对固定点的偏差，画出「scf_epochs → G 漂移」曲线，定位 G 静态所需
  的最少轮数（问题 3）。

预期：A 组 φ_s 散布远大于 B 组 3000 档；B 组 500 档漂移应复现 Stage 9 的伪不动点前兆。

## 10.6 代码改动清单

- [ ] `src/poisson_pinn.py`：修 seed（`seed is None` 不 seed、不 `int(None)` 崩溃）；
  增加 `_reset_model` 后重建 optimizer 的入口（或说明 A 组用新建 solver）。
- [ ] `src/sp_solver.py`：`_make_pinn_poisson_step` 增加 `training_strategy`
  （`'fine_tune'` 默认 / `'from_scratch'`），from_scratch 每轮走两阶段 + 每轮新建
  solver + 每轮新 seed；fine_tune 保持 Stage 9 现状。
- [ ] `configs/default.yaml`：加 `training_strategy: fine_tune`、
  `from_scratch_seeds: [0, 1, 2]`；`seed` 保持 0（fine-tune 可复现），A 组 seed 由
  实验脚本显式覆盖。
- [ ] `experiments/09_training_strategy.py`：Vg ∈ {0.5, 1.0, 1.5, 2.0} × {A, B}
  ×（A 组 3 seed），捕获 RuntimeError 记为中止；输出 metrics CSV + 面板图。

## 10.7 输出产物规划

- `results/figures/training_strategy_metrics.csv`：Vg、strategy、seed、converged、
  abort_iter、iters、G_drift_mV、phi_s_err_mV、rel_l2_pct；
- `results/figures/training_strategy_*.png/pdf`：A/B 收敛历史对照、G 漂移 vs
  scf_epochs 曲线、A 组 seed 散布箱线图。

## 10.8 衔接

- 上游：[stage9.md](stage9.md) §9.4.3（G 静态要求）、§9.5（信任域）；
  [stage8.md](stage8.md) §8.7C（warm_start/seed 铺路、φ 预拟合初值备选）。
- 下游：Stage 11 严格对比矩阵中「PINN 训练策略」一维以本阶段结论为准；
  Stage 12 参数化 PINN 的增量训练同理。

**关键文献**：[references.md](references.md) §A（Bengio 2009 课程学习；Kingma & Ba 2015 Adam）。
