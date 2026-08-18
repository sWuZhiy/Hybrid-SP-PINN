# Stage 11：严格对比实验（汇总 + 补齐 + 统一口径）

## 11.1 定位：本阶段是「汇总补齐」，不是「新实验」

《搭建说明》§14/§37 把 Stage 11 写成「完整实验矩阵」，但逐条对照现有代码后发现：

- 实验 A（Poisson 单模块 FDM vs PINN）已在 [stage8.md](stage8.md) / `experiments/07_poisson_pinn.py` 完成，
  指标 max|Δφ|/MAE/rel-L2/wall time 已算，soft-BC 消融已产出 `pinn_ablation_hard_vs_soft_bc.png`；
- 实验 B（完整 SP：FDM-SP vs Hybrid-SP-PINN）已在 [stage9.md](stage9.md) / `experiments/08_hybrid_sp_pinn.py` 完成，
  φ/n/Ns/迭代数/wall 已算；
- scratch vs fine-tune 已在 [stage10.md](stage10.md) / `experiments/09_training_strategy.py` 完成。

因此 **Stage 11 的价值不在「再做一轮对比」，而在三件事**：

1. **补齐缺口指标**——§16《关键科研指标》清单里有 4 个指标从没算过（见 §11.3）；
2. **统一口径**——把各阶段散落、混用单位/域/归一化的指标收敛到一套定义（见 §11.4）；
3. **产出论文汇总表**——把 §39 结果字段 × 全部实验整理成一张表，逐条映射 RQ1–RQ4（见 §11.7）。

唯一的**新实验**是「粗/细网格」（§37 矩阵里唯一未做的一行，见 §11.5）。

## 11.2 与 Stage 8/9/10 的关系（引用 vs 重跑）

| §37 矩阵行 | 状态 | 本阶段动作 |
|---|---|---|
| 固定 VG | ✅ Stage 8（单模块）/ 9（完整 SP） | 引用，不重跑 |
| 不同 VG | ✅ Stage 8/9（4 点 {0.5,1.0,1.5,2.0}） | 引用；是否加密见 §11.6 |
| 粗/细网格 | ⬜ 未做 | **新做**（§11.5） |
| soft BC | ✅ Stage 8 | 引用 `pinn_ablation_hard_vs_soft_bc.png` |
| scratch PINN | ✅ Stage 10 | 引用 `training_strategy_sp.csv` |
| fine-tune PINN | ✅ Stage 9（默认）/ 10（对照） | 引用 |

「引用」的含义：Stage 11 的汇总表直接读取 07/08/09 已落盘的 CSV，**不再重跑**对应实验。
只有「粗/细网格」和「failure rate 的多 seed 统计」需要新跑。

## 11.3 缺口指标补齐（§16 清单中从未算过的 4 个）

### P1. 子带能级差 E₁ —— 卡死 RQ3 闭环

**为什么必须算**：§40 的 RQ3 问「PINN 的误差是否会通过自洽反馈放大到 n、Eᵢ、Ns」。
现在只算了 φ 误差和 Ns 误差（且 Ns 误差 08 只打印、不落库），**子带能级差从未算过**。

**第一性原理（为什么 E₁ 比 Ns 更敏感）**：

- 量子限域下基态能级 E₁ 由势阱形状直接决定，一阶近似 E₁ ≈ q·⟨φ⟩_阱（电子概率密度
  加权平均的静电势）。φ 差 δφ ~ 0.5 mV → E₁ 差 δE₁ ~ q·δφ ~ 0.5 meV。而 E₁ 本身
  ~ 百 meV 量级，**相对误差可达 ~0.5%**。
- Ns 是 n(z) 的空间积分：Ns = Σᵢ ∫ |ψᵢ|² f(Eᵢ) dz。逐点的 φ 误差在积分里被平滑，
  且占据函数 f(Eᵢ) 在 Eᵢ ≫ EF 时指数抑制、Eᵢ ≈ EF 时对 δEᵢ 只敏感于 f 的斜率（~1/kT）。
  故 Ns 对 φ 误差**相对钝感**。

结论（这是论文 RQ3 的看点）：**φ 误差小、但 E₁ 相对放大、Ns 相对钝感**——这个不对称
必须用 E₁ 差来展示，否则 RQ3 只答了一半。

**实现**：`SPResult.subband_energies` 已存最终 φ 下两组能谷的本征能。取第一能谷组基态
`E₁ = res.subband_energies[0][0]`（该组 m_z = m_l 更重 → 基态更低），
`E1_err = E₁_pinn − E₁_fdm`（单位 meV）。

### P2. charge neutrality error —— 不依赖 FDM 参考的独立自洽校验

**为什么必须算**：§16 要求；它回答「PINN 的软 Robin 到底收敛到多接近电中性」，
而不是「相对 FDM 看起来对」。FDM 的 Robin 是硬边界（恒成立 ~0），PINN 是软损失
（残差 ~0.004），这个对比本身就是一条独立结论。

**第一性原理（Gauss 定理推导）**：

Poisson 方程 ∇·(ε∇φ) = −ρ，ρ = q(p − n − N_A)。从界面 z = t_ox 积分到体区 z = L
（体区电场 E(L) = 0）：

```
ε_si · E_si(t_ox) = ∫_{t_ox}^{L} ρ dz   （= Q_si，半导体单位面积电荷）
```

栅电荷 Q_g = ε_ox · E_ox = ε_ox(Vg − φ_s)/t_ox。电中性要求 Q_g + Q_si = 0，即

```
ε_si · E_si(t_ox) − ∫_{t_ox}^{L} ρ dz = 0
```

**这正是界面 Robin（电位移连续 ε_si φ'_si = ε_ox(Vg−φ_s)/t_ox）的另一种写法**。因此：

- 对 FDM：Robin 是硬 BC，此量 ~0（仅剩离散 O(dz) 误差）；
- 对 PINN：Robin 是软损失，此量 = 训练后残存的 Robin 残差（~0.004）。

**定义**：`charge_neutrality_err = |ε_si E_si(t_ox) − ∫ρ dz| / max(|Q_g|, |Q_si|, 1)`，
分别对 FDM 解和 PINN 解各算一个，再报告两者的量级差。

### P3. failure rate —— 数值稳定性（定义 + 多 seed 统计）

**为什么必须算**：§16 数值稳定性要求 failure rate；§14 实验 C 也是它。现在从没统计过。

**「失败」必须分三类**（Stage 9/10 已分别撞到，不能混成一个数）：

1. **训练发散**：`_check_physical` 抛「NaN/Inf」（强反型窄尖峰 tanh 表达不了）；
2. **伪不动点**：`_check_physical` 抛「φ_s 越界 / Robin 残差 > 0.1」（平带 φ_s≈0 或
   全转移 φ_s≈Vg）；
3. **停滞**：`stagnated=True`（G 漂移使 δ 卡在 ≫ tol 的平台，不触发 1/2 但也不收敛）。

**定义**：对固定 Vg × N 个随机 seed 跑 from-scratch（或 fine-tune），
`failure_rate(Vg) = (# converged=False 的 seed) / N`，并**按三类分别计数**。
N 取多大、是否只测 from-scratch（fine-tune 已证稳定），见 §11.6 待定项。

### P4. inference time —— 训练/推理分离（§14 明确警告）

**为什么必须算**：§14 警告「必须区分训练时间和推理时间，不能只报 PINN 推理速度」；
§16 效率清单要 `PINN training time` 和 `PINN inference time` 分开。现在只测了含训练的 `wall`。

**实现**：warm-start 训练完成后，对固定 φ 做 K 次 `predict_full`（或单次前向）取平均，
得单次推理时间（预期 μs–ms 级）。与 `wall_time`（训练为主）并列报告，
论文里写成「训练 ~40 s / 推理 ~μs」而非「PINN 快」。

## 11.4 统一 metric 口径（P5）

各阶段指标散落、口径不一，Stage 11 收敛到一个函数 `compute_metrics(res_f, res_p, device)`：

| 指标 | 现状态 | 统一后 |
|---|---|---|
| 误差域 | 07 用「全器件（含解析氧化层）」 | **默认 Si 区**（氧化层两版都线性，误差只是 φ_s 误差缩放，会稀释）；另附全器件值 |
| rel-L2 归一化 | 07/08/09 各写一遍 | 统一 `‖φ_pinn − φ_fdm‖₂ / ‖φ_fdm‖₂`（分母一致） |
| Ns 相对误差 | 08 只打印、09 落库 | 统一 `(Ns_p − Ns_f)/Ns_f` |
| E₁ 差 | 没算 | 统一 `E₁_pinn − E₁_fdm`（meV） |
| charge neutrality | 没算 | 统一 §11.3 P2 定义 |
| 单位 | mV 与 V 混用 | 内部 V，报告层统一 mV（φ）/ meV（E₁）/ 无量纲（rel-L2、Ns 相对差） |

产出 `src/metrics.py`：一个 docstring 写清每个指标的定义、域、归一化、单位，07/08/09/11
统一调用，避免论文表格里各阶段数字对不上。

## 11.5 粗/细网格（§37 唯一未做的一行，P6 的算法难点）

**核心难点：网格被 Schrödinger 和 Poisson 共用**（单一 `device.z`，`n_grid=1000`，
`build_mesh` 一次性生成）。想干净做「Poisson 网格细化」，却会同时改变：

- Schrödinger 本征值（三对角对角化 O(dz²) 误差）→ E₁、ψ 全变；
- Poisson 离散（O(dz²)）→ φ 变。

两个效应缠在一起，无法隔离。**协议必须明确**：

- 方案 A（整链粗化/细化）：`n_grid ∈ {250, 500, 1000, 2000}` 整条链一起变，比较的是
  「全求解器对网格的鲁棒性」，不是「Poisson 单独细化」；
- 方案 B（只换 Poisson 配点、Schrödinger 固定细网格）：需要给 PINN 传独立配点，代码
  现在做不到，要改 `_make_pinn_poisson_step` 的配点来源——**本阶段先不做**，记为后续。

**物理预期（要诚实写）**：

- FDM 误差 O(dz²) → 加密单调下降（02/03 已有收敛曲线佐证）；
- PINN 精度受**表达力地板（~1e-4 V）**限制，不是网格限制 → 加密到一定程度后**停在
  表达力地板不再改善**。这是 PINN 的一个已知局限，结论必须写成「PINN 的网格无关性
  **受表达力上限约束**」，不能写成「PINN 天然网格无关」。
- 附带耦合：PINN 训练配点 = 网格点，粗网格 = 少训练样本，也影响训练质量。

## 11.6 多栅压扫描（P7，待定项）

§14 要 `VG = 0, 0.1, …, VG_max`；现有扫描只有 4 点 {0.5,1.0,1.5,2.0}。两个待定：

1. **是否加 Vg=0**：平带退化点（φ≡0）。要检查 `_check_physical` 的容差（此时 lo=hi=0，
   1e-3 V 容差）会不会误判、PINN 训练平凡解是否稳定。物理上 Vg=0 无信息量，倾向**不加**
   或只作 smoke 检查不纳入汇总表。
2. **密度 vs 成本**：每点几十秒（弱反型）到几十分钟（强反型 from-scratch）。加密到
   ΔVg=0.1 需 ~20 点 × 平均 ~2 min ≈ 40 min，可接受但要和 08/09 的 4 点区分清楚
   （密集扫描**引用** 08/09 的 4 点，只补中间点）。

## 11.7 论文结果汇总表 + RQ1–RQ4 映射

Stage 11 的最终产物是一张表，把 §39 字段 × 全部实验整理齐全，并逐条映射 RQ：

| §39 字段 | 来源实验 | 状态 |
|---|---|---|
| phi_MAE / phi_max_error / phi_L2 | 07（单模块）、08（SP） | ✅ 引用 |
| Ns_relative_error | 08/09 | 补口径统一 |
| E1_error | **新算** | P1 |
| EF_error | — | 平凡为 0，**从表中删除**（见下） |
| charge_neutrality_error | **新算** | P2 |
| scf_iterations / convergence curve | 08 | ✅ 引用 |
| failure_rate | **新算** | P3 |
| training_time / wall_clock_time | 08/09 | ✅ 引用 |
| inference_time | **新算** | P4 |

> **EF_error 为何删除**：EF 是 bulk 电中性求根（Stage 6）算出的**输入**，FDM 与 PINN
> 共用同一个 EF，二者之差恒为 0，作为「对比指标」无意义。§16 的 `Ei difference` 应解读为
> **子带能级 E₁ 差**（有意义），而非本征费米能级 Eᵢ（恒等）。在论文中明确这个术语差异。

RQ 映射：

- RQ1（PINN 能否在给定电荷密度下精确解 Poisson）→ 07 单模块指标（引用）；
- RQ2（PINN 入 SP 循环后能否稳定收敛）→ 08 收敛历史 + P3 failure rate；
- RQ3（误差是否放大到 n/E₁/Ns）→ **P1 E₁ 差** + Ns 相对差（新补 E₁，补全）；
- RQ4（多偏压重复模拟下训练成本 vs 推理效率）→ **P4 inference time** + 多栅压总时间。

## 11.8 代码改动清单

1. `src/metrics.py`（新）：`compute_metrics(res_f, res_p, device)` 统一算 §11.4 全部指标
   （含 E₁ 差、charge neutrality、inference time），docstring 写清定义/域/归一化/单位；
2. `experiments/10_rigorous_comparison.py`（新）：粗/细网格实验（§11.5 方案 A）+ failure
   rate 多 seed 统计（§11.3 P3）+ 汇总表生成（§11.7）；
3. `src/sp_solver.py`：若方案 B（独立 Poisson 配点）做，需改配点来源——**本阶段不做**；
4. 不重跑 07/08/09，只读其 CSV。

## 11.9 产物规划

- `results/figures/grid_convergence_*.csv/png/pdf`：粗/细网格误差曲线（FDM vs PINN）；
- `results/figures/failure_rate_*.csv`：三类失败计数；
- `results/figures/summary_table.csv`：§11.7 汇总表（论文直接引用）；
- `results/figures/training_vs_inference_time.*`：效率对比图。

## 11.10 衔接

- 上游：读 [stage8.md](stage8.md) §8.6（单模块对照）、[stage9.md](stage9.md)（SP 对照）、
  [stage10.md](stage10.md)（训练策略）的落盘 CSV；
- 下游：给 [stage12.md](stage12.md)（参数化 PINN）提供「多栅压总成本」基线（RQ4 的
  训练成本，是参数化 PINN 要对比的对手）；
- 论文：§11.7 汇总表 + RQ1–RQ4 映射直接支撑论文第 5 章（结果）与第 6 章（讨论）。
