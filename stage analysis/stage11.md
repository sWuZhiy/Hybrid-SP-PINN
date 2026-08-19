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

**第一性原理（φ 误差如何传播到 E₁ 与 Ns）**：

- E₁：量子限域下基态能级由势阱形状决定，一阶近似 E₁ ≈ q·⟨φ⟩_阱（电子概率密度加权
  平均的静电势），故 δE₁ ≈ q·δφ（**线性**传播）。但 E₁ 绝对值大（~300–430 meV），
  **相对误差反而小**（实测 0.02%–0.15%，见 §11.7）。
- Ns：Ns = Σᵢ ∫ |ψᵢ|² f(Eᵢ) dz。在**亚阈值区**（Vg 接近反型开启）Ns ∝ e^{qφ_s/kT}
  **指数敏感**，δφ_s ~ 0.5 mV → δNs/Ns ~ q·δφ_s/kT ~ 1.8%（放大 ~3.9%/mV）；强反型区
  表面势钉扎，指数敏感被抑制，δNs/Ns 降到 0.2%–0.4%。

结论（论文 RQ3 的看点，**已用实测修正**）：φ 误差对 Ns 的传播在亚阈值区被**指数放大**
（Vg=1.0 时 Ns 相对误差 1.5%，远大于 E₁ 相对误差 0.15%），对 E₁ 只是**线性**传播且被
大绝对值稀释。因此「最敏感的可观测量」是**亚阈值区的 Ns**，不是 E₁——这与最初
「E₁ 相对放大、Ns 相对钝感」的猜想**相反**，实测见 §11.7。

**实现**：`SPResult.subband_energies` 已存最终 φ 下两组能谷的本征能。取第一能谷组基态
`E₁ = res.subband_energies[0][0]`（该组 m_z = m_l 更重 → 基态更低），
`E1_err = E₁_pinn − E₁_fdm`（单位 meV）。

### P2. Robin 残差（界面电位移连续）—— 不依赖 FDM 参考的独立自洽校验

**为什么必须算**：§16 要求；它回答「PINN 的软 Robin 到底收敛到多接近电位移连续」，
而不是「相对 FDM 看起来对」。FDM 的 Robin 是硬边界（通量形式离散自动满足），PINN 是
软损失（训练后残存残差），这个对比本身就是一条独立结论。

**第一性原理（界面 Gauss 定理 / 电位移法向分量连续）**：

在 SiO₂/Si 界面 z=t_ox 两侧无界面面电荷时，电位移法向分量连续：

```
ε_ox·E_ox = ε_si·E_si(t_ox)
```

氧化层内 φ 线性（无空间电荷），E_ox = (Vg − φ_s)/t_ox；Si 侧 E_si = −φ'_si。故 Robin
条件（电位移连续）等价于

```
R_iface = ε_si·φ'_si(t_ox) + ε_ox(Vg − φ_s)/t_ox = 0
```

**定义**：`robin_residual = |R_iface| / D_ref`，`D_ref = ε_ox·max(|Vg|, 0.1)/t_ox`
（与 `sp_solver._check_physical` 的中止判据 `|R_iface|/D_ref > 0.1` 完全一致），分别对
FDM 解和 PINN 解各算一个再对比。

**为什么不是「全局电中性 Q_g + Q_si = 0」（原稿笔误，已修正）**：

把 Poisson 方程 d(εE)/dz = ρ 从 t_ox 积分到 L 得

```
ε(L)E(L) − ε_si·E_si(t_ox) = ∫_{t_ox}^L ρ dz = Q_si
```

再代入 R_iface 的定义消去 ε_si·E_si，得 `Q_g + Q_si = R_iface + ε(L)E(L)`。
「电中性 Q_g+Q_si=0」隐含体区电场 E(L)=0；但本器件（t_ox=2nm, L_si=100nm,
NA=1e17）在强反型（Vg≥1.5）时耗尽区 ~102nm ≈ L_si，Si 被完全耗尽，E(L)≠0
（实测 ε_si·E(L)≈3.1e-4 C/m² ≈ Q_g 的 4.7%），故 Q_g+Q_si 测到的是背面电场而非
Robin 残差，且 FDM/PINN 一样大、区分不出软/硬 Robin。局部 Robin 残差 R_iface
不依赖 E(L)，才是干净的口径。原稿把「ε_si E_si − ∫ρ dz」与「Q_g+Q_si」混作 Robin，
均不成立，已统一修正为上面的 R_iface 定义。

**预期（诚实写，实测见 §11.7）**：FDM 用前向差分求 φ' 仍有 O(dz) 误差（反型层界面
φ'' 大），实测 ~1e-3（D_ref 归一化）；PINN 软损失训练后残存 ~2e-3，与 FDM 的前向差分
离散误差同量级（~2 倍，**不是**「2 个数量级」）。这本身是论文结论：软 Robin 收敛精度
受界面离散/表达力共同限制，与前向差分离散误差同量级。

### P3. failure rate —— 数值稳定性（定义 + 多 seed 统计）

**为什么必须算**：§16 数值稳定性要求 failure rate；§14 实验 C 也是它。现在从没统计过。

**「失败」必须分三类**（Stage 9/10 已分别撞到，不能混成一个数）：

1. **训练发散**：`_check_physical` 抛「NaN/Inf」（强反型窄尖峰 tanh 表达不了）；
2. **伪不动点**：`_check_physical` 抛「φ_s 越界 / Robin 残差 > 0.1」（平带 φ_s≈0 或
   全转移 φ_s≈Vg）；
3. **停滞**：`stagnated=True`（G 漂移使 δ 卡在 ≫ tol 的平台，不触发 1/2 但也不收敛）。

**定义**：对固定 Vg × N 个随机 seed 跑 from-scratch（或 fine-tune），
`failure_rate(Vg) = (# converged=False 的 seed) / N`，并**按三类分别计数**。

**已定参数**：N = 8（seed = base + i，i=0..7），**只测 from_scratch**（fine-tune 已证稳定，
见 [stage10.md](stage10.md)）；可复用 Stage 10 已跑的 3 个 seed（0/1/2）再补 5 个。
只对强反型 Vg ∈ {1.5, 2.0} 统计——弱/中反型（Vg≤1.0）from_scratch 已证收敛，failure rate 恒为 0，
不纳入统计。

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
| Robin 残差 | 没算 | 统一 §11.3 P2 定义 |
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

## 11.6 多栅压扫描（P7，已定）

§14 要 `VG = 0, 0.1, …, VG_max`；现有扫描只有 4 点 {0.5,1.0,1.5,2.0}。已定：

1. **Vg=0 不纳入汇总表**，只作一次 smoke 检查（验证 `_check_physical` 在 lo=hi=0 时
   的 1e-3 V 容差不误判、PINN 训练平凡解 φ≡0 稳定）。物理上 Vg=0 是平带退化点、无信息量。
2. **密度**：**引用** 08/09 已有的 4 点 {0.5,1.0,1.5,2.0}，只补中间点（若论文需要
   更密的 φ_s(Vg)/Ns(Vg) 曲线再加密到 ΔVg=0.1），不重跑整条步进扫描。每点几十秒
   （弱反型）到几十分钟（强反型 from-scratch），补点成本可控。

## 11.7 论文结果汇总表 + RQ1–RQ4 映射

Stage 11 的最终产物是一张表，把 §39 字段 × 全部实验整理齐全，并逐条映射 RQ：

| §39 字段 | 来源实验 | 状态 |
|---|---|---|
| phi_MAE / phi_max_error / phi_L2 | 07（单模块）、08（SP） | ✅ 引用 |
| Ns_relative_error | 08/09 | 补口径统一 |
| E1_error | **新算** | P1 |
| EF_error | — | 平凡为 0，**从表中删除**（见下） |
| robin_residual | **新算** | P2 |
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
   （含 E₁ 差、Robin 残差、inference time）、`subband_ground_state`、`robin_residual`
   （P2 口径修正，docstring 写清定义/域/归一化/单位）；
2. `experiments/10_rigorous_comparison.py`（新）：粗/细网格实验（§11.5 方案 A）+ failure
   rate 多 seed 统计（§11.3 P3）+ 汇总表生成（§11.7）+ 训练/推理计时（§11.3 P4）；
3. `tests/test_metrics.py`（新）：4 项单测覆盖 subband_ground_state / robin_residual
   （物理正确性 + 口径自洽）/ compute_metrics 自洽；
4. `src/sp_solver.py`：若方案 B（独立 Poisson 配点）做，需改配点来源——**本阶段不做**；
5. 不重跑 07/08/09，只读其 CSV。

## 11.9 产物规划

- `results/figures/summary_table.csv/png/pdf`：§11.7 汇总表（论文直接引用）；
- `results/figures/grid_convergence.csv/png/pdf`：粗/细网格误差曲线（FDM vs PINN，方案 A）；
- `results/figures/failure_rate.csv`（三类失败计数汇总）+ `failure_rate_detail.csv`（逐 seed 明细）；
- `results/figures/training_vs_inference_time.csv/png/pdf`：训练 vs 推理效率对比。

## 11.10 衔接

- 上游：读 [stage8.md](stage8.md) §8.6（单模块对照）、[stage9.md](stage9.md)（SP 对照）、
  [stage10.md](stage10.md)（训练策略）的落盘 CSV；
- 下游：给 [stage12.md](stage12.md)（有监督参数化神经代理）提供「多栅压总成本」基线
  （RQ4 的训练成本，是 surrogate 要对比的对手）；
- 论文：§11.7 汇总表 + RQ1–RQ4 映射直接支撑论文第 5 章（结果）与第 6 章（讨论）。

## 11.11 实测结果（summary + inference；grid 见 §11.12、failure 见 §11.13）

统一口径汇总表（`results/figures/summary_table.csv`，`--part summary` 生成）：

| Vg [V] | φ_s 差 [mV] | max\|Δφ\|(Si) [mV] | rel-L2(Si) [%] | Ns 差 [%] | E₁ 差 [meV] | Robin FDM | Robin PINN |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.000 | 0.000 | 0.0000 | 0.000 | 0.000 | 7.29e-4 | 7.29e-4 |
| 1.0 | −0.465 | 0.465 | 0.0363 | −1.485 | +0.413 | 5.20e-4 | 1.12e-3 |
| 1.5 | −0.459 | 0.459 | 0.0268 | −0.422 | +0.191 | 1.18e-3 | 1.89e-3 |
| 2.0 | −0.260 | 0.260 | 0.0312 | −0.207 | +0.095 | 1.68e-3 | 2.23e-3 |

三条独立结论（RQ1/RQ3/P2）：

1. **φ 误差**：Vg≤1.5 时 max|Δφ|(Si)<0.5 mV、rel-L2<0.04%，深反型 Vg=2.0 仍 <0.3 mV
   （Si 区）——Hybrid 复现 FDM 自洽解（RQ1 完整闭环）。
2. **P1（E₁ vs Ns 的传播不对称，已修正原猜想）**：Ns 相对误差在亚阈值区 Vg=1.0 达
   **−1.485%**（指数敏感 q·δφ_s/kT≈1.8%），强反型降到 −0.2%；而 E₁ 相对误差全程
   仅 0.02%–0.15%（线性传播 + 大绝对值稀释）。故**最敏感可观测量是亚阈值区的 Ns**，
   不是 E₁（原 §11.3 P1 的「E₁ 放大、Ns 钝感」猜想被实测**反转**）。
3. **P2（Robin 残差，硬 vs 软）**：FDM 前向差分 Robin 残差 ~5e-4–1.7e-3（反型层界面
   O(dz) 离散误差），PINN 软损失残存 ~7e-4–2.2e-3，二者**同量级、相差 ~1.3–2 倍**，
   不是原稿「2 个数量级」。结论：软 Robin 收敛精度受界面离散/表达力共同限制。

推理时间（`training_vs_inference_time.csv`，`--part inference`）：单次 Poisson 两阶段
训练 **39.16 s**，单次 `predict_full` 推理 **1.16 ms**，加速 **~3.39×10⁴ 倍**（RQ4）。

## 11.12 实测结果（grid；failure 见 §11.13）

粗/细网格收敛（方案 A 整链，Vg=1.5，参考 = FDM@2000；`grid_convergence.csv`）：

| n_grid | dz [nm] | φ_s 差 FDM [mV] | φ_s 差 PINN [mV] | Ns 差 FDM [%] | Ns 差 PINN [%] |
|---|---|---|---|---|---|
| 250 | 0.410 | 4.939 | 0.364 | 6.114 | 0.593 |
| 500 | 0.204 | 2.208 | 0.128 | 2.498 | 0.395 |
| 1000 | 0.102 | 0.751 | 0.497 | 0.813 | 0.424 |
| 2000 | 0.051 | 0（参考） | 0.471 | 0（参考） | 0.530 |

结论（RQ 网格无关性，§11.5 预期兑现）：

1. **FDM 单调收敛**：φ_s 差 4.94 → 2.21 → 0.75 → 0 mV、Ns 差 6.11% → 2.50% → 0.81% → 0，
   随网格加密单调下降（每加密一倍误差降 ~2–3 倍，介于 O(dz)–O(dz²) 之间，与 Stage 3/4
   在光滑问题上验证的二阶量级一致，反型层界面离散使 φ_s 的收敛略慢于理想二阶）。
2. **PINN 停在表达力地板**：φ_s 差在 0.13–0.50 mV、Ns 差在 0.40–0.59% 之间非单调波动，
   不随网格加密收敛到 0。PINN 的 φ_s 收敛到 ~1079.36 mV 的自有固定点，比 FDM@2000 的
   1078.886 mV 高 ~0.47 mV——这是**不可约的表达力偏置**（光滑网络 + 软 Robin 的逼近下限），
   不是网格误差。
3. 因此论文结论须写成「PINN 的网格无关性**受表达力上限约束**」（实测地板 ~0.5 mV，
   比 §11.5 预估值 ~0.1 mV 略高），不能写成「PINN 天然网格无关」。

## 11.13 实测结果（failure rate）

from_scratch 在强反型的失败率（Vg ∈ {1.5, 2.0}，各 N=8 seed，复用 Stage 10 的 seed 0/1/2 +
本阶段补跑 seed 3–7；`failure_rate.csv` / `failure_rate_detail.csv`）：

| Vg | N_seeds | converged | divergence | pseudo_fixed_point | stagnation | failure_rate |
|---|---|---|---|---|---|---|
| 1.5 | 8 | 0 | 0 | 8 | 0 | 1.0 |
| 2.0 | 8 | 0 | 0 | 8 | 0 | 1.0 |

结论：

1. **失败率 100%**：16/16 个 seed 全部失败，0 收敛。失败模式**统一为伪不动点**
   （`pseudo_fixed_point`），无 `divergence`（NaN/Inf）、无 `stagnation`——即 from_scratch
   在强反型下**从不真正发散**，而是稳定地停在错误的不动点。
2. **中止判据正是 Robin 残差**：`_check_physical` 检测到界面电位移不连续
   `|R_iface|/D_ref = 0.102–0.422 > 0.1`（应为 0）才中止。这反过来验证 §11.3 的 P2 口径修正：
   Robin 残差是区分「真不动点（FDM 硬 Robin ~1e-3、PINN 软损失 ~2e-3）vs 伪不动点
   （~0.1–0.4）」的天然判据，二者量级差约两个数量级。
3. **伪不动点 seed 相关**：φ_s 卡在 375–1770 mV 的散乱值（真值 Vg=1.5 时 ~1080 mV、
   Vg=2.0 时 ~1182 mV），随随机初始化落到不同的伪不动点——说明 from_scratch 的损失地形在
   强反型下存在大量局部极小，且都破坏界面电位移连续性。
4. **与 Stage 10 结论一致并量化**：Stage 10 定性发现「from_scratch 强反型不可靠」，本阶段
   把「不可靠」量化为**确定性 100% 失败**，且把失败机制锁定到「伪不动点 + Robin 不连续」，
   从而严格证明论文结论「混合 PINN 强反型必须 warm-start（fine_tune）」的必要性。

## 11.14 阶段小结（Stage 11 汇总补齐）

本阶段把 Stage 7–10 遗留的「缺口指标 + 统一口径 + 粗细网格 + 训练失败率 + 推理成本」
一次补齐，落地为 4 个可复现实验与 5 条论文级结论：

- **补齐指标**：新增 `subband_ground_state`（由 φ 重解 Schrödinger 得 E₁，不重跑 SP）
  与 `robin_residual`（P2 口径修正，见 §11.3），两条均由 `tests/test_metrics.py` 校验。
- **统一口径**：Si 区 max/MAE/rel-L2 由 φ 剖面直接算（与 Stage 9 只报全局一致），
  `compute_metrics` 自洽性由测试 4 保证（同一解相减恒 0）。

五条结论（映射 §11.7 的 RQ1–RQ4）：

1. **RQ1（复现）**：Hybrid 在 Vg≤2.0 全量程复现 FDM 自洽解——max|Δφ|(Si)<0.5 mV、
   rel-L2<0.04%（§11.11）。
2. **P1（敏感量，反转原猜想）**：最敏感可观测量是**亚阈值 Ns**（Vg=1.0 时 −1.485%，
   指数敏感），不是 E₁（线性 + 大绝对值稀释，0.02%–0.15%）。
3. **P2（Robin 硬/软，修正原稿「2 个数量级」）**：FDM 硬 Robin ~5e-4–1.7e-3 与 PINN
   软损失 ~7e-4–2.2e-3 **同量级、仅差 ~2 倍**；且 Robin 残差是**区分真不动点（~1e-3）
   vs 伪不动点（~0.1–0.4）的判据**，直接支撑 §11.13 的失败机制。
4. **网格无关性（RQ 补充）**：FDM 单调收敛（~O(dz)–O(dz²)），PINN 停在 **~0.5 mV 表达力
   地板**——论文须写「PINN 网格无关性受表达力上限约束」，不能写「天然网格无关」。
5. **失败率（Stage 10 的量化）**：from_scratch 在强反型 **100% 失败**（16/16 伪不动点），
   机制为「界面电位移不连续」，故**强反型必须 warm-start（fine_tune）**。

产物：`summary_table.csv/png/pdf`、`grid_convergence.csv/png`、`failure_rate.csv` +
`failure_rate_detail.csv`、`training_vs_inference_time.csv/png`；代码新增 `src/metrics.py`、
`experiments/10_rigorous_comparison.py`、`tests/test_metrics.py`（详见 §11.8/§11.9）。
下游：给 Stage 12（有监督代理）提供「多栅压总成本 + 强反型必须 warm-start」两条基线。
