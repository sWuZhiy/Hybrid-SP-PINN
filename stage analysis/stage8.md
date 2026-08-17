# Stage 8：Poisson-PINN 独立求解器（重点）

- **内容**：给定固定 ρ(z)，训练 PINN 解一维 Poisson，逼近 FDM。Schrödinger/EF/SCF 一律不介入。
- **状态**：✅ 已完成（代码、测试、实验、文档全部落地；git 提交 `f9d8890` + 补齐 `b0fbca6`）。

---

## 8.0 操作清单与本阶段产出

### 8.0.1 本阶段做了什么（实际操作流程）

1. **设计与推导**（§8.1–§8.4）：从静电学第一性原理出发，推导界面 Robin 条件、氧化层解析解、标幺化方案；诊断出「光滑网络无法表示介电导数跳变」的核心难题并给出解法。
2. **编写 [src/poisson_pinn.py](../src/poisson_pinn.py)**（新建）：MLP + 硬约束 + 损失 + 训练循环 + drop-in 入口。
3. **更新 [configs/default.yaml](../configs/default.yaml)**：新增 `pinn` 训练超参数段。
4. **编写 [tests/test_poisson_pinn.py](../tests/test_poisson_pinn.py)**（新建）：7 项测试。
5. **编写 [experiments/07_poisson_pinn.py](../experiments/07_poisson_pinn.py)**（新建）：对照图 + 消融图 + 指标 CSV。
6. **调试中发现并修复 3 个问题**：
   - Robin 界面条件**符号错误**（写成减号）——由 D 连续推导核对发现并改正（§8.2.1）；
   - `predict_full` 对 (1,1) 张量 `float(numpy())` 报 TypeError——改 `.detach().item()`；
   - **强反型冻结 n 直接训练失败**（max|Δφ|=1137 mV）——诊断为鸡-蛋方向冲突，用两阶段课程学习修复（§8.6.2）。
7. **补测**：默认 config 两阶段课程（1500+1500）冻结 n 精度、soft-BC 消融。
8. **全量测试**：`python -m pytest tests/ -q` → **65 项全部通过**（4:06）。
9. **定稿与推送**：实验脚本、图像说明（图 20–23）、README/索引全局补齐，提交推送 GitHub。

### 8.0.2 涉及文件与逐文件职责

| 文件 | 状态 | 职责 |
|---|---|---|
| [src/poisson_pinn.py](../src/poisson_pinn.py) | 新建 | `PoissonPINN`（MLP：1→64×4→1，tanh，线性输出）；`PoissonPINNSolver`（有状态：配置读取、`_to_u` 标幺、`_phi_bar` 硬约束、`_loss`、`train`（warm_start/epochs/n_ramp_frac）、`predict_full` 氧化层重建）；`solve_poisson_pinn`（drop-in 入口，n 非零时默认两阶段课程） |
| [configs/default.yaml](../configs/default.yaml) | 修改 | `pinn` 段：`epochs=3000`、`lam_iface=1.0`、`seed=0`、`hard_constraint=true`、`n_ramp_frac=0.5` |
| [tests/test_poisson_pinn.py](../tests/test_poisson_pinn.py) | 新建 | 7 项测试（见 §8.6.1 下方清单），可直接 `python tests/test_poisson_pinn.py` 运行 |
| [experiments/07_poisson_pinn.py](../experiments/07_poisson_pinn.py) | 新建 | 4 组对照/消融实验 + 图 + `pinn_metrics.csv` |
| [图像物理内涵说明.md](../图像物理内涵说明.md) | 修改 | 新增 Stage 8 图 20–23 的物理内涵与判读要点 |
| [README.md](../README.md) | 修改 | 勾选 Stage 8、测试清单补 `test_poisson_pinn.py` |
| stage analysis/README.md、stage8.md | 修改 | 索引状态 ✅、本文档 |
| scratch_verify_pinn.py / scratch_diagnose_frozen.py / scratch_remeasure.py | 已删除 | 调试期临时脚本（诊断 Robin 符号、强反型失败机理、补测），结论已并入 §8.6，不入库 |

### 8.0.3 输出产物与关键数值

**图表与数据**（[results/figures/](../results/figures/)）：

| 产物 | 内容 |
|---|---|
| `pinn_classical_Vg0.5_vs_fdm.{png,pdf,csv}` | 经典 n=0、Vg=0.5 对照（4 面板图 + 剖面 CSV） |
| `pinn_classical_Vg1.0_vs_fdm.{png,pdf,csv}` | 经典 n=0、Vg=1.0 对照 |
| `pinn_frozen_n_Vg1.5_vs_fdm.{png,pdf,csv}` | 冻结量子 n、Vg=1.5 强反型对照（两阶段课程） |
| `pinn_ablation_hard_vs_soft_bc.{png,pdf}` | hard vs soft BC 消融（逐点误差 + rel-L2 柱状图） |
| `pinn_metrics.csv` | 全部 case 的 max\|Δφ\| / MAE / rel-L2 / φ_s / wall time 汇总 |

**关键数值**（实测，seed=0）：经典 0.685/0.658 mV；冻结 n 3.21 mV（默认 3000 轮两阶段）；soft-BC 53.94 mV；rel-L2 0.10%–0.26%；单次训练 27–45 s。

---

## 8.1 物理模型的第一性原理

### 8.1.1 Poisson 方程从哪来（静电学基本定律链）

PINN 要求解的是 Poisson 方程，它本身是静电学两条基本定律 + 一条本构关系的推论：

1. **高斯定律（Gauss's law，Maxwell 方程组之一）**：`∇·D = ρ_f`。物理内容：静电场穿过闭合面的电位移通量等于面内自由电荷——电场线「从正电荷发出、终止于负电荷」的定量化。微分形式由散度定理从积分形式导出。
2. **电场无旋**：静电场（无时变磁场）满足 `∇×E = 0`，故 E 可写成标量势的负梯度 `E = −∇φ`。负号约定：正电荷沿场线方向运动时电势降低。
3. **线性介质本构关系**：`D = εE`。物理内容：介质被外电场极化，极化电荷削弱外场，净效果是 ε = ε₀ε_r 替换 ε₀（ε_r：相对介电常数，Si 11.7 / SiO₂ 3.9——分别是共价键极化和 Si–O 键离子极化的响应）。

三者合并：`−∇·(ε∇φ) = ρ_f`，一维化即

```
−d/dz[ ε(z)·dφ/dz ] = ρ(z)
```

**为什么 ε 要留在导数里面**（本项目最关键的物理点）：Si/SiO₂ 界面处 ε 从 3.9ε₀ 跳到 11.7ε₀（不连续），`dε/dz` 是 Dirac δ。把 ε 提出导数号外（拉普拉斯形式 `ε·d²φ/dz²=−ρ`）等于假设 ε 均匀——丢掉了 3 倍的介电跳变，物理上错。这是论文式 (3.2) 的错误根源（§8.7B-1）。

### 8.1.2 电荷密度 ρ 的构成（掺杂完全电离 + 载流子统计）

p-Si 内的自由电荷密度 `ρ = q(p − n − N_A)` 三项来自：

- **受主电离 `−N_A`**：B 受主在 Si 中的电离能 ≈ 45 meV，与室温热能量 kT ≈ 25.85 meV 同量级偏小，室温下电离率约 85–90%；作为第一版模型取**完全电离近似** `N_A⁻ = N_A`（耗尽电荷的来源）。第一性原理依据：费米-狄拉克占据下受主能级被空穴占据的概率，在 EF 远高于受主能级时趋近 1。
- **电子 n 与空穴 p**：见 §8.1.3。氧化层内 `ρ_ox = 0`（理想绝缘体无自由电荷）——这正是 §8.1.5 解析氧化层的依据。

### 8.1.3 平衡载流子为什么是指数形式（Fermi-Dirac → Boltzmann）

`p = n_i·exp(−(EF+qφ)/kT)` 是**热平衡 + 非简并近似**的直接推论：

1. **费米-狄拉克分布（第一性原理）**：全同费米子体系在热平衡下（巨正则系综），能量为 E 的单粒子态被占据的概率是 `f(E) = 1/[exp((E−EF)/kT)+1]`——由「总粒子数固定 + 总能量固定下熵最大」导出，EF（化学势）是拉格朗日乘子。
2. **非简并近似**：掺杂浓度 NA=1e17 cm⁻³ ≪ 态密度 ~1e19 cm⁻³，EF 位于禁带内、离带边 ≫ kT，故 `f ≈ exp(−(E−EF)/kT)`（Boltzmann 尾部）。空穴浓度 = 价带态密度 × 占据概率积分 → `p = N_v·exp(−(EF−E_v)/kT)`；电子 `n = N_c·exp(−(E_c−EF)/kT)`。
3. **本征能级参考改写**：引入本征能级 E_i（无掺杂时的费米能级，`E_i=(E_c+E_v)/2+(kT/2)ln(N_v/N_c)`，Si 中约在禁带中央）后，两式合并为对称形式 `n = n_i·exp((EF−E_i)/kT)`、`p = n_i·exp(−(EF−E_i)/kT)`。两式相乘得**质量作用定律** `np = n_i²`——与 EF 无关，热平衡的必然结果。
4. **能带弯曲与规范 A**：加栅压后能带随静电势弯曲 `E_i(z) = E_i(bulk) − qφ(z)`；取能量规范 A（bulk 本征能级为零点，EF 相对 E_i 计量，EF ≈ −406.2 meV，p 型为负），代入得 `p(z) = n_i·exp(−(EF+qφ)/kT)`。φ=0 处回到 `p = n_i·e^{−EF/kT} = N_A`（bulk 电中性）——这是检验公式正确性的锚点（论文式 (2.2) 漏掉 EF 项即违反此锚点，§8.7B-3）。

### 8.1.4 介电界面：D 连续与导数跳变（3 倍）

界面无自由面电荷时，把高斯定律的积分形式用于包住界面两侧的「药盒」面（厚度→0），得到

```
ε_si·φ'_si(t_ox) = ε_ox·φ'_ox(t_ox)        （法向电位移 D 连续）
⟹  φ'_si / φ'_ox = ε_ox/ε_si = 1/3
```

物理图像：E 线穿过界面时「折射」，折射率 = 介电常数比。φ 连续（势连续）+ φ' 跳 3 倍（场跳变）——这一「导数不连续」是 PINN 与 FDM 分道扬镳的地方（§8.3）。实测（经典耗尽 Vg=1.0 V，FDM 参考解 φ_s=0.8998 V，见 §8.6）：E_ox=(Vg−φ_s)/t_ox=5.0e7 V/m，D 连续 ⟹ E_si=E_ox·(ε_ox/ε_si)=1.7e7 V/m，比值 E_ox/E_si=ε_si/ε_ox=3.000，与理论一致 ✓（测试 test_interface_displacement_continuity 直接验 |R_iface|/D_ref<5%）。

### 8.1.5 氧化层 φ 严格线性（ρ_ox=0 的解析解）

理想氧化层无自由电荷：`ρ_ox=0` ⟹ Poisson 退化为 `d²φ/dz²=0` ⟹ φ 线性。两端值 φ(0)=Vg（金属栅等势）、φ(t_ox)=φ_s，线性插值：

```
φ_ox(z) = Vg − (Vg − φ_s)·z/t_ox
```

**推论（本方案的全部依据）**：氧化层不需要数值求解——给定 φ_s，整段 φ_ox 闭式表达、零误差、无 kink。φ_s 未知，由界面条件与 Si 区解耦合确定。这就是「解析氧化层 + Si 单网络」方案的第一性原理来源。

### 8.1.6 PINN 为什么能解 PDE（残差最小化原理）

PINN（Physics-Informed Neural Network，Raissi et al. 2019）的核心是把「求方程的解」转化为「最小化残差」：

1. 把解参数化为神经网络 φ_θ(z)（θ 为权重）；tanh MLP 是 C^∞ 函数，导数可用**自动微分**（反向传播的链式法则）精确求取，无需网格差分；
2. 定义 PDE 残差 `R(z) = −ε_si·d²φ_θ/dz² − q(p(φ_θ)−n−N_A)`（把 φ_θ 代入方程后的「不平衡量」）；
3. **如果 φ_θ 是真解，则 R≡0 处处成立**，损失 `L = mean(R²)` 在真解处取全局最小 0。故 `L(θ)` 的极小化在原理上等价于求解方程——无监督（不需要标签数据，解的唯一来源是物理定律本身）。

两点必须清醒（已与用户确认）：单点 PINN 每次只解一个 (ρ,Vg)，换题必重训，不鲁棒、不普适（泛化只在 Stage 12 参数化后才有）；FDM 解只作验证基准，绝不作训练标签。

### 8.1.7 硬约束为什么有效（架构保证 vs 损失竞争）

- **软约束**：把 BC 残差加进损失 `L += φ̄(1)²`，靠梯度下降逼近。BC 项与 PDE 项在损失中**竞争**梯度，网络在「满足边界」与「满足方程」间折中分配误差。
- **硬约束**：`φ̄(u) = (1−u)·NN(u)` 把 φ(L)=0 **编码进网络架构**——u=1 处乘积恒为 0，与权重无关，边界误差恒为 0，网络的全部容量用于满足 PDE 与界面条件。
- 实测（经典 Vg=1.0）：硬约束 0.658 mV vs 软约束 53.94 mV，**精度差 ~82 倍**；且软约束的 φ(L) 本身满足得很好（0.13 mV）——误差来自 PDE 残差在边界附近的分配恶化，而非 BC 未满足。这是论文「硬约束必要性」的直接实验证据。

---

## 8.2 关键公式推导链（逐步）

### 8.2.1 Robin 界面条件推导（含加号核对——曾犯过符号错误）

**目标**：把 §8.1.4 的 D 连续条件改写成只含 Si 区解 φ_si 的边界条件。

1. 氧化层解线性（§8.1.5）⟹ `φ'_ox = (φ_s − Vg)/t_ox`（负斜率：Vg 高、φ_s 低）。
2. D 连续：`ε_si·φ'_si(t_ox) = ε_ox·φ'_ox = ε_ox·(φ_s − Vg)/t_ox`。
3. 移项整理：

```
ε_si·φ'_si(t_ox) + ε_ox·(Vg − φ_s)/t_ox = 0        ← 加号！
```

4. 这是一条 **Robin（第三类）边界条件**：导数与函数值的线性组合为 0。φ_s 本身是网络在 u=0 的输出，故该项自动把界面值与界面斜率耦合起来。
5. **加号核对**（防止再犯）：物理上 Vg>φ_s（栅极加正压）⟹ φ'_ox<0 ⟹ D 连续要求 φ'_si<0（Si 侧场也指向 −z）；加号形式在 φ'_si<0、Vg−φ_s>0 时两项异号、可相互抵消为 0 ✓。若误写成减号，两项同号、不可能为 0，训练会被推向错误解。
6. **D 参考归一化**：`D_ref = ε_ox·max(|Vg|,0.1)/t_ox`（氧化层电场 × ε_ox 的典型量级），`R_iface = R_iface_phys/D_ref` 归一化到 O(1)；`max(|Vg|,0.1)` 防止平带 Vg=0 时除零。

### 8.2.2 标幺化选择（热电压 + 几何尺度）

- **输入** `u = (z−t_ox)/L_si ∈ [0,1]`（几何长度，Si 区归一化）。不用德拜长度——按本征浓度算 L_D≈33 μm，器件仅 102 nm，输入会被压到 [0,0.003]（灾难性尺度，§8.4）。
- **输出** `φ̄ = qφ/kT`（**热电压**标幺）。妙处：`p = n_i·e^{−(EF+qφ)/kT} = n_i·e^{−EF/kT}·e^{−φ̄} = p_bulk·e^{−φ̄} = N_A·e^{−φ̄}`——EF 与 n_i **同时消掉**，空穴项退化为最简形式，指数溢出从源头消失（Stage 9 的 p(φ) 活体项由此受益）。这正是选热电压而不选 Vg 或任意 V_ref 的原因。
- **导数链式法则**：`dφ/dz = (kT/q)(1/L_si)·dφ̄/du`，`d²φ/dz² = (kT/q)(1/L_si²)·d²φ̄/du²`。
- **残差归一化**：`R_pde = R_phys/(q·N_A)`——除以掺杂电荷密度，量级 O(1)。归一化的意义：PDE 项与 Robin 项都在 O(1) 尺度竞争，权重 λ_iface 才有物理意义（λ=1 即「同等重要」），训练不会偏向大数值项。

### 8.2.3 总损失

```
L = mean(R_pde²) + λ_iface·R_iface²        （soft BC 时另加 φ̄(1)²）
```

其中 `R_pde` 在 Si 区全部网格节点（~980 均匀配点）上取平均，`R_iface` 单点。

---

## 8.3 FDM vs PINN：界面处理的核心差异（为什么 FDM 没有界面难题）

这是本阶段、乃至整个 Hybrid PINN 方案**最核心的数值差异**，须在论文中讲清楚。

**FDM（Stage 3）是「有限体积 / 通量形式」，从不跨界面求导。**

其离散式（[src/poisson_fdm.py](../src/poisson_fdm.py)）为：

```
k[i-1]·φ[i-1] − (k[i-1]+k[i])·φ[i] + k[i]·φ[i+1] = −ρ[i]·cvw[i]
k[i] = ε_half[i] / dz
```

界面处半网格 ε 取**调和平均** `ε_half = 2·ε_ox·ε_si/(ε_ox+ε_si) = 5.85 ε₀`（物理来源：界面两侧两个「半胞」看作串联电容，`1/ε_eff=(1/2)(1/ε_ox+1/ε_si)`）。FDM 只操作**节点上的 φ 值**和**相邻节点间的通量**，从不求 ε 的导数，也不求 φ 跨界面的二阶导。导数跳变被「编码」进界面处 ε_half 里，电位移连续**自动且精确**满足（实测比值 3.002 vs 理论 3.000）。

**PINN 是「配置点 / 强形式」，用全局光滑函数逐点求导。**

1. φ(z) 是 tanh 网络，**处处无穷可微**（tanh ∈ C^∞，复合仍 C^∞），`dφ/dz`（autodiff）必然连续。而真实解要求它在界面**跳 3 倍**。光滑函数**表示不了不连续导数**——这是「表示能力」硬伤，不是「训练不够」。
2. 若按散度形式写残差 `−d/dz(ε·dφ/dz)`，autodiff 求 `dε/dz` 时在界面得到 **Dirac δ**（ε 是阶跃），autodiff 不处理 δ，直接算错。
3. 若按拉普拉斯形式 `d²φ̄/dz̄² + ρ̄`，等于**假设 ε 均匀**，把 3 倍介电跳变整个丢掉，物理上错（论文式 (3.2) 即此错误）。

**一句话总结**：FDM 用调和平均把介电界面「免费消化」；PINN 必须**显式处理**界面条件，否则凭空多出一个「假界面电荷」的误差。显式处理的正确方式是 §8.2.1 的 Robin 条件 + §8.1.5 的解析氧化层。

---

## 8.4 无量纲化修正（方案B §3.5 有误，须修正）

- 方案B 用本征德拜长度 `L_D=√(ε_si·kT/(q²·n_i))`，代入得 **L_D≈33 μm**，而器件仅 102 nm，`z̄∈[0,0.003]`，输入被压缩到近 0，是灾难性尺度。
- **修正**：纯 Poisson（固定 ρ、无 Boltzmann 项）用几何尺度 `z̄=z/L_total∈[0,1]` + `φ̄=φ/V_ref`；残差除以 `ρ_ref=ε₀·V_ref/L_total²` 归一化到 O(1)。热电压标幺 `φ̄=qφ/kT` 只在耦合 Boltzmann 载流子项（本阶段/Stage 9 全 SP）时才需要——本方案正是如此（§8.2.2）。

---

## 8.5 代码实现 ↔ 物理对应

### 8.5.1 网络结构与参数量

`PoissonPINN`：输入 u（1 维）→ 线性(1→64)+tanh → 3×[线性(64→64)+tanh] → 线性(64→1)，共 4 隐藏层 ×64 宽。**参数量 = (1×64+64) + 3×(64×64+64) + (64+1) = 12,673**（论文写「约 8,000」是 3 层×64 的数字，须改，§8.7B-4）。

| 物理/架构概念 | 代码 |
|---|---|
| 解参数化 φ̄_θ(u) | `PoissonPINN.forward(u)` |
| 硬约束 φ̄=(1−u)·NN(u) | `_phi_bar(u)`（`hard_constraint=True`） |
| 标幺 u=(z−t_ox)/L_si | `_to_u(z)` |
| 状态管理（续训/种子/历史） | `PoissonPINNSolver`：`warm_start`、`seed`、`loss_history`、`wall_time`、`n_epochs` |

### 8.5.2 `_loss` 的 autograd 细节

```
u = u.clone().detach().requires_grad_(True)     # 求导变量
phi_bar = self._phi_bar(u)                       # (N,1)
dphi_bar_du  = torch.autograd.grad(phi_bar, u, grad_outputs=ones, create_graph=True)[0]
d2phi_bar_du2 = torch.autograd.grad(dphi_bar_du, u, grad_outputs=ones, create_graph=True)[0]
```

两次 `autograd.grad`（第二次对一阶导图求导，`create_graph=True` 保持二阶可导）得到 d²φ̄/du²，再按 §8.2.2 链式法则换到物理量。空穴项指数截断 `clamp(−(EF+qφ)/kT, −60, 40)` 防止训练暂态 exp 溢出（上限 40 低于 float32 溢出阈值 ~51.5；截断只影响瞬态、不影响收敛解）。界面 Robin 在 `u0=[[0.0]]` 单点计算，`R_iface` 除以 D_ref 归一化。

| 物理量 | 代码 |
|---|---|
| 空穴 p=n_i·e^{−(EF+qφ)/kT}（截断） | `exp_arg=clamp(−(EF+qφ)/kT,−60,40)`; `p=n_i·exp(exp_arg)` |
| PDE 残差 `−ε_si·d²φ/dz² − q(p−n−N_A)` | `R_phys`；`R_pde=R_phys/(q·N_A)` |
| Robin（加号！）`ε_si·φ'(t_ox)+ε_ox(Vg−φ_s)/t_ox` | `R_iface_phys`；`R_iface=R_iface_phys/D_ref` |
| 总损失 | `loss = mean(R_pde²) + λ_iface·R_iface²`（soft BC 另加 `φ̄(1)²`） |

### 8.5.3 `train` 的课程学习（n-ramp 单阶段版）

`train(n_frozen, EF, params, T, Vg, warm_start=False, epochs=None, n_ramp_frac=None)`：

- `warm_start=False` 时 `_reset_model()`（from-scratch），True 时续训现有权重（Stage 10 fine-tune 的接口）；
- `n_ramp_frac` 课程：前 `ramp_epochs` 轮内 `n_eff = n_si·(i+1)/ramp_epochs` 从 0 线性升到满值——网络先学经典解再「充电」电子；
- `solve_poisson_pinn` 默认策略（n 非零且未显式传 n_ramp_frac 时）：**两阶段**——先以 n=0 训练 `max(⌊epochs/2⌋,100)` 轮，再 `warm_start=True、n_ramp_frac=0` 续训满 n 剩余轮数（§8.6.2 机理）。

### 8.5.4 `predict_full` 的装配

Si 区由 PINN 在 u_si 上输出 φ̄、乘 kT/q 还原；氧化层按 §8.1.5 线性重建 `φ[~is_si] = Vg + (φ_s−Vg)·z_ox/t_ox`，其中 `φ_s` 由网络在 u=0 单点求值。边界精度：φ(0)=Vg 由线性重建**精确**保证、φ(L)=0 由硬约束**精确**保证（测试实测偏差 <1e-14）。

### 8.5.5 汇总对照表（计划 → 实现）

| 物理量 | 计划代码（§8.5 旧表） | 实际实现 |
|---|---|---|
| 硬约束 `φ_hat=φ_BC+d(z)·NN` | `d=(z/L)(1−z/L)` | `φ̄=(1−u)·NN(u)`（Si 单域，bulk 端） |
| PDE 残差（Si 区，ε_si 常数） | autodiff 二阶导 | `_loss` 两次 `autograd.grad` |
| 界面 D 连续 | `(ε_si·φ'(t_ox)+ε_ox(Vg−φ_s)/t_ox)²` | 同（加号） |
| 总损失 | `mean(R²)+λ_iface·L_iface` | 同，各项归一化 |
| 氧化层线性重建 | `φ_ox=Vg−(Vg−φ_s)z/t_ox` | `predict_full` |

---

## 8.6 数值验证结果（实测）

验证方式：PINN vs FDM Newton（同一方程、同一冻结 n、同一边界 φ(0)=Vg/φ(L)=0）。

| 情形 | max\|Δφ\| | mean\|Δφ\| | φ_s (PINN/FDM) | 耗时 | 结论 |
|---|---|---|---|---|---|
| n=0, Vg=0.5（经典耗尽） | 0.685 mV | 0.198 mV | 0.4317 / 0.4324 | ~29 s | ✓ |
| n=0, Vg=1.0（经典耗尽） | 0.658 mV | 0.416 mV | 0.8994 / 0.8998 | ~30 s | ✓ |
| 冻结 n, Vg=1.5（强反型）直接训练 | 1137 mV | 373 mV | −0.057 / 1.080 | 28 s | ✗ 错误势阱 |
| 冻结 n + λ_iface=10（无课程） | 711 mV | 305 mV | 0.369 / 1.080 | 38 s | ✗ 权重调大无效 |
| 冻结 n + n-ramp 课程（4000 轮） | 8.64 mV | 3.81 mV | 1.0710 / 1.0796 | 38 s | ✓ |
| 冻结 n + 两阶段续训（3000+3000 轮） | **2.69 mV** | 1.49 mV | 1.0774 / 1.0796 | ~60 s | ✓✓ 最优 |
| 冻结 n，默认 config（两阶段 1500+1500 轮） | **3.21 mV** | 0.77 mV | 1.0765 / 1.0796 | 29 s | ✓ 默认设置复现 |
| soft-BC 消融（hard_constraint=False，经典 n=0, Vg=1.0） | **53.94 mV** | 26.06 mV | —（φ(L)=0.13 mV） | 44 s | ✗ 比硬约束差 ~82 倍 |

**本阶段 7 项单元测试**（`tests/test_poisson_pinn.py`）：

| 测试 | 验证内容 |
|---|---|
| `test_flatband_stays_zero` | Vg=0 唯一物理解 φ≡0 保持（max\|φ\|<5 mV，300 轮） |
| `test_dropin_classical_vs_fdm` | n=0、Vg 0.5/1.0，PINN vs FDM max\|Δφ\|<10 mV |
| `test_dropin_frozen_n_vs_fdm` | SP 自洽 n 冻结、Vg=1.5，max\|Δφ\|<10 mV |
| `test_boundaries_and_oxide_linear` | φ(0)=Vg、φ(L)=0 精确（1e-14）；氧化层二阶差分=0 |
| `test_interface_displacement_continuity` | \|ε_si·φ'+ε_ox(Vg−φ_s)/t_ox\| 相对 D_ref <0.05 |
| `test_input_validation` | T≤0 抛 ValueError；未训练 predict 抛 RuntimeError |
| `test_loss_finite_under_deep_negative_transient` | 强制 φ 深负（exp_arg 远超阈值），损失须保持有限（float32 溢出回归） |

### 8.6.1 强反型直接训练失败机理（鸡-蛋冲突，可写进论文）

界面 Robin 的斜率目标随 φ_s 变化：`φ'(t_ox) = −(ε_ox/ε_si)·(Vg−φ_s)/t_ox`。训练初期 φ_s≈0 ⟹ Robin 要求 dφ̄/du(0) ≈ **−967**（φ̄ 单位，巨陡负斜率）；而强反型电子尖峰（n ≈ 10²·N_A）要求该处 φ̄″ ≈ +58 的正曲率——两者在 u=0 处**方向冲突**。网络逃入 φ_s<0 的错误势阱（那里 p=N_A·e^{+|φ̄|} 指数暴涨，恰好「对冲」电子尖峰），损失平台在 0.92。调大 λ_iface 无效（711 mV）证明这是**方向冲突而非权重失衡**。

### 8.6.2 修复：课程学习（两阶段续训）

先以 n=0 训练经典解（φ_s 升到 ~1.3 V，Robin 斜率目标温和化），再 warm_start 续训满 n（电子尖峰变成对既有解的扰动）。这与 Gummel 外层「先经典、后量子」的物理顺序一致，且正是 Stage 9 fine-tune 的工作流。已作为 `solve_poisson_pinn` 的默认策略（n 非零时自动两阶段，各占一半轮数）；单阶段 n-ramp 作为消融选项保留。

**指标对照论文承诺（MAE<0.02 V、rel-L2<2%）**：实测 MAE 0.2–1.5 mV、rel-L2 0.10%–0.26%——**远超承诺，论文指标安全**。

---

## 8.7 与搭建说明 / 论文的前后一致性核对（2026-08-17）

**（A）与搭建说明 §11–13：代码侧全部满足**

| 搭建说明要求 | 实现 | 状态 |
|---|---|---|
| §11 输入 z → 输出 φ_NN(z) | u=(z−t_ox)/L_si（固定线性映射，等价） | ✓ |
| §11 MLP 4×64 tanh Adam | 同 | ✓ |
| §11 无量纲化 z̄/φ̄/ρ̄ | u、φ̄=qφ/kT、残差÷q·N_A | ✓ |
| §11 硬约束 φ_BC + d·N | φ̄=(1−u)·NN(u)（φ(L)=0 端） | ✓ |
| §11 PDE loss mean(R²) | Si 区（ε_si 常数）；界面 Robin 另立 | ✓ |
| §5.1 界面 D 连续 ε_si·φ'_si=ε_ox·φ'_ox | Robin 加号形式（推导见 §8.2.1） | ✓ |
| §12 只换 Poisson 模块，其余全一致 | `solve_poisson_pinn` 与 `solve_poisson_nonlinear` 同签名 | ✓ |
| §13 A from-scratch / B fine-tune | warm_start + seed | ✓ |
| §11 完成标准「固定 ρ 达到预设精度」 | 解读：SP 循环实际替换的是**非线性** Poisson（p(φ) 活体），故按「冻结 n、p(φ) 活体」验证；n=0 即经典非线性情形，覆盖更全 | ✓（需在论文说明此解读） |

**（B）与论文第三章/第四章：发现 10 处不一致（绝大多数是论文侧错误）**

1. **论文式 (3.2) 损失 = 全域 Laplacian 形式 `d²φ̄/dz̄²+ρ̄`——物理/量纲错误**。ρ̄ 只做浓度归一，ε 与 L² 因子整个丢失，等于假设 ε 均匀（把 3 倍介电跳变丢掉，同 §8.3 与方案B §4.3 的错误）。代码采用解析氧化层 + Robin（正确）。**论文第三章须改写**。
2. **论文 §2.4 用德拜长度 L_D 作长度基准**：按 n_i 计算 L_D≈33 μm，器件仅 102 nm ⟹ z̄∈[0,0.003]，灾难性尺度（同 §8.4）。代码用几何长度。**论文须改为几何长度基准**。
3. **论文式 (2.2) p(z)=n_i·exp(−qφ/kT) 缺 EF 项**：φ=0 处 p=n_i≠N_A，违反 bulk 电中性；且与 §2.2.3「EF 由全域电中性确定」自相矛盾。代码正确（p=n_i·exp(−(EF+qφ)/kT)，φ=0 处 p=N_A，推导见 §8.1.3）。**论文须补 EF**。
4. **论文 §3.2「总参数量约 8,000」**：4 层×64 实际为 12,673（§8.5.1）。**数字须改**。
5. **论文 §3.3「Adam + 余弦退火 1e-3→1e-5」**：实现为恒定 lr=1e-3。实测 3000 轮已达 0.66 mV、无需退火；论文要么实现、要么改描述（Stage 10 可做 lr 调度消融）。
6. **论文 §3.3「配点 ~500、界面高密度」**：实现为 Si 网格全部节点（~980，均匀）。强反型尖峰仅占域 ~3%（~30 点）；界面加密可作为 Stage 10 改进，否则论文改为「~980 均匀配点」。
7. **论文 §2.3「简单混合、Anderson 属后续优化」**：与 §1.2.1 及代码矛盾（Stage 7 已实现 Gummel+Newton+Anderson）。**论文须改**。
8. **论文 §3.5「Python 3.10」**：实际 3.12.13。小改。
9. **论文式 (3.1) 全域双 Dirichlet 硬约束**：与解析氧化层方案不兼容。改写后应为 `φ̄(u)=(1−u)·N(u;θ)`（Si 单域、bulk 端硬约束、界面端 Robin）。**论文第三章须补一段「介电界面难题 → 解析氧化层 + Robin」的叙述**——目前论文完全没提界面难题，而这恰是本文的工作量与创新点。
10. **论文 §3.3 增量训练（首轮 ~5000 轮、后续 ~500 轮@1e-4）**：与 warm_start 设计方向一致 ✓；另须补「两阶段课程学习」策略（§8.6 实测直接训练失败）。

**（C）前向风险清单（Stage 9/10 设计要点，2026-08-17 增补）**

1. **tol_V=1e-6 与 PINN 噪声地板的矛盾**（§8.4 已预警，现有实测佐证）：PINN 单次解精度 ~mV 级。若每轮 from-scratch 随机初始化，迭代间差异 ~mV，δ=max|φ_new−φ|<1e-6 不可达，SCF 会耗尽 max_iter。**缓解 = fine-tune + 固定 seed（论文 §3.3 自己提出的策略）**：near 收敛时 n 几乎不变、训练确定性 ⟹ 相邻两轮解可一致到远好于绝对误差，δ 可能继续下降。但论文「相同收敛阈值下迭代轮数基本一致」的结论**必须在 Stage 9 实测验证**，若不可达则如实改用停滞判据并报告。
2. **drop-in 仅签名兼容、行为不兼容（Stage 9 接口核心风险，2026-08-17 升级）**：`solve_sp` 内层以 `solve_poisson_nonlinear(device, n, EF, params, T, Vg, phi)` 调用（每轮 n 非零、用上一轮 φ 做初值），而 `solve_poisson_pinn` 每次调用都新建 solver 并从头两阶段——Stage 9 若直接替换，每轮随机重训 + 白做 n=0 warmup（~15 s/轮）+ 无法 fine-tune。**Stage 9 必须在循环外持有单个 `PoissonPINNSolver` 实例：首轮两阶段（先 n=0 再续训满 n），后续轮 `solver.train(n, warm_start=True, n_ramp_frac=0.0)`——不能用 `solve_poisson_pinn` 做 drop-in，要用类级 API。** 另注意 `n_ramp_frac` 默认值陷阱：config `n_ramp_frac: 0.5`，而 `solver.train()` 不传该参数时用 `self.n_ramp_frac=0.5` 走**单阶段 ramp**（8.64 mV），并非推荐的两阶段（2.69 mV）；两阶段默认只存在于 `solve_poisson_pinn` wrapper 内。故类级循环每轮必须显式传 `n_ramp_frac=0.0`，否则精度退化近 3 倍。
3. **固定 seed 使 from-scratch 对比失效（Stage 10 消融）**：config 里 `seed=0` 固定，from-scratch 每次同种子实际上是确定性初始化，「from-scratch vs fine-tune」消融失去统计意义（且 seed=0 恰是收敛好的种子）。另注意：yaml 里写 `seed: null` 会让 `int(None)` 抛错（`p.get('seed', 0)` 的默认值只在键不存在时生效）。Stage 10 要真随机初始化时需显式传不同 seed。
4. **强反型更深处未验证（Stage 9 精度风险）**：只验证到 Vg=1.5。Stage 9 电压扫描到 Vg≥2.0 时电子尖峰更窄更陡，3.21 mV 可能恶化。**建议 Stage 9 前补测 Vg=2.0 冻结 n 一例**。
5. **phi0 参数未使用（接口，仅效率）**：签名与 `solve_poisson_nonlinear` 一致（drop-in 要求），但传入的 phi0 被忽略——Stage 7 的电压连续化初值加速在 PINN 侧没有对应利用。Stage 10 可考虑用 phi0 做预拟合（L2 拟合初值）作为 fine-tune 的增强。
6. **全局随机种子重置副作用**：每次构造 solver 都 `torch.manual_seed(seed)` 重置全局随机状态——未来实验里 PINN 训练与其他随机过程交错时会互相干扰。可改为局部 generator。
7. **效率叙事**：单点 Hybrid 训练 27–45 s vs FDM ~ms，处于劣势；效率优势只在 Stage 12 参数化推理。论文 §1.1 的定位（参数化扫描场景）方向正确，表述须守住。
8. **Stage 12 参数化**：Vg 并入网络输入后，Robin 损失显式依赖 Vg，需把 Vg 按批并入 `_loss`——架构兼容，非障碍。

---

## 关键文献（详见 [references.md](references.md) §E 论文引用地图）

- **PINN 基础**：Raissi et al. 2019（残差损失）；Lagaris et al. 1998（硬约束试函数先驱）；Sirignano & Spiliopoulos 2018（DGM）；Karniadakis et al. 2021（综述）。
- **训练策略**：Bengio et al. 2009（课程学习——两阶段课程的依据）；Kingma & Ba 2015（Adam）。
- **界面处理**：Jagtap & Karniadakis 2020（XPINN 区域分解+界面通量连续）；Sarma et al. 2024（I-PINN 界面问题框架）——与「解析氧化层 + Robin」方案直接对应；FDM 侧对照 Patankar 1980（调和平均）。
- **半导体 PINN 近作**：Riganti et al. 2025（DDNet）；Cai et al. 2024（DAC 多阶微分网络）；Radu & Duque 2022（量子阱薛定谔）；Singhal & Agarwal 2024（EDTM 定态薛定谔）。
- **对照论证**：Grossmann et al. 2024（PINN vs FEM，Poisson 等方程）；Savović et al. 2023（FDM vs PINN 实证）。
- **物理侧**：BenDaniel & Duke 1966（BD 条件——Robin 所保证的 D 连续是其静电类比）；Patankar 1980（与 Robin 的对照见 §8.3）。
