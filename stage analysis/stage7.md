# Stage 7：完整 FDM SP baseline

- **内容**：把 Stage 3–6 组装成平衡态自洽循环（Schrödinger→n→Poisson→…），产出 ground truth。

### 物理内涵与模型

**模型**：Schrödinger 与 Poisson 相互耦合的自洽问题——φ 决定 Ec → 决定 ψ/E → 决定 n → 反作用于 ρ → 决定新 φ。

- **自洽耦合（鸡生蛋）**：势阱由电荷决定、电荷又由束缚在势阱中的电子决定，须迭代到 φ 不再变化。
- **能带弯曲**：`Ec(z)=Eg/2−qφ(z)`；耗尽/反型时界面附近能带向下弯曲形成三角势阱。
- **耗尽 → 弱反型 → 强反型**：随 Vg 增大表面势 φ_s 上升；`φ_s=2φ_F` 时表面电子浓度达到 NA（强反型开启判据，`φ_F=(E_i−EF)/q`）。
- **量子限域的修正**：电子波函数在界面处为零、峰值在界面下 ~1 nm，等效「推离」界面 → 阈值电压比经典预期略高（量子电容/反型层电容效应）。
- **能量规范 A**：能量零点取 bulk 本征能级 E_i=0，EF 相对 E_i 不转换、亚带能级与 EF 直接可比。
- **Gummel 迭代**：冻结电荷 → 解 Poisson → 更新电荷，外层固定点 `φ=G(φ)`；用 Anderson 加速其收敛。
- **涉及文件**：
  - [src/sp_solver.py](../src/sp_solver.py)：`SPResult`、`classical_hole_density`、`solve_subbands_si`、`compute_carriers`、`solve_poisson_nonlinear`（Newton 内层）、`solve_sp`（Gummel 外层 + Anderson 加速）。
  - [tests/test_sp_solver.py](../tests/test_sp_solver.py)：9 项测试（平带、bulk 电中性、束缚态、自洽残差、电荷守恒、Vg 扫描、输入校验）。
  - [experiments/06_sp_baseline.py](../experiments/06_sp_baseline.py)：4 图 + CSV（剖面、亚带演化、Ns/φ_surf、收敛历史）。
  - [configs/default.yaml](../configs/default.yaml)：`solver` 段（`num_states=15`、`mixing_alpha=0.5`、`tol_V=1e-6`）。
- **要点**：
  - **能量规范 A**：全项目能量零点在 bulk 本征能级 E_i(bulk)=0；`Ec=Eg/2−qφ`，EF 相对 E_i 不转换；经典空穴 `p=n_i·e^{−(EF+qφ)/kT}` 在 φ=0 处回到 `p=NA`。
  - **求解策略**：Gummel 外层 + Newton 内层 + Anderson 加速（m_hist=5）；经典耗尽解作初值（量子限域对 φ 只是修正）。
  - **参考值**：EF=−406.2029 meV；反型开启 Vg≈1.00 V；Vg=1.5 → Ns=3.395e12 cm⁻²（二重 2.275e12 主导），φ_surf≈1079.6 mV；自洽残差 ~5e-7 V。
  - **已验证的收敛性**：`num_states` 从 10 加到 30 时 Ns 仅变 0.0003%（15 已充分）；网格一阶误差 ~1%（`n_grid=1000` vs `2400`，Ns 差 ~0.95%），论文需注明。

**代码 ↔ 物理对应：**

| 物理量 | 代码实现 |
|---|---|
| 导带底（规范 A）`Ec(z)=Eg/2−q·φ(z)` | `Ec = 0.5*params.E_g − q*phi` |
| 经典空穴 `p=n_i·e^{−(EF+qφ)/kT}` | `classical_hole_density`（`exp_arg=-clip((EF+qφ)/kT)`） |
| 量子电子 `n=Σ Ns_i|ψ_i|²`（复用 Stage 5） | `compute_carriers → quantum_density_multi` |
| 非线性 Poisson（Newton 线性化 `dp/dφ=−q p/kT`） | `solve_poisson_nonlinear`，Jacobian `diag[i]=−(km+kp)−q²p·cvw/kT` |
| 固定点 `φ=G(φ)` + Anderson 加速 | `solve_sp`：`r=φ_newton−φ`，`np.linalg.lstsq` 最小二乘外推 |

**关键文献**：[references.md](references.md) §C（Gummel 1964 自洽迭代开山；Anderson 1965 + Walker & Ni 2011 固定点加速与收敛理论；Kelley 2003 Newton 内层；Selberherr 1984 器件模拟教科书）+ §B（Stern & Howard 1967 反型层自洽开山）。
