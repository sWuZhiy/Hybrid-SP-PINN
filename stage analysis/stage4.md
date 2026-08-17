# Stage 4：Schrödinger-FDM 独立验证

- **内容**：BenDaniel-Duke 有效质量薛定谔方程 `[-ħ²/2·d/dz((1/m*)d/dz)+Ec]ψ = Eψ` 的对角化。

### 物理内涵与模型

**模型**：量子限域下电子的定态薛定谔方程，采用有效质量/包络函数近似。

- **定态薛定谔方程**：`Hψ=Eψ`，本征值 E 是允许能量（亚带能级），本征函数 ψ 是包络波函数。
- **BenDaniel-Duke 边界条件**：在有效质量突变处，`ψ` 与 `(1/m*)·dψ/dz` 连续（保证概率流守恒）；代码用 `1/m*` 的调和平均 `a_{i+1/2}=ħ²/(m_i+m_{i+1})` 实现。
- **理想氧化层无限势垒**：SiO₂ 的 ΔEc≈3.1 eV 远大于亚带能量尺度，第一版把界面当 `ψ=0` 的硬墙——物理上对应「电子被完全限制在 Si 内」。
- **量子限域与能级量子化**：z 方向被限制在几 nm 的势阱里，能级分立为亚带；阱越窄、有效质量越大，基态能级越高。
- **波函数归一化**：`|ψ(z)|²dz` 是找到电子的概率，故 `∫|ψ|²dz=1`。
- **涉及文件**：
  - [src/schrodinger_fdm.py](../src/schrodinger_fdm.py)：`build_hamiltonian`、`solve_schrodinger`（用 `scipy.linalg.eigh_tridiagonal`）。
  - [tests/test_schrodinger_fdm.py](../tests/test_schrodinger_fdm.py)：10 项测试（无限/有限/三角阱 + 收敛阶）。
  - [experiments/03_schrodinger_fdm.py](../experiments/03_schrodinger_fdm.py)：三类势阱波函数/能级 + 收敛阶。
- **要点**：半网格 `a_{i+1/2}=ħ²/(m_i+m_{i+1})`（1/m* 的调和平均）等价 BenDaniel-Duke 界面条件；三对角本征求解比稠密 `eigh` 快 1–2 个数量级，是 Stage 7 自洽循环提速的关键。第一版仅在 Si 区求解（界面无限高势垒 ψ=0）。

**代码 ↔ 物理对应：**

物理 `[-ħ²/2·d/dz((1/m*)d/dz)+Ec]ψ = Eψ`：

| 物理量 | 代码实现 |
|---|---|
| BenDaniel-Duke 半网格系数 `a_{i+1/2}=ħ²/(m_i+m_{i+1})`（≡ ħ²/2 × (1/m*) 调和平均） | `a_half = hbar**2/(mass[:-1]+mass[1:])` |
| 动能对角 `(a_{i-1/2}+a_{i+1/2})/dz² + Ec_i` | `diag = (km+kp) + Ec[1:-1]` |
| 动能次对角 `-a_{i+1/2}/dz²` | `offdiag = -kp[:-1]` |
| 最低 N 个本征对 | `eigh_tridiagonal(diag, offdiag, select='i', select_range=(0,k-1))` |
| 归一化 `∫ψ²dz=1`、符号固定 | `psi = psi/sqrt(trapezoid(psi²,z))` + 最大幅值取正 |
