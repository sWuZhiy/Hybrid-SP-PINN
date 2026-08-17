# Stage 5：量子电子密度

- **内容**：把子带能级/波函数转成进入 Poisson 的量子电子体密度 `n(z)`。

### 物理内涵与模型

**模型**：z 方向量子化 + x-y 平面自由 → 二维电子气的子带统计。

- **二维态密度（DOS）**：单一子带的 2D DOS 为 `m*/(2πħ²)`（每自旋），与能量无关（2D 系统的特征，区别于 3D 的 √E 依赖）。
- **Fermi-Dirac 占据（有限温）**：子带能级 E_i 的占据由费米分布决定，对能量积分得 `Ns_i ∝ kT·ln(1+e^{(EF−E_i)/kT})`。
- **子带概念**：每个 z 量子态 + 其面内动能构成一条 2D 子带；不同子带能级不同、占据不同。
- **简并度**：自旋 `g_s=2` 与能谷 `g_v`（二重/四重）各自相乘；`g_s=2` 已并入 π 分母（`2·(1/2π)=1/π`）。
- **电子密度合成**：`n(z)=Σ_i Ns_i·|ψ_i(z)|²`——面密度 × 概率密度，把子带占据还原成空间分布。
- **数值稳定**：`ln(1+e^x)` 在 x 大时用 `np.logaddexp(0,x)` 避免 `exp` 溢出。
- **涉及文件**：
  - [src/quantum_density.py](../src/quantum_density.py)：`sheet_density`、`quantum_density`、`quantum_density_multi`。
  - [tests/test_quantum_density.py](../tests/test_quantum_density.py)：9 项测试。
  - [experiments/04_quantum_density.py](../experiments/04_quantum_density.py)：n(z)、Ns、温度/能谷组叠加。
- **要点**：子带面密度 `Ns_i = g_s·g_v·(m_par/2πħ²)·kT·ln[1+exp((EF−E_i)/kT)]`（有限温 Fermi-Dirac，非零温）；自旋 `g_s=2` 吸收进 π 分母，等价写作 `g_v·(m_par/πħ²)·…`。`n(z)=Σ_i Ns_i·|ψ_i|²`。用 `np.logaddexp` 保证数值稳定。

**代码 ↔ 物理对应：**

| 物理量 | 代码实现 |
|---|---|
| 单自旋 2D 态密度 `m_par/(2πħ²)` | `dos_per_spin = m_par/(2π*hbar**2)` |
| 费米占据积分 `ln[1+exp((EF−E_i)/kT)]` | `log_occ = np.logaddexp(0, (EF-energies)/kT)` |
| 子带面密度 `Ns_i = g_s·g_v·DOS·kT·log_occ` | `g_s*g_v*dos_per_spin*kT*log_occ` |
| 体密度 `n(z) = Σ_i Ns_i·|ψ_i(z)|²` | `n = (np.abs(psi)**2) @ Ns_i` |
| 多能谷组叠加 | `quantum_density_multi` 循环各 ladder 求和 |
