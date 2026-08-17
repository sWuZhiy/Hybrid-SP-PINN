# Stage 6：费米能级 / 电中性

- **内容**：均匀 p-Si bulk 的平衡费米能级 EF，由电中性 `q(p−n−NA)=0` 唯一确定。

### 物理内涵与模型

**模型**：无带弯曲、无量子限域的均匀 p-Si bulk 里，平衡费米能级由电中性唯一确定。

- **费米能级（化学势）**：热平衡判据；同一材料各处 EF 相同。
- **电中性条件**：bulk 中总电荷为零 `p−n−NA=0`。
- **Boltzmann 近似**：非简并（NA=1e17 ≪ 态密度 ~1e19）下 `n=n_i·e^{EF/kT}`、`p=n_i·e^{-EF/kT}`（E_i=0 参考）。
- **质量作用定律**：`np=n_i²`，是上述两式的直接推论。
- **本征能级 E_i 参考**：E_i 处 `n=p=n_i`；p 型掺杂使 EF 低于 E_i（`EF≈−406 meV`）。
- **解析解**：`EF=−kT·asinh(NA/2n_i)`，由 `p−n−NA=0` 代入 sinh 恒等式得到。
- **涉及文件**：
  - [src/fermi_level.py](../src/fermi_level.py)：`carrier_densities`、`bulk_charge_density`、`analytic_fermi_level`、`find_fermi_level`。
  - [tests/test_fermi_level.py](../tests/test_fermi_level.py)：9 项测试。
  - [experiments/05_fermi_level.py](../experiments/05_fermi_level.py)：Q(EF) 穿越零点、载流子、掺杂依赖。
- **要点**：解析解 `EF = −kT·asinh(NA/2n_i)`；数值求根用 `brentq`，须显式 `xtol=1e-30`（默认 2e-12 远大于能量尺度 kT~4e-21 J，会直接返回端点）。**能量参考约定**：EF 相对本征能级 E_i（方案 A，E_i=0），p 型为负。本模块返回 `EF ≈ −406.2 meV`。

**代码 ↔ 物理对应：**

| 物理量 | 代码实现 |
|---|---|
| 平衡载流子 `n=n_i·e^{EF/kT}`、`p=n_i·e^{-EF/kT}` | `carrier_densities(EF, n_i, T)` |
| 电中性残差 `Q(EF)=q·(p-n-NA)`（随 EF 单调递减） | `bulk_charge_density` |
| 解析解 `EF=-kT·asinh(NA/2n_i)` | `analytic_fermi_level` |
| 数值求根（Brent，`xtol=1e-30`） | `find_fermi_level`（`brentq(Q, lo, hi, xtol=1e-30)`） |

**关键文献**：[references.md](references.md) §B（Sze & Ng 质量作用定律 `np=n_i²` 与电中性；Ashcroft & Mermin 非简并 Boltzmann 近似判据）。
