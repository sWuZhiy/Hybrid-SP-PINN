# Stage 2：几何与材料分区

- **内容**：一维均匀网格 + 逐点材料剖面，产出贯穿全项目的 `Device1D` 核心对象。

### 物理内涵与模型

**模型**：理想 MOS 电容（Metal / SiO₂ / p-Si）沿界面法向 z 的一维描述。两条核心近似是**有效质量近似**与**完全电离近似**。

- **有效质量近似**：晶体中电子在能带极值附近满足 `E(k)≈ħ²k²/(2m*)`，宏观行为像自由粒子、只是质量换成 m*；薛定谔方程中的 `m*(z)` 由此而来。
- **p 型掺杂与完全电离**：受主杂质（如 B）能级浅（≈0.045 eV ≪ 室温 kT），室温下全部电离 `NA⁻≈NA`，这是耗尽电荷 `-q·NA` 的来源。
- **(100) Si 的能谷结构**：Si 导带底是 6 个等价的 Δ 能谷（沿 ⟨100⟩ 方向，k≈0.85(2π/a)），纵向有效质量 `m_l=0.91m0`、横向 `m_t=0.19m0`。量子限域沿 z（垂直 (100) 面）时，6 个能谷按 **z 向限域质量**分裂为**二重简并（m_z=m_l）**与**四重简并（m_z=m_t）**两组——这是后面两组亚带能级差异的根源。
- **介电常数 ε**：材料对电场的极化响应；Si=11.7、SiO₂=3.9，界面 ε 跳变是 Stage 3/8 的处理核心。
- **导带带阶 ΔEc**：Si/SiO₂ 界面两侧导带底不连续（≈3.1 eV），形成限制电子的势垒；第一版把氧化层当无限高势垒（ψ=0）处理。
- **本征载流子浓度 n_i**：无掺杂时的平衡电子/空穴浓度，是载流子统计的基准。
- **涉及文件**：
  - [src/mesh.py](../src/mesh.py)：`Mesh`、`build_mesh`（`z = linspace(0, L_total, n_grid)`，`i_interface` 标注第一个 Si 节点）。
  - [src/materials.py](../src/materials.py)：`MaterialParams`（含能谷简并 `m_z=[m_l,m_t]`、`m_par=[m_t,√(m_l·m_t)]`）、`region_mask`、`eps_profile`、`delta_ec_profile`、`mass_z_profile`、`doping_profile`。
  - [src/device.py](../src/device.py)：`Device1D` 组合网格 + 材料剖面；`load_config`。
  - [src/plotting.py](../src/plotting.py)：`plot_device_profiles`（材料剖面图）。
  - [tests/test_device.py](../tests/test_device.py)：8 项测试。
  - [experiments/00_device_check.py](../experiments/00_device_check.py)：材料剖面图 + CSV。
- **要点**：(100) Si 的 6 个 Δ 能谷按 z 向限域质量分裂为二重（`m_z=m_l`）与四重（`m_z=m_t`）两组；界面 `t_ox` 可能落在相邻格点之间，由 `i_interface` 明确标注。

**代码 ↔ 物理对应：**

| 物理模型 | 代码实现 |
|---|---|
| 三明治结构 `[0,t_ox]=SiO₂` / `[t_ox,L]=p-Si` | `build_mesh` 产出 `z`；`region_mask(z,t_ox)` 产出 `is_si` |
| 介电剖面 `ε(z)=ε_ox·1_{z<t_ox}+ε_si·1_{z≥t_ox}` | `eps_profile` = `np.where(is_si, p.eps_si, p.eps_ox)` |
| 导带带阶 `ΔEc(z)=ΔEc·1_{z<t_ox}`（氧化层势垒） | `delta_ec_profile` = `np.where(is_si, 0.0, p.delta_Ec)` |
| 掺杂剖面 `NA(z)=NA·1_{z≥t_ox}`（完全电离） | `doping_profile` = `np.where(is_si, p.NA, 0.0)` |
| (100) 能谷分裂 `m_z=[m_l,m_t]`、`m_par=[m_t,√(m_l·m_t)]`、`g_v=[2,4]` | `material_params_from_config` |

**关键文献**：[references.md](references.md) §B（Luttinger & Kohn 1955 有效质量理论；Fang & Howard 1966 六谷二重/四重分裂；AFS 1982 参数 0.916/0.190——本项目 0.91/0.19 为其舍入值，论文勿与 Sze 附录 0.98/0.19 回旋共振传统混用）。
