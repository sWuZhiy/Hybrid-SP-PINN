# Stage 3：Poisson-FDM 独立验证

- **内容**：一维 Poisson `-d/dz(ε dφ/dz) = ρ` 的 FDM 求解，两端 Dirichlet。

### 物理内涵与模型

**模型**：静电势 φ 与电荷密度 ρ 的关系由 Poisson 方程（Gauss 定律的微分形式）给出。

- **Poisson 方程**：`∇·D = ρ`，代入 `D=εE`、`E=−∇φ` 得一维 `-d/dz(ε dφ/dz)=ρ`。
- **边界条件**：栅极 `φ(0)=Vg`（金属等势）、bulk `φ(L)=0`（远场中性区电势零点），两端 Dirichlet。
- **材料界面条件**：界面无片电荷时，法向电位移连续 `ε_ox·E_ox = ε_si·E_si`，即 `dφ/dz` 在界面跳变 `ε_si/ε_ox` 倍；由半网格调和平均 ε 自动保证。
- **调和平均的物理来源**：把界面两侧两个「半胞」看作串联电容，等效 ε 满足 `1/ε_eff=(1/2)(1/ε_ox+1/ε_si)` → `ε_eff=2ε_oxε_si/(ε_ox+ε_si)`。
- **耗尽电荷模型**：耗尽区 `ρ=−q·NA`（电离受主），氧化层 `ρ=0`。
- **涉及文件**：
  - [src/poisson_fdm.py](../src/poisson_fdm.py)：`harmonic_mean`、`solve_poisson`、`solve_poisson_fdm`。
  - [tests/test_poisson_fdm.py](../tests/test_poisson_fdm.py)：8 项测试（解析解对照、收敛阶、界面 D 连续）。
  - [experiments/02_poisson_fdm.py](../experiments/02_poisson_fdm.py)：解析对照 + 收敛阶 + MOS 耗尽冒烟。
- **要点**：**通量/控制体离散**，半网格 ε 取调和平均 `2ε_oxε_si/(ε_ox+ε_si)`，界面处自动满足法向电位移 D 连续——这是后文 Stage 8 与 PINN 对照的关键（见下）。

**代码 ↔ 物理对应：**

物理方程 `-d/dz(ε dφ/dz) = ρ`（内部节点 i）离散为 `k_{i-1}φ_{i-1}-(k_{i-1}+k_i)φ_i+k_iφ_{i+1} = -ρ_i·cvw_i`：

| 物理量 | 代码实现 |
|---|---|
| 半网格通量系数 `k_i = ε_{i+1/2}/dz` | `k = _eps_half(eps) / dz` |
| 界面等效 ε（调和平均）`2ε_iε_{i+1}/(ε_i+ε_{i+1})` | `harmonic_mean(a,b)` |
| 控制体宽度 `cvw_i=(z_{i+1}-z_{i-1})/2` | `cvw = 0.5*(z[i+1]-z[i-1])` |
| 三对角主/次对角元 | `diag[i]=-(km+kp)`、`lower=km`、`upper=kp` |
| Dirichlet `φ(0)=Vg, φ(L)=0` | `b[0]=phi_left`、`b[-1]=phi_right` |

**关键文献**：[references.md](references.md) §C（Patankar 1980 界面系数调和平均的标准做法；Selberherr 1984 半导体器件离散化；LeVeque 2002 有限体积法）。
