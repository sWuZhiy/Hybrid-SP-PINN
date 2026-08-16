"""完整 FDM Schrödinger–Poisson 自洽求解器（Stage 7）。

把 Stage 3–6 的模块组装成平衡态自洽循环（项目搭建说明 §7 / Stage 7）：

    φ(0) → Schrödinger(只在 Si) → E_i, ψ_i → n(z), p(z) → ρ → 非线性 Poisson → 收敛?

自洽策略采用 **Gummel 外层（Anderson 加速）+ Newton 内层**：
  - 外层（Gummel）：给定 φ 解 Schrödinger 得量子电子密度 n(z)（冻结），
    再解非线性 Poisson 得到 G(φ)，把自洽问题写成固定点 φ = G(φ)。残差
    r = G(φ) − φ 用 **Anderson 混合**（最近 m 个历史残差的最小二乘外推，
    见 solve_sp）加速收敛——强反型区简单欠松弛需 ~100 轮，Anderson 外推
    可压缩到几十轮甚至几轮；
  - 内层（Newton）：在 n(z) 冻结下，把经典空穴项 p(φ)=n_i·e^{−(EF+qφ)/kT}
    按 d p/dφ=−q p/kT 线性化，用 Newton 迭代解非线性 Poisson。这一步对
    指数非线性无条件稳定，避免了固定点法在反型区发散（见下文「数值说明」）。

能量规范（方案 A，见《图像物理内涵说明》）：全项目能量零点定在 bulk 本征能级
E_i(bulk)=0。于是

  - 费米能级 EF 相对 E_i，沿用 Stage 6 的 find_fermi_level，**不转换**；
  - 导带底 Ec(z) = E_g/2 − q·φ(z)（比本征能级高半个带隙 E_g/2）；
  - Schrödinger 本征值 E_i_state 与 EF 同基准，占据因子直接用 (EF − E_i_state)/kT；
  - 经典空穴 p(z) = n_i·exp(−(EF + q·φ)/kT)，φ=0 处 p = n_i·e^{−EF/kT} ≈ NA，
    自动保证 bulk 电中性。

Schrödinger 只在 Si 区域求解，Si/SiO₂ 界面视为无限高势垒（ψ=0，搭建说明 §5.2），
氧化层内不求解量子态（故 ΔEc、m_ox 在本阶段未使用）。

数值说明：耗尽初值在反型区会高估表面势（约 1.4 V vs 真实 ~0.8 V），而电子密度
n(φ) 对 φ 近指数敏感（强反型下 Vg 每增加 ~60 mV，Ns 增加一个数量级），直接
固定点迭代会振荡（高 φ → n 过大 → φ 被打回负值 → n≈0，如此往复）。故用内层
Newton 处理 p 的指数非线性；外层把 G(φ) 当作固定点映射，先用无量子电子的
「经典解」作初值（量子限域对 φ 只是修正量），再用 Anderson 混合加速收敛。
正栅压下物理解满足 0 ≤ φ ≤ Vg（E(L)=0、ρ≤0 时由 Gauss 定理可证），迭代暂态
对该范围做 0.5 V 余量的截断，防止 φ 落入深负区使 exp 项溢出。
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from . import constants
from .fermi_level import find_fermi_level
from .poisson_fdm import harmonic_mean, solve_poisson_fdm
from .quantum_density import quantum_density_multi
from .schrodinger_fdm import solve_schrodinger


@dataclass
class SPResult:
    """自洽求解结果（单位均为 SI；能量相对 bulk 本征能级 E_i=0）。"""
    phi: np.ndarray                 # 静电势 [V]，形状 (n_grid,)
    Efield: np.ndarray              # 电场 -dphi/dz [V/m]
    Ec: np.ndarray                  # 导带底 Ec(z) = E_g/2 - q*phi [J]
    n: np.ndarray                   # 量子电子体密度 [1/m^3]
    p: np.ndarray                   # 经典空穴体密度 [1/m^3]
    rho: np.ndarray                 # 空间电荷密度 [C/m^3]
    subband_energies: list          # 各组能谷束缚态能级 [J]（列表，元素为 ndarray）
    subband_psi: list               # 各组能谷波函数（列表，元素为 (n_grid, k)）
    Ns_total: float                 # 总电子面密度 [1/m^2]
    Ns_per_ladder: list             # 各组能谷的子带面密度 [1/m^2]（列表）
    EF: float                       # 费米能级 [J]（相对 E_i）
    EF_Ec: float                    # 费米能级相对 bulk 导带底 Ec_bulk [J]（= EF - E_g/2）
    converged: bool                 # 是否在 max_iter 内收敛
    iterations: int                 # 实际外层迭代轮数
    history: list                   # 每轮 max|phi_new - phi_old| [V]
    Vg: float                       # 栅压 [V]


def classical_hole_density(phi, EF, n_i, T, is_si):
    """经典空穴浓度 p(z) [1/m^3]（平衡 Boltzmann，EF 相对 E_i）。

    p(z) = n_i · exp(−(EF + q·φ(z))/kT)，仅 Si 内非零（氧化层无空穴）。
    φ=0（bulk 中性区）处 p = n_i·e^{−EF/kT} ≈ NA，保证 bulk 电中性。
    """
    if T <= 0:
        raise ValueError("T 必须为正")
    phi = np.asarray(phi, dtype=float)
    is_si = np.asarray(is_si, dtype=bool)
    kT = constants.kB * T
    # 指数截断 [-60, 60]：正常解 |φ| 不超过 ~2 V（指数 <= ~93）时，截断只影响
    # 深积累（|φ| > 1.5 V）与数值暂态，防止 exp 溢出（本项目聚焦反型区）。
    exp_arg = np.clip(-(EF + constants.q * phi) / kT, -60.0, 60.0)
    p = n_i * np.exp(exp_arg)
    return np.where(is_si, p, 0.0)


def solve_subbands_si(device, phi, params, num_states):
    """在 Si 区域求解 Schrödinger，返回 (energies_list, psi_list)。

    Args:
        device: Device1D。
        phi: 静电势 [V]，形状 (n_grid,)。
        params: MaterialParams。
        num_states: 每组能谷求解的态数目。

    Returns:
        energies_list: 列表（长度 n_ladders），每项为该能谷组的束缚态能级
            （已过滤 E < E_g/2，即低于 bulk 导带底）[J]。
        psi_list: 列表（长度 n_ladders），每项为 (n_grid, k) 波函数，已补零
            到全网格（氧化层内为 0）。
    """
    is_si = device.is_si
    z_si = device.z[is_si]
    if z_si.size < 3:
        raise ValueError("Si 区域网格点不足（< 3）")

    # 导带底（E_i=0 规范）：Ec = E_g/2 - q*phi
    Ec_si = 0.5 * params.E_g - constants.q * phi[is_si]

    energies_list = []
    psi_list = []
    for mz in params.m_z:                      # 两组能谷：m_l 与 m_t
        mass_si = np.full(z_si.size, mz)
        E, psi_si = solve_schrodinger(z_si, mass_si, Ec_si, num_states)
        bound = E < 0.5 * params.E_g           # 真正束缚态：低于 bulk 导带底
        E = E[bound]
        psi_si = psi_si[:, bound]
        psi = np.zeros((device.z.size, E.size))
        psi[is_si, :] = psi_si
        energies_list.append(E)
        psi_list.append(psi)
    return energies_list, psi_list


def compute_carriers(device, phi, EF, params, T, num_states):
    """由 φ 计算电子 n(z)、空穴 p(z) 及子带信息。

    Returns:
        (n, p, subband_energies, subband_psi, Ns_total, Ns_per_ladder)。
    """
    energies_list, psi_list = solve_subbands_si(device, phi, params, num_states)
    ladders = [
        (energies_list[i], psi_list[i], params.m_par[i], params.g_v[i], params.g_s)
        for i in range(params.n_ladders)
    ]
    n, Ns_per_ladder, Ns_total = quantum_density_multi(ladders, EF, T)
    p = classical_hole_density(phi, EF, params.n_i, T, device.is_si)
    return n, p, energies_list, psi_list, Ns_total, Ns_per_ladder


def solve_poisson_nonlinear(device, n_frozen, EF, params, T, Vg,
                            phi0, phi_bulk=0.0, max_newton=40, tol_newton=1e-10):
    """Newton 求解非线性 Poisson（量子电子密度 n 冻结）。

    求解（§4.1）：

        -d/dz[eps(z) dphi/dz] = q [ p(phi) - n - NA ],  p(phi)=n_i·e^{-(EF+qφ)/kT}

    对指数型空穴项做 Newton 线性化（d p/dφ = −q p/kT），使迭代对指数非线性
    无条件稳定（避免固定点法在反型区发散）。边界：φ(0)=Vg，φ(L)=phi_bulk。

    Args:
        device: Device1D。
        n_frozen: 冻结的电子体密度 [1/m^3]。
        EF: 费米能级 [J]（相对 E_i）。
        params: MaterialParams。
        T: 温度 [K]。
        Vg: 栅压 [V]。
        phi0: 初始 φ [V]。
        phi_bulk: bulk 端电势 [V]（默认 0）。
        max_newton, tol_newton: Newton 迭代上限与收敛阈值 [V]。

    Returns:
        phi: 非线性 Poisson 的解 [V]。
    """
    z = device.z
    eps = device.eps
    NA = device.NA
    is_si = device.is_si
    if T <= 0:
        raise ValueError("T 必须为正")
    kT = constants.kB * T
    n_grid = z.size

    eps_half = harmonic_mean(eps[:-1], eps[1:])
    k = eps_half / np.diff(z)              # 半网格通量系数 [F/m^2]
    cvw = np.zeros(n_grid)
    cvw[1:-1] = 0.5 * (z[2:] - z[:-2])     # 控制体宽度

    # φ 的物理范围：正栅压下 0 <= φ <= Vg（E(L)=0、ρ<=0 时由 Gauss 定理可证，
    # 见本文件头「数值说明」），迭代暂态允许 0.5 V 余量；截断可防止暂态 φ 落入
    # 深负区使 exp 项溢出。
    phi_min = min(0.0, Vg) - 0.5
    phi_max = max(0.0, Vg) + 0.5

    phi = np.asarray(phi0, dtype=float).copy()
    for _ in range(max_newton):
        # 指数截断到 [-60, 60]：仅当 |φ| 超过 ~1.5 V 的暂态才生效（本项目
        # 反型区 |φ| <= 2 V，正常解不受影响），用于防止 exp 溢出。
        exp_arg = np.clip(-(EF + constants.q * phi) / kT, -60.0, 60.0)
        p = np.where(is_si, params.n_i * np.exp(exp_arg), 0.0)
        rho = constants.q * (p - n_frozen - NA)

        # Newton 系统 J·δφ = -R
        lower = np.zeros(n_grid - 1)
        diag = np.zeros(n_grid)
        upper = np.zeros(n_grid - 1)
        R = np.zeros(n_grid)

        i = np.arange(1, n_grid - 1)
        km = k[i - 1]
        kp = k[i]
        R[i] = (km * phi[i - 1] - (km + kp) * phi[i] + kp * phi[i + 1]) \
            + rho[i] * cvw[i]
        diag[i] = -(km + kp) - constants.q ** 2 * p[i] * cvw[i] / kT
        lower[i - 1] = km
        upper[i] = kp

        # Dirichlet 边界：φ(0)=Vg、φ(L)=phi_bulk（δφ=0）
        diag[0] = 1.0
        upper[0] = 0.0
        R[0] = phi[0] - Vg
        diag[-1] = 1.0
        lower[-1] = 0.0
        R[-1] = phi[-1] - phi_bulk

        A = diags([lower, diag, upper], offsets=[-1, 0, 1], format='csr')
        dphi = spsolve(A, -R)
        phi = np.clip(phi + dphi, phi_min, phi_max)
        if np.max(np.abs(dphi)) < tol_newton:
            break
    return phi


def solve_sp(device, Vg, config, phi0=None):
    """组装完整 Schrödinger–Poisson 自洽循环（Gummel+Newton），返回 SPResult。

    Args:
        device: Device1D（含材料剖面、网格）。
        Vg: 栅压 [V]（φ(0)=Vg，忽略功函数差，见搭建说明 §5.1 的简化假设）。
        config: 与 Device1D.config 相同的 dict（读 thermal/solver）。
        phi0: 初始静电势 [V]（可选）。若为 None，用纯耗尽近似作初值。反型区
            耗尽初值会高估表面势，建议用「电压扫描」以上一栅压的收敛解作为
            phi0（见 experiments/06_sp_baseline.py 的 Vg 扫描）。

    Returns:
        SPResult。
    """
    params = device.params
    T = float(config['thermal']['T_K'])
    num_states = int(config['solver']['num_states'])
    tol = float(config['solver']['tol_V'])
    max_iter = int(config['solver']['max_iter'])

    # 费米能级（相对 E_i，bulk 电中性，Stage 6）
    EF = find_fermi_level(params.n_i, params.NA, T).EF

    # 初值：纯耗尽 ρ = -q NA（Si），解一次线性 Poisson
    if phi0 is None:
        rho_dep = -constants.q * device.NA
        phi = solve_poisson_fdm(device, rho_dep, Vg, 0.0)
        # 先收敛到「经典解」（无量子电子，n≡0 的非线性 Poisson）：量子限域
        # 对 φ 只是修正量，经典解比耗尽解更接近真解，可大幅减少外层迭代。
        phi = solve_poisson_nonlinear(device, np.zeros(device.z.size), EF,
                                      params, T, Vg, phi)
    else:
        phi = np.asarray(phi0, dtype=float).copy()
        phi[0] = Vg
        phi[-1] = 0.0

    # Anderson 混合加速（历史深度 m_hist）：把外层迭代看作固定点 φ = G(φ)
    # （G = 量子密度 + 非线性 Poisson），用最近 m_hist 个历史残差做最小二乘
    # 外推，可把强反型区的收敛从「简单欠松弛的 ~100 轮」加速到几十轮。
    # alpha 为外推的保守混合因子（0 < alpha <= 1，强反型区取 0.5 更稳）。
    alpha = float(config['solver'].get('mixing_alpha', 1.0))
    if not (0.0 < alpha <= 1.0):
        raise ValueError('solver.mixing_alpha 必须在 (0, 1] 内')
    m_hist = 5

    # φ 的物理范围（同 solve_poisson_nonlinear 的截断，防止外推暂态越界）
    phi_min = min(0.0, Vg) - 0.5
    phi_max = max(0.0, Vg) + 0.5

    history = []
    converged = False
    iterations = 0
    x_hist = []   # 历史 φ
    r_hist = []   # 历史残差 r = G(φ) - φ

    for it in range(max_iter):
        # 外层 Gummel：由当前 φ 解 Schrödinger 得量子电子密度 n(φ)，再解
        # 非线性 Poisson 得到 G(φ)，残差 r = G(φ) - φ 度量自洽程度。
        n, p, energies_list, psi_list, Ns_total, Ns_per_ladder = compute_carriers(
            device, phi, EF, params, T, num_states)
        phi_newton = solve_poisson_nonlinear(device, n, EF, params, T, Vg, phi)
        r = phi_newton - phi
        delta = float(np.max(np.abs(r)))
        history.append(delta)
        iterations = it + 1

        x_hist.append(phi.copy())
        r_hist.append(r.copy())
        if len(x_hist) > m_hist:
            x_hist.pop(0)
            r_hist.pop(0)

        if delta < tol:
            converged = True
            break

        # Anderson 外推：min_γ ||r_k + Σ_j γ_j (r_j - r_k)||² 后
        # φ_{k+1} = φ_k + α r_k + Σ_j γ_j [(φ_j - φ_k) + α (r_j - r_k)]
        n_hist = len(x_hist)
        phi = x_hist[-1] + alpha * r_hist[-1]
        if n_hist >= 2:
            D = np.stack([r_hist[j] - r_hist[-1] for j in range(n_hist - 1)],
                         axis=1)
            gamma, *_ = np.linalg.lstsq(D, -r_hist[-1], rcond=None)
            for j in range(n_hist - 1):
                phi = phi + gamma[j] * (
                    (x_hist[j] - x_hist[-1]) + alpha * (r_hist[j] - r_hist[-1]))
        phi = np.clip(phi, phi_min, phi_max)

    # 以最终 φ 重算各量，保证输出自洽
    n, p, energies_list, psi_list, Ns_total, Ns_per_ladder = compute_carriers(
        device, phi, EF, params, T, num_states)
    rho = constants.q * (p - n - device.NA)
    Efield = -np.gradient(phi, device.z)
    Ec = 0.5 * params.E_g - constants.q * phi

    return SPResult(
        phi=phi,
        Efield=Efield,
        Ec=Ec,
        n=n,
        p=p,
        rho=rho,
        subband_energies=energies_list,
        subband_psi=psi_list,
        Ns_total=Ns_total,
        Ns_per_ladder=Ns_per_ladder,
        EF=EF,
        EF_Ec=EF - 0.5 * params.E_g,
        converged=converged,
        iterations=iterations,
        history=history,
        Vg=Vg,
    )
