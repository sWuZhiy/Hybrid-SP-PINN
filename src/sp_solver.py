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
from .poisson_pinn import PoissonPINNSolver
from .quantum_density import quantum_density_multi
from .schrodinger_fdm import solve_schrodinger


@dataclass
class SPResult:
    """自洽求解结果（单位均为 SI；能量相对 bulk 本征能级 E_i=0）。"""
    phi: np.ndarray                 # 静电势 [V]，形状 (n_grid,)
    Efield: np.ndarray              # 电场 -dphi/dz [V/m]
    Ec: np.ndarray                  # 导带底 Ec(z) = E_g/2 - q*phi [J]（仅 Si 内有物理意义；氧化层内只是 E_g/2 - q*phi 的算术值，不具物理导带底含义）
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


def _make_fdm_poisson_step(device, EF, params, T, Vg, config=None):
    """内层 Poisson 求解器（FDM Newton）：poisson_step(n, phi) -> phi_new。

    phi 作为 Newton 迭代初值（即当前 Gummel 迭代的 φ），与 Stage 7 内层
    行为完全一致。
    """
    def step(n, phi):
        return solve_poisson_nonlinear(device, n, EF, params, T, Vg, phi)
    return step


def _make_pinn_poisson_step(device, EF, params, T, Vg, config):
    """内层 Poisson 求解器（PINN，有状态 warm-start）：poisson_step(n, phi) -> phi_new。

    Stage 9 混合循环在循环外持**单个** PoissonPINNSolver（权重跨轮复用），
    不能每轮新建 solver。内层每次**训足固定轮数到收敛**（不用早停）：
      - 首轮：两阶段课程（先以 n=0 训练经典解建立正确势阱，再续训满 n）；
      - 后续轮：warm_start=True、n_ramp_frac=0.0 续训 scf_epochs 轮。

    为什么不用早停：强反型下 loss_pde 在电子尖峰处停滞在 ~7e-2，而界面 Robin
    残差仍在下降——「损失停滞」≠「解收敛」，实测早停会停在 Robin≈0.5 的坏局部
    极小（φ_s 既非平带也非物理解）。故 Stage 9 内层训足固定轮数（首轮 epochs、
    续训 scf_epochs=3000），实测收敛到 Robin≈0.004。

    为什么要训到收敛（方案一，stage9.md §9.4）：外层 Gummel+Anderson 固定点
    迭代的全部收敛理论都要求内层 G(n) 是**静态映射**（给定 n 唯一决定 φ_new）。
    此前固定 scf_epochs=500 每轮不收敛，G 随网络权重每轮漂移，Anderson 在
    「由不同映射产生的残差」上外推失效，强反型漂移到伪不动点却报收敛。
    训足轮数后 G 近似静态，Anderson 才合法；伪不动点由 _check_physical 兜底。
    phi 参数当前仅用于签名兼容（PINN 靠 warm_start 而非初值跟踪 φ；用 φ 做
    预拟合初值是 Stage 10 的 fine-tune 策略，见 stage8.md §8.7C-5）。
    """
    solver = PoissonPINNSolver(device, config)
    n_epochs = solver.epochs
    p_cfg = config.get('pinn', {})
    scf_epochs = int(p_cfg.get('scf_epochs', n_epochs))
    initialized = False
    # 界面（第一个 Si 点）与材料量，供物理守卫用
    i0 = int(np.argmax(device.is_si))
    eps_si = device.params.eps_si
    eps_ox = device.params.eps_ox
    t_ox = device.t_ox

    def _check_physical(phi_out):
        """方案二：物理守卫，防止内层漂移到伪不动点却报收敛。

        伪不动点有两个：平带（φ_s≈0，Vg=1.5 崩溃）与全转移（φ_s≈Vg，
        Vg=2.0 崩溃）。二者都满足外层 δ<tol（被误判收敛），但都违反界面
        电位移连续（Robin）：ε_si φ'_si(t_ox) + ε_ox(Vg−φ_s)/t_ox = 0。
        本守卫在每轮内层解产出后检查 φ_s 是否越界、Robin 残差是否过大，
        违反则抛错（中止而非假收敛）。
        """
        if not np.isfinite(phi_out).all():
            raise RuntimeError(
                "PINN 训练发散（输出含 NaN/Inf）。强反型区从零初值的经典解会高估 "
                "φ_s，产生远超物理量级的暂态电子尖峰 n（~1e4·NA），tanh MLP 无法"
                "表达（FDM 的 Newton 内层对这种尖峰无条件稳定，PINN 的梯度下降则"
                "不行）。请改用电压扫描初值 phi0（上一栅压的收敛解）。")
        phi_s = float(phi_out[i0])
        lo, hi = min(0.0, Vg), max(0.0, Vg)
        # 容差 1e-3 V：PINN 单解噪声地板 ~1e-4 V，平带（Vg=0）φ_s 有 ~1e-6 V 数值
        # 误差，1e-9 会把它误判为越界；1 mV 远低于伪不动点的偏差（~100 mV），
        # 仍能可靠拦截 φ_s<0 或 φ_s>Vg 的粗劣越界（全转移伪不动点）。
        if not (lo - 1e-3 <= phi_s <= hi + 1e-3):
            raise RuntimeError(
                f"PINN 内层解非物理：表面势 φ_s={phi_s*1e3:.3f} mV 越出 "
                f"[{lo*1e3:.1f}, {hi*1e3:.1f}] mV（正栅压下 0≤φ_s≤Vg，Gauss 定理）。"
                "这是伪不动点（全转移 φ_s≈Vg），已中止而非假收敛。")
        dz0 = device.z[i0 + 1] - device.z[i0]
        dphi_si = (phi_out[i0 + 1] - phi_out[i0]) / dz0
        R_iface = eps_si * dphi_si + eps_ox * (Vg - phi_s) / t_ox
        D_ref = eps_ox * max(abs(Vg), 0.1) / t_ox
        if abs(R_iface) / D_ref > 0.1:
            raise RuntimeError(
                f"PINN 内层解非物理：界面电位移不连续 |R_iface|/D_ref="
                f"{abs(R_iface)/D_ref:.3f} > 0.1（应为 0，φ_s={phi_s*1e3:.1f} mV）。"
                "这是伪不动点（平带 φ_s≈0），已中止而非假收敛。")

    def step(n, phi):
        nonlocal initialized
        n_arr = np.asarray(n, dtype=float)
        if not initialized:
            if np.max(np.abs(n_arr)) > 0.0:
                warm_ep = max(int(round(0.5 * n_epochs)), 100)
                solver.train(np.zeros(device.z.size), EF, params, T, Vg,
                             epochs=warm_ep)
                solver.train(n_arr, EF, params, T, Vg, warm_start=True,
                             epochs=n_epochs - warm_ep, n_ramp_frac=0.0)
            else:
                solver.train(n_arr, EF, params, T, Vg)
            initialized = True
        else:
            solver.train(n_arr, EF, params, T, Vg, warm_start=True,
                         epochs=scf_epochs, n_ramp_frac=0.0)
        phi_out = solver.predict_full(EF, params, T, Vg)
        _check_physical(phi_out)
        return phi_out

    return step


def solve_sp(device, Vg, config, phi0=None):
    """FDM 版 SP 自洽求解（Stage 7，内层 Poisson 用 Newton），返回 SPResult。"""
    return _solve_sp(device, Vg, config, phi0, _make_fdm_poisson_step)


def solve_sp_pinn(device, Vg, config, phi0=None):
    """Hybrid SP-PINN（Stage 9，内层 Poisson 用 PINN），返回 SPResult。

    网格/材料/Schrödinger/密度公式/EF/mixing 与 solve_sp 完全相同，唯一差别是
    内层 Poisson 用 PINN warm-start（搭建说明 §35.2）。收敛判据结构相同（δ < tol），
    但 tol 用更松的 tol_V_pinn：PINN 单解有 ~1e-4 V 噪声地板，1e-6 不可达（见
    stage9.md）。

    强反型区（Vg ≳ 1.5）必须用电压扫描初值 phi0（上一栅压收敛解）：从零初值的
    经典解会高估 φ_s（~1.4 V vs 真实 ~1.1 V）并产生 ~1e4·NA 的暂态电子尖峰，
    tanh MLP 无法表达（训练发散）。这是 PINN 相对 FDM 的一个已知鲁棒性差异，
    Stage 9 的受控对照统一用电压扫描初值（见 experiments/08_hybrid_sp_pinn.py）。

    即使有电压扫描初值，Anderson 外推仍会在强反型区把 φ_s 超调几十 mV，经 n(φ)
    指数放大出远超真解的窄瞬态尖峰（实测 49 mV → n_max≈1900·NA），tanh MLP 照样
    表达不了（NaN 发散）。故 solve_sp_pinn 额外用外层信任域 max_dphi_V_pinn 限制
    每步 φ 变化，从源头防止超调（见 stage9.md §9.5）。
    """
    tol = float(config['solver'].get('tol_V_pinn', 5e-4))
    max_dphi = config['solver'].get('max_dphi_V_pinn', None)
    if max_dphi is not None:
        max_dphi = float(max_dphi)
    return _solve_sp(device, Vg, config, phi0, _make_pinn_poisson_step,
                     tol=tol, max_dphi=max_dphi)


def _solve_sp(device, Vg, config, phi0, make_step, tol=None, max_dphi=None):
    """组装完整 Schrödinger–Poisson 自洽循环（Gummel+Anderson），返回 SPResult。

    内层 Poisson 由 make_step(device, EF, params, T, Vg, config) 返回的
    poisson_step(n, phi) -> phi_new 提供（FDM Newton 或 PINN）。两版共享
    **完全相同**的网格、材料、Schrödinger、密度公式、EF、mixing 与收敛
    判据结构（δ < tol），唯一变量是 Poisson 子问题求解器——即 Stage 9 的
    受控对照（搭建说明 §35.2）。tol 取值因内层求解器而异：FDM 可达 1e-6，
    PINN 受噪声地板 ~1e-4 限制只能用更松的 tol_V_pinn（见 stage9.md）。

    Args:
        device: Device1D（含材料剖面、网格）。
        Vg: 栅压 [V]（φ(0)=Vg，忽略功函数差，见搭建说明 §5.1 的简化假设）。
        config: 与 Device1D.config 相同的 dict（读 thermal/solver）。
        phi0: 初始静电势 [V]（可选）。若为 None，用纯耗尽近似作初值。反型区
            耗尽初值会高估表面势，建议用「电压扫描」以上一栅压的收敛解作为
            phi0（见 experiments/06_sp_baseline.py 的 Vg 扫描）。
        make_step: 内层 Poisson 求解器工厂，签名 (device, EF, params, T, Vg,
            config) -> poisson_step(n, phi) -> phi_new。
        tol: 收敛阈值 [V]（默认 None → 读 config['solver']['tol_V']）。PINN
            循环由 solve_sp_pinn 传入更松的 tol_V_pinn（噪声地板 ~1e-4）。
        max_dphi: 外层信任域——每步 φ 变化上限 [V]（默认 None 不限制）。仅 PINN
            循环由 solve_sp_pinn 传入，防止 Anderson 超调产生 tanh MLP 表达不了
            的窄瞬态电子尖峰（见 stage9.md）；FDM 内层对任意 n 稳定，保持 None。

    Returns:
        SPResult。
    """
    params = device.params
    T = float(config['thermal']['T_K'])
    num_states = int(config['solver']['num_states'])
    tol = float(config['solver']['tol_V']) if tol is None else float(tol)
    max_iter = int(config['solver']['max_iter'])

    # 费米能级（相对 E_i，bulk 电中性，Stage 6）
    EF = find_fermi_level(params.n_i, params.NA, T).EF

    # 内层 Poisson 求解器（FDM Newton 或 PINN warm-start）
    poisson_step = make_step(device, EF, params, T, Vg, config)

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
        phi_newton = poisson_step(n, phi)
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
        # 信任域（仅 PINN 用，max_dphi 非 None）：把 Anderson 外推后的步长限制在
        # ±max_dphi，防止强反型区 φ_s 超调 → n(φ) 指数放大出 tanh MLP 表达不了的
        # 窄瞬态尖峰（实测 49 mV 超调 → n_max≈1900·NA → NaN 发散）。FDM 内层
        # Newton 对任意 n 无条件稳定，max_dphi=None 不启用。
        if max_dphi is not None:
            dphi = phi - x_hist[-1]
            dphi = np.clip(dphi, -max_dphi, max_dphi)
            phi = x_hist[-1] + dphi
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
