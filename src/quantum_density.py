"""一维量子电子密度模块（Stage 5）。

把 Stage 4 得到的子带能级 E_i 与波函数 psi_i(z) 转成能进入 Poisson 方程
的量子电子体密度 n(z)（项目搭建说明 §4.4 / §30）。

物理：x-y 平面为二维自由电子气，z 方向量子化为若干子带；各子带按有限温
Fermi-Dirac 统计占据。子带 i 的二维面密度（单自旋 DOS 约定，§30.3）：

    Ns_i = g_s * g_v * (m_par / (2 pi hbar^2)) * kT * ln[1 + exp((EF - E_i)/kT)]

其中：
  - g_s = 2 为自旋简并；g_v 为能谷简并（(100) Si 为 2 或 4）；
  - m_par / (2 pi hbar^2) 为【单自旋】2D 态密度（m_par 为平面 DOS 质量，
    不含任何简并度）；自旋求和后等价于 m_par / (pi hbar^2)，故上式亦可写为
        Ns_i = g_v * (m_par / (pi hbar^2)) * kT * ln[1 + exp(...)]，
    即「自旋 g_s=2 已含于 π 分母，能谷 g_v 单独相乘」，二者完全一致；
  - ln[1 + exp(...)] 为有限温 Fermi-Dirac 占据积分，用 logaddexp 保证数值稳定。

最终体密度 n(z) = sum_i Ns_i * |psi_i(z)|^2，总面密度 Ns_total = sum_i Ns_i。
"""

import numpy as np

from . import constants


def sheet_density(EF, energies, T, m_par, g_v, g_s=2):
    """计算各子带的二维面密度 Ns_i [1/m^2]（有限温 Fermi-Dirac）。

    公式（单自旋 DOS，见模块 docstring）：
        Ns_i = g_s * g_v * (m_par / (2 pi hbar^2)) * kT * ln[1 + exp((EF - E_i)/kT)]

    Args:
        EF: 费米能级 [J]（标量）。
        energies: 子带能级 [J]，形状 (num_states,)。
        T: 温度 [K]，必须为正。
        m_par: 平面（x-y）DOS 有效质量 [kg]，不含简并度因子。
        g_v: 能谷简并度（整数）。
        g_s: 自旋简并度（默认 2）。

    Returns:
        Ns_i: 各子带面密度 [1/m^2]，形状 (num_states,)。
    """
    EF = float(EF)
    energies = np.asarray(energies, dtype=float)
    if T <= 0:
        raise ValueError("T 必须为正")
    if m_par <= 0:
        raise ValueError("m_par 必须为正")
    if g_s <= 0 or g_v <= 0:
        raise ValueError("g_s / g_v 必须为正")

    kT = constants.kB * T
    # 单自旋 2D 态密度 [1/(J·m^2)]
    dos_per_spin = m_par / (2.0 * np.pi * constants.hbar ** 2)
    # ln(1 + e^x)：x 很大时避免 exp 溢出
    log_occ = np.logaddexp(0.0, (EF - energies) / kT)
    return g_s * g_v * dos_per_spin * kT * log_occ


def quantum_density(energies, psi, EF, T, m_par, g_v, g_s=2):
    """由子带能级与波函数计算量子电子体密度 n(z) 与面密度（单能谷组）。

    Args:
        energies: 子带能级 [J]，形状 (num_states,)。
        psi: 归一化波函数 [m^-1/2]，形状 (n_grid, num_states)，
            各列满足 ∫ psi_i^2 dz = 1。
        EF, T, m_par, g_v, g_s: 见 sheet_density。

    Returns:
        n: 电子体密度 [1/m^3]，形状 (n_grid,)。
        Ns_i: 各子带面密度 [1/m^2]，形状 (num_states,)。
        Ns_total: 总面密度 [1/m^2] = sum(Ns_i)。
    """
    energies = np.asarray(energies, dtype=float)
    psi = np.asarray(psi, dtype=float)
    if energies.ndim != 1:
        raise ValueError("energies 必须是一维数组")
    if psi.ndim != 2 or psi.shape[1] != energies.size:
        raise ValueError("psi 形状必须为 (n_grid, num_states)")

    Ns_i = sheet_density(EF, energies, T, m_par, g_v, g_s)
    n = (psi ** 2) @ Ns_i          # 形状 (n_grid, num_states) @ (num_states,) -> (n_grid,)
    Ns_total = float(np.sum(Ns_i))
    return n, Ns_i, Ns_total


def quantum_density_multi(ladders, EF, T):
    """多能谷组（ladder）叠加，返回总密度与各组分解。

    本项目的 (100) Si 有两组能谷：二重（g_v=2, m_par=m_t）与四重
    （g_v=4, m_par=sqrt(m_l*m_t)），各自有独立的子带能级与波函数，
    需分别计算 n(z) 后叠加（§4.4 / §30.2）。

    Args:
        ladders: 列表，每项为 (energies, psi, m_par, g_v) 或
            (energies, psi, m_par, g_v, g_s) 元组，描述一组能谷。
        EF: 费米能级 [J]。
        T: 温度 [K]。

    Returns:
        n: 叠加后的电子体密度 [1/m^3]，形状 (n_grid,)。
        Ns_per_ladder: 各组子带面密度 [1/m^2] 的列表。
        Ns_total: 总面密度 [1/m^2]。
    """
    n_total = None
    Ns_per_ladder = []
    Ns_total = 0.0
    for spec in ladders:
        energies, psi, m_par, g_v = spec[:4]
        g_s = spec[4] if len(spec) > 4 else 2
        n, Ns_i, Ns_sum = quantum_density(energies, psi, EF, T, m_par, g_v, g_s)
        n_total = n if n_total is None else n_total + n
        Ns_per_ladder.append(Ns_i)
        Ns_total += Ns_sum
    return n_total, Ns_per_ladder, Ns_total
