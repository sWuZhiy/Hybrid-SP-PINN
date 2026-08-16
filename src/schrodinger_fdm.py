"""一维 BenDaniel-Duke 有效质量薛定谔方程的 FDM 求解器（Stage 4）。

求解（项目搭建说明 §4.3 / §29）：

    [-hbar^2/2 * d/dz(1/m*(z) d/dz) + Ec(z)] psi_i(z) = E_i psi_i(z)

边界条件（§5.2）：计算域两端 psi = 0（理想氧化物无限势垒 / 有限箱近似）。
第一版只在 Si 区域求解量子态，Si/SiO2 界面视为无限高势垒。

离散：与 Poisson 求解器同构的控制体 / 通量形式。半网格处的 1/m* 取
调和平均，等价于 BenDaniel-Duke 界面条件（psi 连续、(1/m*) dpsi/dz
连续）；在均匀介质区退化为标准中心差分。

本模块只求解单电子本征问题，Ec(z) 与 m*(z) 由调用方传入（Stage 4 用
解析测试势阱，后续 Stage 用自洽得到的势能）。

实现说明：哈密顿矩阵为三对角（1D 拉普拉斯），用 scipy 的三对角本征求解器
eigh_tridiagonal（只取最低若干本征对），比稠密 eigh 快 1–2 个数量级；
这在 Stage 7 自洽循环（每轮多次求解）中至关重要。
"""

import numpy as np
from scipy.linalg import eigh_tridiagonal

from . import constants


def _tridiag_coeffs(z, mass, Ec):
    """构建并校验哈密顿量，返回内部节点的三对角系数 (diag, offdiag)。

    Args:
        z: 节点坐标 [m]，严格递增、均匀，形状 (n,)，n >= 3。
        mass: z 向有效质量 m*(z) [kg]，形状 (n,)，必须为正。
        Ec: 势能 Ec(z) [J]，形状 (n,)。

    Returns:
        diag: 对角元 [J]，形状 (n-2,)。
        offdiag: 次对角元 [J]，形状 (n-3,)（H[i, i+1] = H[i+1, i]）。
    """
    z = np.asarray(z, dtype=float)
    mass = np.asarray(mass, dtype=float)
    Ec = np.asarray(Ec, dtype=float)
    n = z.size

    if n < 3:
        raise ValueError("网格点数必须 >= 3")
    if mass.shape != (n,) or Ec.shape != (n,):
        raise ValueError("mass / Ec 长度必须与 z 一致")
    if np.any(np.diff(z) <= 0):
        raise ValueError("z 必须严格递增")
    if np.any(mass <= 0):
        raise ValueError("mass 必须为正")
    dz = np.diff(z)
    if not np.allclose(dz, dz[0]):
        raise ValueError("本模块当前要求均匀网格")
    dz = dz[0]

    # 半网格 a_{i+1/2} = hbar^2 / (m_i + m_{i+1})，BenDaniel-Duke 条件
    a_half = constants.hbar ** 2 / (mass[:-1] + mass[1:])   # [J·m^2]，形状 (n-1,)

    m = n - 2                       # 内部节点数
    km = a_half[:-1] / dz ** 2      # 左半网格系数 a_{i-1/2}/dz^2，形状 (m,)
    kp = a_half[1:] / dz ** 2       # 右半网格系数 a_{i+1/2}/dz^2，形状 (m,)

    diag = (km + kp) + Ec[1:-1]     # 对角元
    offdiag = -kp[:-1]              # 次对角元 H[i, i+1] = -a_{i+1/2}/dz^2
    return diag, offdiag


def build_hamiltonian(z, mass, Ec):
    """构建 BenDaniel-Duke 有效质量哈密顿矩阵（psi=0 Dirichlet BC）。

    通量/控制体离散（与 poisson_fdm.solve_poisson 同构，系数 eps -> a）：

        a_{i+1/2} = hbar^2 / (m_i + m_{i+1})          # (1/m*) 的调和平均

    均匀网格下内部节点 i 的三对角动能矩阵（T = -hbar^2/2 · d/dz[(1/m)d/dz]）：

        T[i, i-1] = -a_{i-1/2}/dz^2
        T[i, i]   = +(a_{i-1/2} + a_{i+1/2})/dz^2
        T[i, i+1] = -a_{i+1/2}/dz^2

    Args:
        z: 节点坐标 [m]，严格递增、均匀，形状 (n,)，n >= 3。
        mass: z 向有效质量 m*(z) [kg]，形状 (n,)，必须为正。
        Ec: 势能 Ec(z) [J]，形状 (n,)。

    Returns:
        H: 内部节点 (n-2, n-2) 的实对称哈密顿矩阵 [J]。
    """
    diag, offdiag = _tridiag_coeffs(z, mass, Ec)
    m = diag.size
    H = np.zeros((m, m))
    idx = np.arange(m)
    H[idx, idx] = diag
    H[idx[:-1], idx[1:]] = offdiag
    H[idx[1:], idx[:-1]] = offdiag
    return H


def solve_schrodinger(z, mass, Ec, num_states):
    """求解 BenDaniel-Duke 本征问题，返回最低 num_states 个束缚态。

    Args:
        z: 节点坐标 [m]（见 build_hamiltonian）。
        mass: z 向有效质量 m*(z) [kg]。
        Ec: 势能 Ec(z) [J]。
        num_states: 返回的态数目（>= 1）。

    Returns:
        energies: 升序能级 [J]，形状 (k,)，其中 k = min(num_states, n-2)。
        wavefunctions: 归一化波函数 [m^-1/2]，形状 (n, k)，端点为 0；
            各列已按「最大幅值处取正」固定符号，保证输出可复现。
    """
    if num_states < 1:
        raise ValueError("num_states 必须 >= 1")

    diag, offdiag = _tridiag_coeffs(z, mass, Ec)
    m = diag.size                       # 内部节点数 = n - 2
    k = min(num_states, m)

    # 三对角本征求解，只取最低 k 个本征对
    evals, psi_int = eigh_tridiagonal(
        diag, offdiag, select='i', select_range=(0, k - 1))
    evals = np.atleast_1d(evals)
    psi_int = np.atleast_2d(psi_int)
    if psi_int.shape[0] != m:                   # k==1 时可能返回 (m,)，统一为 (m, k)
        psi_int = psi_int.T

    n = z.size
    psi = np.zeros((n, k))
    psi[1:-1, :] = psi_int

    # 物理归一化：∫ psi^2 dz = 1（梯形积分）
    norms = np.sqrt(np.trapezoid(psi ** 2, z, axis=0))
    psi = psi / norms

    # 固定符号：最大幅值处取正
    imax = np.argmax(np.abs(psi), axis=0)
    signs = np.sign(psi[imax, np.arange(k)])
    signs = np.where(signs == 0.0, 1.0, signs)
    psi = psi * signs[None, :]

    return evals, psi
