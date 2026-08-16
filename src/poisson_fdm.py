"""一维 Poisson 方程的有限差分（FDM）求解器（Stage 3）。

求解（项目搭建说明 §4.1 / §28）：

    -d/dz [ eps(z) * dphi/dz ] = rho(z)

采用控制体 / 通量形式离散，半网格位置取介电常数的调和平均，从而在
Si/SiO2 界面处自动满足法向电位移 D = eps * dphi/dz 的连续性（无界面
片电荷时）。边界条件为两端 Dirichlet：

    phi(0) = phi_gate（栅极侧电势），phi(L) = 0（bulk 中性区，电势零点）

本模块只求解 Poisson 单模块，不涉及 Schrödinger；电荷密度 rho(z) 由
调用方传入（Stage 3 用解析测试剖面，后续 Stage 用自洽得到的电荷）。
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def harmonic_mean(a, b):
    """两数组的逐点调和平均（用于半网格介电常数）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 2.0 * a * b / (a + b)


def _eps_half(eps):
    """半网格位置（节点 i 与 i+1 中点）的介电常数，取调和平均。

    调和平均是 1D 介质界面处保证电位移 D 连续的等效介电常数；在均匀区
    自动退化为常数值。
    """
    eps = np.asarray(eps, dtype=float)
    return harmonic_mean(eps[:-1], eps[1:])


def solve_poisson(rho, z, eps, phi_left, phi_right):
    """求解 -d/dz[eps(z) dphi/dz] = rho(z)，Dirichlet 边界条件。

    通量形式离散（内部节点 i = 1..n-2）：

        k[i-1]*phi[i-1] - (k[i-1] + k[i])*phi[i] + k[i]*phi[i+1]
            = -rho[i] * cvw[i]

    其中 k[i] = eps[i+1/2] / dz[i]（半网格通量系数），cvw[i] 为节点 i 的
    控制体宽度 = (z[i+1] - z[i-1]) / 2。

    Args:
        rho: 电荷密度 [C/m^3]，形状 (n,)，与 z 同长。
        z: 节点坐标 [m]，严格递增，形状 (n,)。
        eps: 介电常数 [F/m]，形状 (n,)。
        phi_left: z[0]（栅极侧）电势 [V]。
        phi_right: z[-1]（bulk 侧）电势 [V]。

    Returns:
        phi: 电势 [V]，形状 (n,)。

    Raises:
        ValueError: 网格点数 < 2、长度不一致、z 非严格递增或 eps 非正。
    """
    rho = np.asarray(rho, dtype=float)
    z = np.asarray(z, dtype=float)
    eps = np.asarray(eps, dtype=float)
    n = z.size

    if n < 2:
        raise ValueError("网格点数必须 >= 2")
    if rho.shape != (n,) or eps.shape != (n,):
        raise ValueError("rho / eps 长度必须与 z 一致")
    if np.any(np.diff(z) <= 0):
        raise ValueError("z 必须严格递增")
    if np.any(eps <= 0):
        raise ValueError("eps 必须为正")

    dz = np.diff(z)                       # 节点间距 [m]
    k = _eps_half(eps) / dz               # 半网格通量系数 [F/m^2]，形状 (n-1,)

    # 三对角矩阵（CSR）：A[i, i-1], A[i, i], A[i, i+1]
    lower = np.zeros(n - 1)
    diag = np.zeros(n)
    upper = np.zeros(n - 1)
    b = np.zeros(n)

    # 内部节点（向量化）
    i = np.arange(1, n - 1)
    km = k[i - 1]                         # 左半网格通量系数
    kp = k[i]                             # 右半网格通量系数
    cvw = 0.5 * (z[i + 1] - z[i - 1])     # 控制体宽度
    lower[i - 1] = km
    diag[i] = -(km + kp)
    upper[i] = kp
    b[i] = -rho[i] * cvw

    # Dirichlet 边界
    diag[0] = 1.0
    upper[0] = 0.0
    b[0] = phi_left
    diag[-1] = 1.0
    lower[-1] = 0.0
    b[-1] = phi_right

    A = diags([lower, diag, upper], offsets=[-1, 0, 1], format='csr')
    phi = spsolve(A, b)
    return np.asarray(phi, dtype=float)


def solve_poisson_fdm(device, rho, phi_gate, phi_bulk=0.0):
    """在 Device1D 上求解 Poisson，栅极侧 φ=phi_gate，bulk 侧 φ=phi_bulk。

    便捷封装：直接使用 device 的网格 z 与介电常数剖面 eps。

    Args:
        device: Device1D 实例。
        rho: 电荷密度 [C/m^3]，形状 (n_grid,)。
        phi_gate: z[0]（栅极侧）电势 [V]。
        phi_bulk: z[-1]（bulk 侧）电势 [V]，默认 0。

    Returns:
        phi: 电势 [V]，形状 (n_grid,)。
    """
    return solve_poisson(rho, device.z, device.eps, phi_gate, phi_bulk)
