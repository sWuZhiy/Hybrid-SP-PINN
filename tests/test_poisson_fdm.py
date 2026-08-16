"""poisson_fdm 的单元测试（Stage 3）。

覆盖项目搭建说明 §28 的验证问题：
  A. 零电荷 → 电势线性；
  B. 常电荷 → 电势二次；
  C. 分段介电常数 → φ 连续、D = eps*dφ/dz 连续（通量守恒）；
  以及 Dirichlet 边界条件、二阶收敛阶、MOS 几何冒烟测试。

可直接运行（无需 pytest）：
    python tests/test_poisson_fdm.py
也可用 pytest：
    pytest tests/test_poisson_fdm.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import constants, units
from src.device import Device1D
from src.poisson_fdm import harmonic_mean, solve_poisson, solve_poisson_fdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')

EPS_SI = 11.7 * constants.eps0
EPS_OX = 3.9 * constants.eps0
L = 100e-9  # 测试域长度 [m]


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


def _uniform(n, L=L):
    return np.linspace(0.0, L, n)


# ---------------------------------------------------------------------------
# 问题 A：零电荷 → 线性
# ---------------------------------------------------------------------------
def test_zero_charge_linear():
    """ρ=0 且 ε 均匀时，φ 应精确为两端点间的线性插值。"""
    n = 101
    z = _uniform(n)
    eps = np.full(n, EPS_SI)
    rho = np.zeros(n)
    phi = solve_poisson(rho, z, eps, phi_left=0.5, phi_right=0.0)
    exact = 0.5 * (1.0 - z / L)
    assert np.allclose(phi, exact, atol=1e-12)


# ---------------------------------------------------------------------------
# 问题 B：常电荷 → 二次
# ---------------------------------------------------------------------------
def test_const_charge_quadratic():
    """ρ=ρ0 且 ε 均匀时，φ 应精确为二次函数（中心差分离散对二次无截断误差）。"""
    n = 101
    z = _uniform(n)
    eps = np.full(n, EPS_SI)
    rho0 = 1.0e6  # [C/m^3]
    rho = np.full(n, rho0)
    V0, VL = 0.5, 0.0
    phi = solve_poisson(rho, z, eps, V0, VL)

    # 解析解 φ = V0 + C1 z - (rho0/(2 eps)) z^2
    C1 = (VL - V0 + (rho0 / (2.0 * EPS_SI)) * L ** 2) / L
    exact = V0 + C1 * z - (rho0 / (2.0 * EPS_SI)) * z ** 2
    assert np.allclose(phi, exact, rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# 问题 C：分段介电常数 → D 连续（通量守恒）
# ---------------------------------------------------------------------------
def _pw_linear_exact(z, a, eps1, eps2, V0):
    """分段线性解析解：φ(0)=V0, φ(L)=0, 界面 z=a 处 φ 连续、ε dφ/dz 连续。"""
    m1 = V0 / ((eps1 / eps2) * (a - L) - a)
    m2 = (eps1 / eps2) * m1
    return np.where(z < a, V0 + m1 * z, m2 * (z - L))


def test_piecewise_eps_flux_conserved():
    """分段 ε（Si/SiO2 界面）：离散通量 D = ε dφ/dz 应处处相等（守恒）。"""
    n = 401
    z = _uniform(n)
    a = L * 0.4  # 界面位置（介于节点之间）
    eps = np.where(z < a, EPS_OX, EPS_SI)
    rho = np.zeros(n)
    phi = solve_poisson(rho, z, eps, phi_left=0.5, phi_right=0.0)

    # 半网格离散通量：F[i] = eps_{i+1/2} * (phi[i+1]-phi[i]) / dz[i]
    eh = harmonic_mean(eps[:-1], eps[1:])
    dz = np.diff(z)
    flux = eh * np.diff(phi) / dz
    # 对 ρ=0，通量形式离散保证 F 严格为常数
    assert np.allclose(flux, flux[0], rtol=1e-9, atol=1e-12)


def test_piecewise_eps_analytic():
    """分段 ε 的数值解应收敛到分段线性解析解。"""
    n = 801
    z = _uniform(n)
    a = L * 0.4
    eps = np.where(z < a, EPS_OX, EPS_SI)
    rho = np.zeros(n)
    phi = solve_poisson(rho, z, eps, phi_left=0.5, phi_right=0.0)
    exact = _pw_linear_exact(z, a, EPS_OX, EPS_SI, 0.5)
    # 界面落在节点之间，局部 O(dz) 误差；细网格下应 < 1e-3
    assert np.max(np.abs(phi - exact)) < 1e-3


# ---------------------------------------------------------------------------
# Dirichlet 边界条件
# ---------------------------------------------------------------------------
def test_dirichlet_boundary():
    """两端点应精确满足 Dirichlet 条件。"""
    n = 51
    z = _uniform(n)
    eps = np.full(n, EPS_SI)
    rho = np.full(n, 1.0e6)
    phi = solve_poisson(rho, z, eps, phi_left=1.2, phi_right=-0.3)
    _assert_close(phi[0], 1.2)
    _assert_close(phi[-1], -0.3)


# ---------------------------------------------------------------------------
# 收敛阶：光滑问题应二阶收敛
# ---------------------------------------------------------------------------
def test_convergence_order():
    """光滑问题（精确解为 sin）应达到二阶收敛。"""
    exact = lambda z: np.sin(np.pi * z / L)
    rho_of = lambda z: EPS_SI * (np.pi / L) ** 2 * np.sin(np.pi * z / L)

    errs = []
    for n in [101, 201, 401, 801]:
        z = _uniform(n)
        eps = np.full(n, EPS_SI)
        phi = solve_poisson(rho_of(z), z, eps, 0.0, 0.0)
        errs.append(np.max(np.abs(phi - exact(z))))
    # 相邻网格的收敛阶 = log2(err[n] / err[2n])
    orders = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    mean_order = sum(orders) / len(orders)
    assert mean_order > 1.9, f"收敛阶不足：{orders}"


# ---------------------------------------------------------------------------
# MOS 几何冒烟测试
# ---------------------------------------------------------------------------
def test_mos_depletion_smoke():
    """真实 MOS 几何 + 耗尽电荷：φ 单调、BC 满足、界面处 φ 连续。"""
    dev = Device1D.from_yaml(CONFIG)
    # 耗尽近似：氧化层内 ρ=0，Si 内 ρ = -q*NA（全耗尽）
    rho = -constants.q * dev.NA
    V_G = 1.0
    phi = solve_poisson_fdm(dev, rho, phi_gate=V_G, phi_bulk=0.0)

    assert phi.shape == dev.z.shape
    _assert_close(phi[0], V_G)
    _assert_close(phi[-1], 0.0)
    # Si 内耗尽电荷为负 → 电势自栅极单调下降，无负值过冲
    assert np.all(np.diff(phi) <= 1e-12)
    assert np.all(phi >= 0.0)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------
def test_input_validation():
    """非法输入应抛出 ValueError。"""
    z = _uniform(5)
    eps = np.full(5, EPS_SI)
    rho = np.zeros(5)
    with _expect_raises(ValueError):
        solve_poisson(rho, z[:4], eps, 0.0, 0.0)  # 长度不一致
    with _expect_raises(ValueError):
        solve_poisson(rho, np.array([0, 2, 1, 3, 4]) * 1e-9, eps, 0.0, 0.0)  # 非递增
    with _expect_raises(ValueError):
        solve_poisson(rho, z, np.full(5, -1.0), 0.0, 0.0)  # eps 非正


class _expect_raises:
    """轻量 context manager：断言指定异常被抛出。"""

    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(f"期望抛出 {self.exc_type.__name__}，但未抛出")
        return issubclass(exc_type, self.exc_type)


if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
                passed += 1
            except Exception as e:
                print(f"FAIL  {name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
