"""quantum_density 的单元测试（Stage 5）。

覆盖项目搭建说明 §30.4 的验证问题：
  1. 非负性：n(z) >= 0，Ns_i >= 0；
  2. 守恒：∫ n(z) dz = Ns_total（对归一化波函数）；
  3. 单能级占据：一个子带时面密度的解析值；
  4. 简并度记账（§30.3）：自旋求和 2D DOS 约定，避免重复计数；
  5. 低温极限 / 高温趋势 / 费米能级单调趋势；
  6. 多能谷组叠加（二重 + 四重）；
  7. 输入校验。

可直接运行（无需 pytest）：
    python tests/test_quantum_density.py
也可用 pytest：
    pytest tests/test_quantum_density.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import constants
from src.schrodinger_fdm import solve_schrodinger
from src.quantum_density import (
    sheet_density,
    quantum_density,
    quantum_density_multi,
)

M_T = 0.19 * constants.m0
M_L = 0.91 * constants.m0
M_PAR_4 = np.sqrt(M_L * M_T)          # 四重能谷平面 DOS 质量


def _inf_well(z, mass, L, num_states=3):
    """无限深方势阱：返回 (energies, psi)。"""
    return solve_schrodinger(z, np.full(z.size, mass), np.zeros(z.size),
                             num_states=num_states)


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


# ---------------------------------------------------------------------------
# 1. 非负性
# ---------------------------------------------------------------------------
def test_nonnegativity():
    """n(z) 与 Ns_i 对费米能级在能级上、下方时都应非负。"""
    L = 10e-9
    z = np.linspace(0.0, L, 501)
    E, psi = _inf_well(z, M_T, L, num_states=4)
    for EF in [E[0] - 0.1 * constants.q, E[0] + 0.1 * constants.q,
               E[-1] + 0.3 * constants.q]:
        n, Ns_i, Ns_total = quantum_density(E, psi, EF, 300.0, M_T, g_v=2)
        assert np.all(n >= 0.0)
        assert np.all(Ns_i >= 0.0)
        assert Ns_total >= 0.0


# ---------------------------------------------------------------------------
# 2. 守恒：∫ n dz = Ns_total
# ---------------------------------------------------------------------------
def test_density_integral_equals_Ns_total():
    """对归一化波函数，∫ n(z) dz 应等于 Ns_total（相对误差报告并断言）。"""
    L = 10e-9
    n_grid = 501
    z = np.linspace(0.0, L, n_grid)
    E, psi = _inf_well(z, M_T, L, num_states=3)
    EF = E[0] + 0.05 * constants.q
    n, Ns_i, Ns_total = quantum_density(E, psi, EF, 300.0, M_T, g_v=2)

    integral = np.trapezoid(n, z)
    rel_err = abs(integral - Ns_total) / Ns_total
    # 归一化波函数下，误差仅来自梯形积分离散
    assert rel_err < 1e-9, f"∫n dz 与 Ns_total 相对误差 {rel_err:.3e} 超限"
    print(f"  ∫n dz = {integral:.6e}, Ns_total = {Ns_total:.6e}, "
          f"相对误差 = {rel_err:.3e}")


# ---------------------------------------------------------------------------
# 3. 单能级占据（解析值）
# ---------------------------------------------------------------------------
def test_single_subband_sheet_density():
    """单个子带的面密度应匹配解析式（含自旋+能谷简并）。"""
    E = np.array([0.0])
    EF = 0.1 * constants.q
    T = 300.0
    m_par = M_T
    g_v = 2
    g_s = 2
    kT = constants.kB * T
    x = (EF - E[0]) / kT
    expected = g_s * g_v * (m_par / (2.0 * np.pi * constants.hbar ** 2)) \
        * kT * np.log1p(np.exp(x))
    got = sheet_density(EF, E, T, m_par, g_v, g_s)[0]
    _assert_close(got, expected, tol=1e-12)


# ---------------------------------------------------------------------------
# 4. 简并度记账（§30.3）：自旋已含于 π 分母，避免重复计数
# ---------------------------------------------------------------------------
def test_spin_summed_convention():
    """代码采用自旋求和 2D DOS：Ns_i = g_v * m_par/(πħ²) * kT * ln(1+e^x)。

    这等价于显式 g_s*g_v 与单自旋 DOS m_par/(2πħ²) 的乘积，二者一致；
    若误把 g_s 再乘进 π 分母（如 g_s*g_v*m_par/(πħ²)）会多一个因子 2。
    """
    EF = 0.1 * constants.q
    E = np.array([0.0])
    T = 300.0
    m_par = M_T
    g_v = 4
    kT = constants.kB * T
    x = (EF - E[0]) / kT

    # 自旋求和约定
    expected = g_v * (m_par / (np.pi * constants.hbar ** 2)) * kT * np.log1p(np.exp(x))
    got = sheet_density(EF, E, T, m_par, g_v, g_s=2)[0]
    _assert_close(got, expected, tol=1e-12)

    # 显式自旋：g_s=2 时 = 2 × (g_s=1 单自旋结果)
    single = sheet_density(EF, E, T, m_par, g_v=1, g_s=1)[0]
    _assert_close(sheet_density(EF, E, T, m_par, g_v=1, g_s=2)[0], 2.0 * single,
                  tol=1e-12)


# ---------------------------------------------------------------------------
# 5. 低温极限 / 高温趋势 / 费米能级趋势
# ---------------------------------------------------------------------------
def test_zero_temperature_limit():
    """T -> 0 时，EF 上方子带面密度趋近 g_v*m_par/(πħ²)*(EF-E)。"""
    E = np.array([0.0])
    EF = 0.05 * constants.q
    T = 1.0                       # kT = 8.6e-5 eV << EF-E = 0.05 eV
    m_par = M_T
    g_v = 2
    expected = g_v * (m_par / (np.pi * constants.hbar ** 2)) * (EF - E[0])
    got = sheet_density(EF, E, T, m_par, g_v, g_s=2)[0]
    # kT/(EF-E) ~ 1.7e-3，低温极限应在该量级内吻合
    assert math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-15)


def test_temperature_trend():
    """固定 EF 下，面密度随温度升高而单调增大（有限温占据展宽）。"""
    E = np.array([0.0])
    EF = 0.02 * constants.q
    m_par = M_T
    Ns = [sheet_density(EF, E, T, m_par, g_v=2, g_s=2)[0]
          for T in [10.0, 77.0, 200.0, 300.0, 500.0]]
    assert np.all(np.diff(Ns) > 0), f"面密度应随温度单调递增：{Ns}"


def test_fermi_level_trend():
    """固定 T 下，面密度随费米能级升高而单调增大。"""
    E = np.array([0.0, 0.03 * constants.q])
    T = 300.0
    m_par = M_T
    EFs = np.linspace(-0.05, 0.2, 50) * constants.q
    Ns = [np.sum(sheet_density(EF, E, T, m_par, g_v=2, g_s=2)) for EF in EFs]
    assert np.all(np.diff(Ns) > 0), "总面密度应随 EF 单调递增"


# ---------------------------------------------------------------------------
# 6. 多能谷组叠加（二重 + 四重）
# ---------------------------------------------------------------------------
def test_multi_ladder_combination():
    """二重/四重能谷各自求解后叠加，应与单独计算之和一致。"""
    L = 10e-9
    z = np.linspace(0.0, L, 501)
    EF = 0.1 * constants.q
    T = 300.0

    # 二重能谷：m_z = m_l，m_par = m_t，g_v = 2
    E1, psi1 = _inf_well(z, M_L, L, num_states=3)
    # 四重能谷：m_z = m_t，m_par = sqrt(m_l*m_t)，g_v = 4
    E2, psi2 = _inf_well(z, M_T, L, num_states=3)

    n1, Ns1, Nst1 = quantum_density(E1, psi1, EF, T, M_T, g_v=2)
    n2, Ns2, Nst2 = quantum_density(E2, psi2, EF, T, M_PAR_4, g_v=4)

    n, Ns_per_ladder, Nst = quantum_density_multi(
        [(E1, psi1, M_T, 2), (E2, psi2, M_PAR_4, 4)], EF, T)

    assert np.allclose(n, n1 + n2)
    assert math.isclose(Nst, Nst1 + Nst2, rel_tol=1e-12)
    assert len(Ns_per_ladder) == 2
    assert np.allclose(Ns_per_ladder[0], Ns1)
    assert np.allclose(Ns_per_ladder[1], Ns2)


# ---------------------------------------------------------------------------
# 7. 输入校验
# ---------------------------------------------------------------------------
class _expect_raises:
    def __init__(self, exc_type):
        self.exc_type = exc_type

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, tb):
        if exc_type is None:
            raise AssertionError(f"期望抛出 {self.exc_type.__name__}，但未抛出")
        return issubclass(exc_type, self.exc_type)


def test_input_validation():
    E = np.array([0.0, 1e-20])
    z = np.linspace(0.0, 10e-9, 101)
    _, psi = _inf_well(z, M_T, 10e-9, num_states=2)

    with _expect_raises(ValueError):
        sheet_density(0.0, E, 0.0, M_T, g_v=2)              # T <= 0
    with _expect_raises(ValueError):
        sheet_density(0.0, E, 300.0, -M_T, g_v=2)           # m_par <= 0
    with _expect_raises(ValueError):
        sheet_density(0.0, E, 300.0, M_T, g_v=0)            # g_v <= 0
    with _expect_raises(ValueError):
        quantum_density(E, psi[:, :1], 0.0, 300.0, M_T, 2)  # psi 列数不匹配
    with _expect_raises(ValueError):
        quantum_density(np.zeros((2, 2)), psi, 0.0, 300.0, M_T, 2)  # energies 2D
    with _expect_raises(ValueError):
        quantum_density_multi([], 0.0, 300.0)                        # 空 ladders


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
