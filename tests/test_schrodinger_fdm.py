"""schrodinger_fdm 的单元测试（Stage 4）。

覆盖项目搭建说明 §29 的验证问题：
  1. 无限深方势阱（解析能级 + 波函数）；
  2. 有限深方势阱（束缚态 + 深阱极限）；
  3. 三角势阱（Airy 函数能级）；
  以及归一化、正交性、波函数节点数、能级排序、输入校验。

可直接运行（无需 pytest）：
    python tests/test_schrodinger_fdm.py
也可用 pytest：
    pytest tests/test_schrodinger_fdm.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import brentq
from scipy.special import ai_zeros

from src import constants
from src.schrodinger_fdm import build_hamiltonian, solve_schrodinger

# (100) 硅横向有效质量，测试用
M_T = 0.19 * constants.m0

# Airy 函数 Ai(-x)=0 的正根（升序），用于三角势阱解析能级
_AIRY_NEG_ZEROS = ai_zeros(20)[0]          # Ai(x)=0 的（负）零点
AIRY_ROOTS = -_AIRY_NEG_ZEROS              # 正根 a_n


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


# ---------------------------------------------------------------------------
# 1. 无限深方势阱
# ---------------------------------------------------------------------------
def _inf_well_energy(n, m, L):
    """E_n = n^2 pi^2 hbar^2 / (2 m L^2)。"""
    return n ** 2 * np.pi ** 2 * constants.hbar ** 2 / (2.0 * m * L ** 2)


def _inf_well_wf(z, n, L):
    """psi_n = sqrt(2/L) sin(n pi z / L)。"""
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * z / L)


def _finite_well_energies(V0, L, m):
    """对称有限深方势阱束缚态能级（超越方程解析参考，从阱底起算，单位 J）。

    常数质量下 BenDaniel-Duke 退化为 psi、psi' 连续，束缚态满足：
      偶数态：tan(u) = sqrt(u0^2 - u^2)/u
      奇数态：tan(u) = -u/sqrt(u0^2 - u^2)
    其中 u = (L/2) sqrt(2mE)/hbar，u0 = (L/2) sqrt(2m V0)/hbar，E = (2 hbar^2/(m L^2)) u^2。
    """
    hbar = constants.hbar
    u0 = (L / 2.0) * np.sqrt(2.0 * m * V0) / hbar
    c = 2.0 * hbar ** 2 / (m * L ** 2)

    def even_f(u):
        return np.tan(u) - np.sqrt(u0 ** 2 - u ** 2) / u

    def odd_f(u):
        return np.tan(u) + u / np.sqrt(u0 ** 2 - u ** 2)

    roots = []
    for n in range(int(np.ceil(2.0 * u0 / np.pi)) + 1):
        lo, hi = n * np.pi / 2.0, min((n + 1) * np.pi / 2.0, u0)
        if hi <= lo:
            continue
        f = even_f if (n % 2 == 0) else odd_f
        eps = 1e-9
        a, b = lo + eps, hi - eps
        if a >= b:
            continue
        if f(a) * f(b) < 0:
            roots.append(c * brentq(f, a, b) ** 2)
    return np.array(sorted(roots))


def test_infinite_well_energies():
    """前 5 个能级应匹配解析解（细网格下 ~1e-4 相对误差）。"""
    L = 10e-9
    n = 501
    z = np.linspace(0.0, L, n)
    E, _ = solve_schrodinger(z, np.full(n, M_T), np.zeros(n), num_states=5)
    for k in range(5):
        exact = _inf_well_energy(k + 1, M_T, L)
        assert math.isclose(E[k], exact, rel_tol=1e-3, abs_tol=1e-30), \
            f"E[{k}] = {E[k]} != {exact}"


def test_infinite_well_wavefunctions():
    """波函数应匹配解析解（符号对齐后）。"""
    L = 10e-9
    n = 501
    z = np.linspace(0.0, L, n)
    E, psi = solve_schrodinger(z, np.full(n, M_T), np.zeros(n), num_states=3)
    for k in range(3):
        ana = _inf_well_wf(z, k + 1, L)
        if np.dot(psi[:, k], ana) < 0:
            ana = -ana
        # 波函数逐点最大误差 < 1%
        assert np.max(np.abs(psi[:, k] - ana)) < 0.01 * np.max(np.abs(ana))


# ---------------------------------------------------------------------------
# 2. 归一化与正交性
# ---------------------------------------------------------------------------
def test_normalization():
    """每个态 ∫ psi^2 dz = 1。"""
    L = 10e-9
    n = 401
    z = np.linspace(0.0, L, n)
    _, psi = solve_schrodinger(z, np.full(n, M_T), np.zeros(n), num_states=5)
    for k in range(5):
        norm = np.trapezoid(psi[:, k] ** 2, z)
        assert math.isclose(norm, 1.0, rel_tol=1e-10, abs_tol=1e-12)


def test_orthogonality():
    """不同态 ⟨psi_i, psi_j⟩ = 0。"""
    L = 10e-9
    n = 401
    z = np.linspace(0.0, L, n)
    _, psi = solve_schrodinger(z, np.full(n, M_T), np.zeros(n), num_states=5)
    for i in range(5):
        for j in range(i + 1, 5):
            inner = np.trapezoid(psi[:, i] * psi[:, j], z)
            assert abs(inner) < 1e-8, f"<psi_{i},psi_{j}> = {inner}"


# ---------------------------------------------------------------------------
# 3. 波函数节点数与能级排序
# ---------------------------------------------------------------------------
def test_node_count_and_ordering():
    """第 n 个态应有 n-1 个内部节点；能级严格升序。"""
    L = 10e-9
    n = 501
    z = np.linspace(0.0, L, n)
    E, psi = solve_schrodinger(z, np.full(n, M_T), np.zeros(n), num_states=5)
    assert np.all(np.diff(E) > 0)
    for k in range(5):
        # 内部节点 = 相邻格点符号变化次数（忽略数值零）
        s = np.sign(psi[1:-1, k])
        s = s[s != 0]
        nodes = int(np.sum(s[1:] != s[:-1]))
        assert nodes == k, f"态 {k + 1} 的节点数为 {nodes}，应为 {k}"


# ---------------------------------------------------------------------------
# 4. 有限深方势阱
# ---------------------------------------------------------------------------
def test_finite_well_bound_states():
    """束缚态能级位于 (-V0, 0)；阱变深（阱底降低）→ 能级降低（§29.4）。"""
    L_dom, L_w, n = 40e-9, 5e-9, 801
    z = np.linspace(0.0, L_dom, n)
    xc = L_dom / 2.0

    Es = {}
    for V0_eV in [0.3, 0.6, 1.0]:
        V0 = V0_eV * constants.q
        # 壁垒参考为 0，阱底 = -V0；阱越深 → 相对固定壁垒的能级越低
        Ec = np.where(np.abs(z - xc) < L_w / 2.0, -V0, 0.0)
        E, _ = solve_schrodinger(z, np.full(n, M_T), Ec, num_states=8)
        bound = E[E < 0.0]            # 束缚态：能级低于壁垒（0）
        assert len(bound) >= 1, f"V0={V0_eV} eV 应有束缚态"
        assert np.all(bound > -V0)
        Es[V0_eV] = bound
    # 阱变深 → 基态能级降低（§29.4「能级随势阱变深而降低」）
    assert Es[1.0][0] < Es[0.6][0] < Es[0.3][0]


def test_finite_well_transcendental():
    """有限阱束缚态能级应匹配超越方程解析解（含波函数穿透效应）。"""
    L_dom, L_w, n = 40e-9, 5e-9, 801
    z = np.linspace(0.0, L_dom, n)
    xc = L_dom / 2.0
    V0 = 0.5 * constants.q             # 0.5 eV 深阱
    Ec = np.where(np.abs(z - xc) < L_w / 2.0, 0.0, V0)
    E, _ = solve_schrodinger(z, np.full(n, M_T), Ec, num_states=6)
    exact = _finite_well_energies(V0, L_w, M_T)
    for k in range(min(6, len(exact))):
        # 势阱边缘的势能阶跃落在节点之间被抹平，带来 O(dz) 一阶离散误差
        # （无限阱 / 三角阱无内部间断，可达二阶/精确匹配），故此处用 ~2% 容差。
        assert math.isclose(E[k], exact[k], rel_tol=2e-2, abs_tol=1e-30), \
            f"E[{k}] = {E[k]} != {exact[k]}"


# ---------------------------------------------------------------------------
# 5. 三角势阱（Airy 函数）
# ---------------------------------------------------------------------------
def test_triangular_well_energies():
    """能级应匹配 Airy 函数解析解 E_n = a_n (hbar^2 q^2 F^2 / 2m)^{1/3}。"""
    L = 50e-9
    n = 801
    z = np.linspace(0.0, L, n)
    F = 1.0e7                          # 电场 [V/m]
    Ec = constants.q * F * z           # V(z) = q F z
    E, _ = solve_schrodinger(z, np.full(n, M_T), Ec, num_states=5)
    scale = (constants.hbar ** 2 * (constants.q * F) ** 2 / (2.0 * M_T)) ** (1.0 / 3.0)
    for k in range(5):
        exact = AIRY_ROOTS[k] * scale
        assert math.isclose(E[k], exact, rel_tol=1e-2, abs_tol=1e-30), \
            f"E[{k}] = {E[k]} != {exact}"


# ---------------------------------------------------------------------------
# 6. 哈密顿矩阵对称性
# ---------------------------------------------------------------------------
def test_hamiltonian_symmetric():
    """H 应为实对称矩阵。"""
    L = 10e-9
    n = 51
    z = np.linspace(0.0, L, n)
    H = build_hamiltonian(z, np.full(n, M_T), np.zeros(n))
    assert H.shape == (n - 2, n - 2)
    assert np.allclose(H, H.T)


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
    z = np.linspace(0.0, 10e-9, 21)
    m = np.full(21, M_T)
    Ec = np.zeros(21)
    with _expect_raises(ValueError):
        solve_schrodinger(z, m, Ec, 0)                      # num_states < 1
    with _expect_raises(ValueError):
        solve_schrodinger(z[:20], m, Ec, 3)                 # 长度不一致
    with _expect_raises(ValueError):
        solve_schrodinger(z, np.full(21, -M_T), Ec, 3)      # mass 非正
    bad_z = np.linspace(0.0, 10e-9, 21)
    bad_z[5] = bad_z[4]                                     # 非严格递增
    with _expect_raises(ValueError):
        solve_schrodinger(bad_z, m, Ec, 3)


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
