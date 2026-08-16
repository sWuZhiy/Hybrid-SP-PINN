"""fermi_level 的单元测试（Stage 6）。

覆盖项目搭建说明 §31 的验证问题：
  1. Brent 求根与解析解 asinh 一致；
  2. 电中性残差：求根点 |Q| ≈ 0（p - n - NA = 0）；
  3. 单位与数量级：p 型 EF 为负，且 ≈ -0.406 eV（NA=1e17, T=300）；
  4. 载流子浓度自洽：p ≈ NA，n ≈ n_i^2/NA；
  5. Q(EF) 单调递减（根唯一）；
  6. 掺杂趋势：NA 越高，EF 越深（越负）；温度趋势：|EF| 随 T 增大；
  7. 输入校验。

可直接运行（无需 pytest）：
    python tests/test_fermi_level.py
也可用 pytest：
    pytest tests/test_fermi_level.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import constants, units
from src.fermi_level import (
    FermiResult,
    analytic_fermi_level,
    bulk_charge_density,
    carrier_densities,
    find_fermi_level,
)

# 默认参数（与 configs/default.yaml 一致）
N_I = units.cm3_to_m3(1.5e10)
NA = units.cm3_to_m3(1.0e17)
T = 300.0


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


# ---------------------------------------------------------------------------
# 1. 求根与解析解一致
# ---------------------------------------------------------------------------
def test_brent_matches_analytic():
    res = find_fermi_level(N_I, NA, T)
    assert isinstance(res, FermiResult)
    _assert_close(res.EF, res.EF_analytic, tol=1e-12)


# ---------------------------------------------------------------------------
# 2. 电中性残差
# ---------------------------------------------------------------------------
def test_neutrality_residual():
    res = find_fermi_level(N_I, NA, T)
    assert res.neutrality_error < 1e-12
    # 解析解满足 sinh(asinh x)=x，p - n - NA 应代数精确为 0
    Q_an = bulk_charge_density(res.EF_analytic, N_I, NA, T)
    assert abs(Q_an) / (constants.q * NA) < 1e-12


# ---------------------------------------------------------------------------
# 3. 单位与数量级（p 型 EF 为负，≈ -0.406 eV）
# ---------------------------------------------------------------------------
def test_fermi_level_magnitude():
    res = find_fermi_level(N_I, NA, T)
    EF_eV = res.EF * units.J_TO_EV
    assert EF_eV < 0.0                       # p 型：EF 低于本征能级
    # 手算参考：-kT asinh(NA/2n_i) ≈ -0.4062 eV（1% 容差，抓单位换算错误）
    assert abs(EF_eV + 0.4062) / 0.4062 < 0.01, f"EF = {EF_eV:.4f} eV"


# ---------------------------------------------------------------------------
# 4. 载流子浓度自洽
# ---------------------------------------------------------------------------
def test_carrier_densities_self_consistent():
    res = find_fermi_level(N_I, NA, T)
    # p ≈ NA（多数载流子），n ≈ n_i^2 / NA（少数载流子）
    assert math.isclose(res.p, NA, rel_tol=1e-10)
    assert math.isclose(res.n, N_I ** 2 / NA, rel_tol=1e-10)
    # n * p = n_i^2（平衡态质量作用定律）
    assert math.isclose(res.n * res.p, N_I ** 2, rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 5. Q(EF) 单调递减（根唯一）
# ---------------------------------------------------------------------------
def test_Q_monotonic():
    kT = constants.kB * T
    res = find_fermi_level(N_I, NA, T)
    EFs = res.EF + np.linspace(-10, 10, 21) * kT
    Qs = [bulk_charge_density(ef, N_I, NA, T) for ef in EFs]
    assert np.all(np.diff(Qs) < 0), "Q(EF) 应单调递减"


# ---------------------------------------------------------------------------
# 6. 掺杂 / 温度趋势
# ---------------------------------------------------------------------------
def test_doping_trend():
    """NA 越高，EF 越低于本征能级（越负）。"""
    EFs = [find_fermi_level(N_I, units.cm3_to_m3(na), T).EF
           for na in [1e15, 1e16, 1e17, 1e18]]
    assert EFs[0] > EFs[1] > EFs[2] > EFs[3]


def test_temperature_trend():
    """固定 n_i、NA 下，|EF| 随温度升高而增大（EF = -kT asinh，与 T 成正比）。"""
    EFs = [find_fermi_level(N_I, NA, tt).EF for tt in [150.0, 300.0, 450.0]]
    assert EFs[0] > EFs[1] > EFs[2]       # 更负
    assert all(ef < 0 for ef in EFs)


# ---------------------------------------------------------------------------
# 7. 解析解退化到非简并极限
# ---------------------------------------------------------------------------
def test_analytic_nondegenerate_limit():
    """NA >> n_i 时 asinh(NA/2n_i) ≈ ln(NA/n_i)。"""
    ef_exact = analytic_fermi_level(N_I, NA, T)
    ef_approx = -constants.kB * T * np.log(NA / N_I)
    # 相对误差 ~ 1/(2 ln(NA/n_i)) 量级，< 5%
    assert math.isclose(ef_exact, ef_approx, rel_tol=0.05)


# ---------------------------------------------------------------------------
# 8. 输入校验
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
    with _expect_raises(ValueError):
        find_fermi_level(N_I, 0.0, T)          # NA <= 0
    with _expect_raises(ValueError):
        find_fermi_level(0.0, NA, T)           # n_i <= 0
    with _expect_raises(ValueError):
        find_fermi_level(N_I, NA, 0.0)         # T <= 0
    with _expect_raises(ValueError):
        carrier_densities(0.0, N_I, 0.0)       # T <= 0


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
