"""units.py 与 constants.py 的单元测试。

可直接运行（无需 pytest）：
    python tests/test_units.py
也可用 pytest：
    pytest tests/test_units.py
"""

import math
import os
import sys

# 使测试能直接以 `python tests/test_units.py` 方式运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants, units


def _assert_close(actual, expected, tol=1e-12):
    # 使用相对+绝对双重容差，兼顾大数（如 1e17）与小数（如 1e-9）
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


def test_ev_joule_roundtrip():
    """eV ↔ J 换算应互为逆运算。"""
    _assert_close(units.ev_to_joule(1.0), constants.q)
    _assert_close(units.joule_to_ev(constants.q), 1.0)
    for x in (0.0, 0.5, 3.1, 100.0):
        _assert_close(units.joule_to_ev(units.ev_to_joule(x)), x)


def test_nm_m_roundtrip():
    """nm ↔ m 换算应互为逆运算。"""
    _assert_close(units.nm_to_m(1.0), 1e-9)
    _assert_close(units.m_to_nm(1e-9), 1.0)
    for x in (0.0, 2.0, 100.0):
        _assert_close(units.m_to_nm(units.nm_to_m(x)), x)


def test_cm3_m3_roundtrip():
    """cm^-3 ↔ m^-3 换算应互为逆运算。"""
    _assert_close(units.cm3_to_m3(1.0), 1e6)
    _assert_close(units.m3_to_cm3(1e6), 1.0)
    for x in (0.0, 1.0e17, 1.0e20):
        _assert_close(units.m3_to_cm3(units.cm3_to_m3(x)), x)


def test_thermal_voltage():
    """kT/q 在 300 K 下应约为 0.02585 V（物理量级 sanity check）。"""
    vt = constants.kB * 300.0 / constants.q
    _assert_close(vt, 0.02585, tol=1e-3)


def test_constants_positive():
    """所有基本常数应为正数。"""
    for name in ("q", "hbar", "m0", "kB", "eps0"):
        assert getattr(constants, name) > 0, name


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
