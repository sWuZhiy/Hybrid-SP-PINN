"""mesh / materials / device 的单元测试（Stage 2）。

可直接运行（无需 pytest）：
    python tests/test_device.py
也可用 pytest：
    pytest tests/test_device.py
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import constants, units
from src.device import Device1D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, 'configs', 'default.yaml')


def _assert_close(actual, expected, tol=1e-12):
    assert math.isclose(actual, expected, rel_tol=tol, abs_tol=1e-15), \
        f"{actual} != {expected} (tol={tol})"


def _load_device():
    return Device1D.from_yaml(CONFIG)


def test_mesh_monotonic_and_coverage():
    """网格应严格递增并覆盖 [0, L_total]。"""
    dev = _load_device()
    z = dev.z
    assert z[0] == 0.0
    _assert_close(z[-1], dev.mesh.L_total)
    assert np.all(np.diff(z) > 0)
    assert len(z) == dev.mesh.n_grid


def test_interface_index():
    """i_interface 应为第一个 z >= t_ox 的格点索引。"""
    dev = _load_device()
    i = dev.mesh.i_interface
    assert dev.z[i] >= dev.t_ox
    assert dev.z[i - 1] < dev.t_ox


def test_region_mask():
    """is_si / is_oxide 应互斥且与 z >= t_ox 一致。"""
    dev = _load_device()
    assert np.all(dev.is_si == (dev.z >= dev.t_ox))
    assert np.all(dev.is_oxide == ~dev.is_si)
    assert not np.any(dev.is_si & dev.is_oxide)


def test_eps_profile():
    """eps：氧化层内为 eps_ox，硅内为 eps_si。"""
    dev = _load_device()
    assert np.allclose(dev.eps[dev.is_oxide], dev.params.eps_ox)
    assert np.allclose(dev.eps[dev.is_si], dev.params.eps_si)


def test_delta_ec_profile():
    """ΔEc：氧化层内为 delta_Ec，硅内为 0。"""
    dev = _load_device()
    assert np.allclose(dev.delta_ec[dev.is_oxide], dev.params.delta_Ec)
    assert np.allclose(dev.delta_ec[dev.is_si], 0.0)


def test_mass_z_profile():
    """mass_z：两组能谷，硅内分别为 m_l 与 m_t，氧化层内为 m_ox。"""
    dev = _load_device()
    assert dev.mass_z.shape == (dev.params.n_ladders, dev.mesh.n_grid)
    assert dev.params.n_ladders == 2
    assert np.allclose(dev.mass_z[0][dev.is_si], dev.params.m_l)
    assert np.allclose(dev.mass_z[1][dev.is_si], dev.params.m_t)
    assert np.allclose(dev.mass_z[:, dev.is_oxide], dev.params.m_ox)


def test_doping_profile():
    """掺杂：氧化层内为 0，硅内为 NA = 1e17 cm^-3 = 1e23 m^-3。"""
    dev = _load_device()
    assert np.allclose(dev.NA[dev.is_oxide], 0.0)
    assert np.allclose(dev.NA[dev.is_si], units.cm3_to_m3(1.0e17))


def test_valley_params():
    """能谷参数：g_v=[2,4]，m_z=[m_l,m_t]，m_par=[m_t,sqrt(m_l*m_t)]。"""
    dev = _load_device()
    p = dev.params
    assert p.g_v == [2, 4]
    assert np.allclose(p.m_z[0], p.m_l)
    assert np.allclose(p.m_z[1], p.m_t)
    assert np.allclose(p.m_par[0], p.m_t)
    assert np.allclose(p.m_par[1], (p.m_l * p.m_t) ** 0.5)


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
