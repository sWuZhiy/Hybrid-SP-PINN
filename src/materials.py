"""材料参数与 z 向材料剖面。

所有材料参数在内部统一为 SI 单位（由 config 中的相对值换算），
绘图时再换算回 eV / nm / cm^-3。能谷简并按 (100) 硅表面处理：
6 个 Delta 能谷按 z 向（限域方向）有效质量分裂为两组——
  - 二重能谷（g_v=2）：m_z = m_l（纵向），m_par = m_t（横向 DOS）；
  - 四重能谷（g_v=4）：m_z = m_t（横向），m_par = sqrt(m_l*m_t)。
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from . import constants, units


@dataclass
class MaterialParams:
    """SI 单位的材料参数。"""
    eps_si: float          # 硅介电常数 [F/m]
    eps_ox: float          # 氧化层介电常数 [F/m]
    m_l: float             # 硅纵向有效质量 [kg]
    m_t: float             # 硅横向有效质量 [kg]
    m_ox: float            # 氧化层有效质量 [kg]
    delta_Ec: float        # Si/SiO2 导带带阶 [J]
    NA: float              # 受主掺杂浓度 [1/m^3]
    g_s: float             # 自旋简并度
    g_v: List[int]         # 能谷简并度 [二重, 四重]
    m_z: List[float]       # 各能谷组 z 向（限域）有效质量 [kg]
    m_par: List[float]     # 各能谷组平面 DOS 有效质量 [kg]
    n_ladders: int = field(init=False)  # 能谷组数

    def __post_init__(self):
        self.n_ladders = len(self.m_z)


def material_params_from_config(config):
    """从 config dict 构建 SI 单位的 MaterialParams。"""
    m = config['material']
    s = config['substrate']
    valley = m.get('valley', {})

    m_l = m['m_l'] * constants.m0
    m_t = m['m_t'] * constants.m0

    g_s = float(valley.get('g_s', 2))
    g_v = [int(x) for x in valley.get('g_v', [2, 4])]
    # z 向限域质量：二重能谷 m_z=m_l，四重能谷 m_z=m_t（(100) 表面）
    m_z = [m_l, m_t]
    # 平面 DOS 质量：二重能谷 m_par=m_t；四重能谷 m_par=sqrt(m_l*m_t)
    m_par = [m_t, (m_l * m_t) ** 0.5]

    return MaterialParams(
        eps_si=m['eps_si_r'] * constants.eps0,
        eps_ox=m['eps_ox_r'] * constants.eps0,
        m_l=m_l,
        m_t=m_t,
        m_ox=m['m_ox'] * constants.m0,
        delta_Ec=units.ev_to_joule(m['delta_Ec_eV']),
        NA=units.cm3_to_m3(float(s['NA_cm3'])),
        g_s=g_s,
        g_v=g_v,
        m_z=m_z,
        m_par=m_par,
    )


def region_mask(z, t_ox):
    """返回 (is_si, is_oxide) 布尔数组。节点 z >= t_ox 归入 Si。"""
    is_si = z >= t_ox
    is_oxide = ~is_si
    return is_si, is_oxide


def eps_profile(z, t_ox, p):
    """介电常数 eps(z) [F/m]，氧化层为 eps_ox，硅为 eps_si。"""
    is_si, _ = region_mask(z, t_ox)
    return np.where(is_si, p.eps_si, p.eps_ox)


def delta_ec_profile(z, t_ox, p):
    """导带带阶 ΔEc(z) [J]，氧化层内为 delta_Ec，硅内为 0。"""
    is_si, _ = region_mask(z, t_ox)
    return np.where(is_si, 0.0, p.delta_Ec)


def delta_ec_profile_eV(z, t_ox, p):
    """导带带阶 ΔEc(z) [eV]，用于绘图。"""
    return units.joule_to_ev(delta_ec_profile(z, t_ox, p))


def mass_z_profile(z, t_ox, p):
    """各能谷组的 z 向有效质量 m_z(z) [kg]。

    返回形状为 (n_ladders, n_grid) 的数组：氧化层内为 m_ox（第一版不在
    氧化层求解 Schrödinger，仅供完整性），硅内分别为 m_l 与 m_t。
    """
    is_si, _ = region_mask(z, t_ox)
    profiles = [np.where(is_si, mz, p.m_ox) for mz in p.m_z]
    return np.asarray(profiles)


def doping_profile(z, t_ox, p):
    """电离受主浓度 NA(z) [1/m^3]，氧化层内为 0（完全电离近似）。"""
    is_si, _ = region_mask(z, t_ox)
    return np.where(is_si, p.NA, 0.0)
