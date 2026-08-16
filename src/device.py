"""一维 MOS 器件结构：网格 + 材料剖面 + 掺杂剖面。

Device1D 是贯穿后续各 Stage 的核心数据对象，其字段约定见
项目搭建说明 §27.4：z、region、is_oxide、is_si、eps、mass_z、delta_ec。
所有字段在内部统一为 SI 单位。
"""

import numpy as np
import yaml

from . import units
from .materials import (
    MaterialParams,
    delta_ec_profile,
    delta_ec_profile_eV,
    doping_profile,
    eps_profile,
    mass_z_profile,
    material_params_from_config,
    region_mask,
)
from .mesh import build_mesh


def load_config(path):
    """从 YAML 文件加载配置 dict。"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class Device1D:
    """一维 MOS 电容器件。

    组合均匀网格与材料参数，提供各 z 向材料剖面（单位均为 SI）。
    """

    def __init__(self, config):
        self.config = config
        g = config['geometry']
        self.t_ox = units.nm_to_m(g['t_ox_nm'])
        self.L_si = units.nm_to_m(g['L_si_nm'])

        self.mesh = build_mesh(self.t_ox, self.L_si, g['n_grid'])
        self.z = self.mesh.z
        self.params = material_params_from_config(config)

        # 区域掩码（§27.4）
        self.is_si, self.is_oxide = region_mask(self.z, self.t_ox)
        self.region = np.where(self.is_si, 'Si', 'SiO2')

        # 材料剖面（§27.4）
        self.eps = eps_profile(self.z, self.t_ox, self.params)                # [F/m]
        self.delta_ec = delta_ec_profile(self.z, self.t_ox, self.params)      # [J]
        self.delta_ec_eV = delta_ec_profile_eV(self.z, self.t_ox, self.params)  # [eV]
        self.mass_z = mass_z_profile(self.z, self.t_ox, self.params)          # [kg]，(n_ladders, n_grid)
        self.NA = doping_profile(self.z, self.t_ox, self.params)              # [1/m^3]

    @classmethod
    def from_yaml(cls, path):
        """从 YAML 配置路径构建 Device1D。"""
        return cls(load_config(path))
