"""一维均匀网格生成。

网格覆盖整个 Poisson 求解域 [0, L_total]，其中 L_total = t_ox + L_si
（氧化层 + 硅区）。第一版本采用均匀网格（见项目搭建说明 §27.3），
Si/SiO₂ 界面 t_ox 可能落在相邻格点之间，由 Device1D 通过 i_interface
（第一个属于 Si 的格点索引）明确标注；界面的介电常数跳变在 Poisson
离散时以半网格通量形式处理（后续 Stage）。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """一维均匀网格。

    Attributes:
        z: 网格节点坐标 [m]，严格递增，覆盖 [0, L_total]。
        dz: 网格间距 [m]（均匀）。
        L_total: 仿真域总长度 [m]，= t_ox + L_si。
        t_ox: 氧化层厚度 [m]。
        i_interface: 第一个属于 Si 的格点索引（z >= t_ox）。
        n_grid: 总格点数。
    """
    z: np.ndarray
    dz: float
    L_total: float
    t_ox: float
    i_interface: int
    n_grid: int


def build_mesh(t_ox, L_si, n_grid):
    """构建一维均匀网格。

    Args:
        t_ox: 氧化层厚度 [m]。
        L_si: 硅区长度 [m]。
        n_grid: 总格点数（>= 2）。

    Returns:
        Mesh 对象。

    Raises:
        ValueError: n_grid < 2 或网格尺寸非法。
    """
    if n_grid < 2:
        raise ValueError("n_grid 必须 >= 2")
    if t_ox <= 0 or L_si <= 0:
        raise ValueError("t_ox 与 L_si 必须为正数")

    L_total = t_ox + L_si
    z = np.linspace(0.0, L_total, n_grid)
    dz = z[1] - z[0]
    i_interface = int(np.searchsorted(z, t_ox))
    return Mesh(z=z, dz=dz, L_total=L_total, t_ox=t_ox,
                i_interface=i_interface, n_grid=n_grid)
