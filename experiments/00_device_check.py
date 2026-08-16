"""Stage 2 数值图：材料剖面（eps、ΔEc、m_z、网格间距）。

仅绘制/导出来自 Device1D 真实计算数据的数值结果（图 + CSV），
不包含器件结构示意图等概念图（见项目搭建说明 §46）。

运行：
    python experiments/00_device_check.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constants import m0
from src.device import Device1D
from src.plotting import plot_device_profiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    dev = Device1D.from_yaml(os.path.join(ROOT, 'configs', 'default.yaml'))

    fig_dir = os.path.join(ROOT, 'results', 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # 导出剖面原始数据（CSV），便于复现与论文核对
    df = pd.DataFrame({
        'z_nm': dev.z * 1e9,
        'is_si': dev.is_si.astype(int),
        'eps_Fm': dev.eps,
        'delta_Ec_eV': dev.delta_ec_eV,
        'm_z_ladder1_m0': dev.mass_z[0] / m0,
        'm_z_ladder2_m0': dev.mass_z[1] / m0,
        'NA_m3': dev.NA,
    })
    csv_path = os.path.join(fig_dir, 'device_profiles.csv')
    df.to_csv(csv_path, index=False)

    # 生成数值图（PNG + PDF）
    fig_path = os.path.join(fig_dir, 'device_material_profiles.png')
    plot_device_profiles(dev, save_path=fig_path)

    i = dev.mesh.i_interface
    print(f'数据已保存：{csv_path}')
    print(f'图已保存：  {fig_path}  (+ 同名 .pdf)')
    print(f'网格：n_grid={dev.mesh.n_grid}, dz={dev.mesh.dz * 1e9:.4f} nm, '
          f'L_total={dev.mesh.L_total * 1e9:.3f} nm')
    print(f'界面：i_interface={i}, z[{i}]={dev.z[i] * 1e9:.4f} nm, '
          f'z[{i - 1}]={dev.z[i - 1] * 1e9:.4f} nm (t_ox={dev.t_ox * 1e9:.3f} nm)')


if __name__ == '__main__':
    main()
