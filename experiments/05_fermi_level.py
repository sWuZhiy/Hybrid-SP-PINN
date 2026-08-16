"""Stage 6 数值实验：bulk 费米能级 / 电中性。

内容（对应项目搭建说明 §31）：
  1. Q(EF) 随费米能级单调递减并穿越零点（求根区间证明）；
  2. 平衡载流子浓度 n、p 随 EF 的变化（对数坐标）；
  3. 费米能级随掺杂浓度的变化（p 型越重掺，EF 越低于 E_i）；
  并输出 EF、Q_total、neutrality_error。

运行：
    python experiments/05_fermi_level.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun',
                                   'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import constants, units
from src.fermi_level import (
    bulk_charge_density,
    carrier_densities,
    find_fermi_level,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

N_I = units.cm3_to_m3(1.5e10)
NA = units.cm3_to_m3(1.0e17)
T = 300.0


def fig_Q_vs_EF():
    """面板 1：Q(EF) 单调递减穿越零点。"""
    res = find_fermi_level(N_I, NA, T)
    EF_root = res.EF
    # 以根为中心，±0.15 eV 扫描
    EFs = EF_root + np.linspace(-0.15, 0.15, 401) * constants.q
    Qs = np.array([bulk_charge_density(ef, N_I, NA, T) for ef in EFs])
    Q_norm = Qs / (constants.q * NA)          # (p - n - NA)/NA

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(EFs * units.J_TO_EV, Q_norm, lw=1.4, label='Q(EF)/(q·NA)')
    ax.axhline(0.0, color='gray', ls='--', lw=0.8)
    ax.axvline(EF_root * units.J_TO_EV, color='tab:red', ls=':', lw=1.2,
               label=f'EF = {EF_root * units.J_TO_EV * 1e3:.2f} meV')
    ax.set_xlabel('EF [eV]（相对本征能级 E_i）')
    ax.set_ylabel('归一化电荷密度 Q/(q·NA)')
    ax.set_title('bulk 电中性：Q(EF) 单调递减、穿越零点')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_carriers_vs_EF():
    """面板 2：n、p 随 EF 变化（对数坐标）。"""
    res = find_fermi_level(N_I, NA, T)
    EFs = np.linspace(-0.6, 0.2, 401) * constants.q
    ns = np.array([carrier_densities(ef, N_I, T)[0] for ef in EFs])
    ps = np.array([carrier_densities(ef, N_I, T)[1] for ef in EFs])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogy(EFs * units.J_TO_EV, ns * units.M3_TO_CM3, lw=1.4, label='n')
    ax.semilogy(EFs * units.J_TO_EV, ps * units.M3_TO_CM3, lw=1.4, label='p')
    ax.axhline(NA * units.M3_TO_CM3, color='gray', ls=':', lw=0.8, label='NA')
    ax.axhline(N_I * units.M3_TO_CM3, color='tab:orange', ls=':', lw=0.8,
               label='n_i')
    ax.axvline(res.EF * units.J_TO_EV, color='tab:red', ls='--', lw=1.0,
               label=f'EF = {res.EF * units.J_TO_EV * 1e3:.0f} meV')
    ax.set_xlabel('EF [eV]（相对本征能级 E_i）')
    ax.set_ylabel('载流子浓度 [cm^-3]')
    ax.set_title('平衡载流子浓度 n、p 随费米能级的变化')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    return fig


def fig_EF_vs_NA():
    """面板 3：费米能级随掺杂浓度变化。"""
    NAs = np.logspace(15, 18, 40)             # cm^-3
    EFs = np.array([find_fermi_level(N_I, units.cm3_to_m3(na), T).EF
                    for na in NAs])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogx(NAs, EFs * units.J_TO_EV * 1e3, 'o-', ms=3)
    ax.axhline(0.0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('NA [cm^-3]')
    ax.set_ylabel('EF [meV]（相对本征能级 E_i）')
    ax.set_title('费米能级随掺杂浓度变化（p 型）')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()

    pd.DataFrame({'NA_cm3': NAs, 'EF_meV': EFs * units.J_TO_EV * 1e3}).to_csv(
        os.path.join(FIG_DIR, 'fermi_level_EF_vs_NA.csv'), index=False)
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    res = find_fermi_level(N_I, NA, T)
    print(f'EF          = {res.EF * units.J_TO_EV * 1e3:.4f} meV（相对 E_i）')
    print(f'EF_analytic = {res.EF_analytic * units.J_TO_EV * 1e3:.4f} meV')
    print(f'Q_total     = {res.Q_total:.3e} C/m^3')
    print(f'neutrality_error = {res.neutrality_error:.3e}')
    print(f'n = {res.n * units.M3_TO_CM3:.3e} cm^-3, '
          f'p = {res.p * units.M3_TO_CM3:.3e} cm^-3')

    jobs = [
        (fig_Q_vs_EF, 'fermi_level_Q_vs_EF.png'),
        (fig_carriers_vs_EF, 'fermi_level_carriers_vs_EF.png'),
        (fig_EF_vs_NA, 'fermi_level_EF_vs_NA.png'),
    ]
    for fn, name in jobs:
        fig = fn()
        p = os.path.join(FIG_DIR, name)
        fig.savefig(p, dpi=200, bbox_inches='tight')
        fig.savefig(p.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'图已保存：{p} (+ .pdf)')


if __name__ == '__main__':
    main()
