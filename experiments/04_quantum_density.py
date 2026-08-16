"""Stage 5 数值实验：量子电子密度模块验证。

内容（对应项目搭建说明 §30）：
  1. 无限阱子带下 n(z) 随费米能级的变化（|ψ|² 形状加权）；
  2. 总面密度 Ns_total 随费米能级单调上升；
  3. 面密度随温度的单调上升；
  4. 二重/四重能谷组叠加（MOS 相关质量配置）；
  并报告 ∫ n dz 与 Ns_total 的守恒误差。

运行：
    python experiments/04_quantum_density.py
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
from src.schrodinger_fdm import solve_schrodinger
from src.quantum_density import quantum_density, quantum_density_multi, sheet_density

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

M_L = 0.91 * constants.m0
M_T = 0.19 * constants.m0
M_PAR_4 = np.sqrt(M_L * M_T)


def _inf_well(z, mass, L, num_states):
    return solve_schrodinger(z, np.full(z.size, mass), np.zeros(z.size),
                             num_states=num_states)


def fig_density_vs_EF():
    """面板 1：n(z) 随费米能级变化（二重能谷，无限阱）。"""
    L = 10e-9
    n_grid = 501
    z = np.linspace(0.0, L, n_grid)
    E, psi = _inf_well(z, M_L, L, num_states=5)
    z_nm = z * 1e9

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    EFs = np.array([0.02, 0.06, 0.12]) * constants.q
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for EF, c in zip(EFs, colors):
        n, _, Ns_total = quantum_density(E, psi, EF, 300.0, M_T, g_v=2)
        ax.plot(z_nm, n * units.M3_TO_CM3, color=c, lw=1.4,
                label=f'EF = {EF * units.J_TO_EV * 1e3:.0f} meV'
                      f'（Ns = {Ns_total * 1e-4:.2e} cm^-2）')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('n(z) [cm^-3]')
    ax.set_title('量子电子密度 n(z) 随费米能级变化（二重能谷，无限阱）')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_Ns_vs_EF():
    """面板 2：总面密度 Ns_total 随费米能级单调上升。"""
    L = 10e-9
    z = np.linspace(0.0, L, 501)
    E, psi = _inf_well(z, M_L, L, num_states=5)
    EFs = np.linspace(0.0, 0.2, 80) * constants.q
    Ns = [quantum_density(E, psi, EF, 300.0, M_T, g_v=2)[2] for EF in EFs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(EFs * units.J_TO_EV * 1e3, np.array(Ns) * 1e-16, 'o-', ms=3)
    for Ek in E * units.J_TO_EV * 1e3:
        ax.axvline(Ek, color='gray', ls=':', lw=0.7)
    ax.set_xlabel('EF [meV]')
    ax.set_ylabel('Ns_total [10^12 cm^-2]')
    ax.set_title('总面密度随费米能级的变化（虚线为子带能级）')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    pd.DataFrame({'EF_meV': EFs * units.J_TO_EV * 1e3,
                  'Ns_cm2': np.array(Ns) * 1e-4}).to_csv(
        os.path.join(FIG_DIR, 'quantum_density_Ns_vs_EF.csv'), index=False)
    return fig


def fig_Ns_vs_T():
    """面板 3：面密度随温度单调上升。"""
    E = np.array([0.0])
    EF = 0.02 * constants.q
    Ts = np.array([10, 30, 77, 150, 300, 500])
    Ns = [float(sheet_density(EF, E, T, M_T, g_v=2)[0]) for T in Ts]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogx(Ts, np.array(Ns) * 1e-16, 'o-')
    ax.set_xlabel('T [K]')
    ax.set_ylabel('Ns [10^12 cm^-2]')
    ax.set_title('面密度随温度的单调上升（EF 固定在 E1 上方 20 meV）')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    pd.DataFrame({'T_K': Ts, 'Ns_cm2': np.array(Ns) * 1e-4}).to_csv(
        os.path.join(FIG_DIR, 'quantum_density_Ns_vs_T.csv'), index=False)
    return fig


def fig_valley_groups():
    """面板 4：二重/四重能谷组叠加的密度分解。"""
    L = 10e-9
    z = np.linspace(0.0, L, 501)
    EF = 0.1 * constants.q
    T = 300.0
    z_nm = z * 1e9

    E1, psi1 = _inf_well(z, M_L, L, num_states=5)   # 二重：m_z=m_l
    E2, psi2 = _inf_well(z, M_T, L, num_states=5)   # 四重：m_z=m_t
    n1, _, Nst1 = quantum_density(E1, psi1, EF, T, M_T, g_v=2)
    n2, _, Nst2 = quantum_density(E2, psi2, EF, T, M_PAR_4, g_v=4)
    n, _, Nst = quantum_density_multi(
        [(E1, psi1, M_T, 2), (E2, psi2, M_PAR_4, 4)], EF, T)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(z_nm, n1 * units.M3_TO_CM3, lw=1.2, label='二重能谷 (g_v=2)')
    ax.plot(z_nm, n2 * units.M3_TO_CM3, lw=1.2, label='四重能谷 (g_v=4)')
    ax.plot(z_nm, n * units.M3_TO_CM3, 'k--', lw=1.6, label='叠加')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('n(z) [cm^-3]')
    ax.set_title(f'二重/四重能谷叠加（EF = {EF * units.J_TO_EV * 1e3:.0f} meV，'
                 f'Ns_total = {Nst * 1e-4:.2e} cm^-2）')
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # 守恒报告：∫ n dz vs Ns_total
    L = 10e-9
    z = np.linspace(0.0, L, 501)
    E, psi = _inf_well(z, M_L, L, num_states=5)
    n, Ns_i, Ns_total = quantum_density(E, psi, 0.1 * constants.q, 300.0,
                                        M_T, g_v=2)
    integral = np.trapezoid(n, z)
    rel_err = abs(integral - Ns_total) / Ns_total
    print(f'守恒检验：∫ n dz = {integral:.6e} m^-2，'
          f'Ns_total = {Ns_total:.6e} m^-2，相对误差 = {rel_err:.3e}')
    print(f'子带面密度 Ns_i [10^12 cm^-2]：'
          f'{np.round(Ns_i * 1e-16, 3).tolist()}')

    jobs = [
        (fig_density_vs_EF, 'quantum_density_vs_EF.png'),
        (fig_Ns_vs_EF, 'quantum_density_Ns_vs_EF.png'),
        (fig_Ns_vs_T, 'quantum_density_Ns_vs_T.png'),
        (fig_valley_groups, 'quantum_density_valley_groups.png'),
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
