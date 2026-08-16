"""Stage 7 数值实验：完整 FDM Schrödinger–Poisson 自洽求解基线。

内容（对应项目搭建说明 §31 / Stage 7）：
  1. 强反型自洽解剖面：φ(z)、导带底 Ec(z) 与亚带能级/波函数、n(z)/p(z)、ρ(z)；
  2. 亚带能级随 Vg 的演化（反型开启：能级下移穿过 EF）；
  3. 面密度 Ns(Vg) 与表面势 φ_surf(Vg)（弱反型 → 强反型）；
  4. 自洽迭代收敛历史（Anderson 混合）。

能量规范（方案 A）：全部相对 bulk 本征能级 E_i(bulk)=0。

运行：
    python experiments/06_sp_baseline.py
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
from src.device import Device1D, load_config
from src.fermi_level import find_fermi_level
from src.sp_solver import solve_sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, 'results', 'figures')

VGS = np.arange(0.0, 2.05, 0.1)              # Vg 扫描点 [V]
V_PROFILE = 1.5                              # 剖面图所用的强反型栅压 [V]
NS_THRESHOLD = 1.0e10                        # 反型开启判据 [cm^-2]


def run_sweep(device, config):
    """Vg 扫描（电压连续化：以上一 Vg 的收敛解作初值）。

    Returns:
        DataFrame（每行一个 Vg：phi_surf、Ns、各能谷 Ns、亚带能级、迭代数）。
    """
    params = device.params
    num_states = int(config['solver']['num_states'])
    rows = []
    phi0 = None
    for Vg in VGS:
        res = solve_sp(device, float(Vg), config, phi0=phi0)
        phi0 = res.phi
        # 束缚态能级（eV）：先二重能谷（m_z=m_l）后四重能谷（m_z=m_t）
        e2 = res.subband_energies[0] * units.J_TO_EV
        e4 = res.subband_energies[1] * units.J_TO_EV
        row = {
            'Vg_V': float(Vg),
            'phi_surf_V': res.phi[device.is_si][0],
            'Ns_total_cm2': res.Ns_total / 1e4,
            'Ns_2fold_cm2': float(np.sum(res.Ns_per_ladder[0])) / 1e4,
            'Ns_4fold_cm2': float(np.sum(res.Ns_per_ladder[1])) / 1e4,
            'E0_2fold_eV': e2[0] if e2.size else np.nan,
            'E1_2fold_eV': e2[1] if e2.size > 1 else np.nan,
            'E2_2fold_eV': e2[2] if e2.size > 2 else np.nan,
            'E0_4fold_eV': e4[0] if e4.size else np.nan,
            'E1_4fold_eV': e4[1] if e4.size > 1 else np.nan,
            'iters': res.iterations,
        }
        rows.append(row)
        print(f'  Vg={Vg:4.1f} V  收敛={res.converged}  迭代={res.iterations:3d}  '
              f'phi_surf={res.phi[device.is_si][0] * 1e3:7.1f} mV  '
              f'Ns={res.Ns_total / 1e4:.4g} cm^-2')
    return pd.DataFrame(rows)


def fig_profile(device, config, res):
    """面板图 1：强反型自洽解剖面（Vg = V_PROFILE）。"""
    z_nm = device.z * units.M_TO_NM
    t_ox_nm = device.t_ox * units.M_TO_NM
    is_si = device.is_si
    params = device.params

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8))

    # (a) 导带底 + 亚带能级 + 波函数（近界面放大，能量相对 E_i）
    # 氧化层导带底带阶 ΔEc=3.1 eV（求解时氧化层视为无限高势垒 ψ=0，
    # 此处只画物理导带位置；Si 内 Ec = E_g/2 - qφ 即求解所用势能）。
    ax = axes[0, 0]
    zmax = 30.0
    sel = z_nm <= zmax
    Ec_eV = res.Ec * units.J_TO_EV
    Ec_plot = np.where(device.is_si, Ec_eV,
                       Ec_eV + params.delta_Ec * units.J_TO_EV)
    ax.plot(z_nm[sel], Ec_plot[sel], lw=1.5, color='black', label='Ec(z)')
    ax.axhline(res.EF * units.J_TO_EV, color='tab:red', ls='--', lw=1.2,
               label=f'EF = {res.EF * units.J_TO_EV * 1e3:.0f} meV')
    # 波函数 |ψ|² 包络：按能级偏移、按组着色（二重能谷深色 / 四重能谷浅色）
    scale = 0.03 / max(
        [np.max(psi[:, :3] ** 2) if psi.shape[1] else 1.0
         for psi in res.subband_psi])
    for i_lad, (energies, psi, color) in enumerate(zip(
            res.subband_energies, res.subband_psi,
            ['tab:blue', 'tab:orange'])):
        for k in range(min(3, energies.size)):
            E_eV = energies[k] * units.J_TO_EV
            ax.axhline(E_eV, color=color, ls=':', lw=0.7)
            ax.plot(z_nm[sel], E_eV + scale * psi[sel, k] ** 2, lw=1.0,
                    color=color, alpha=0.85,
                    label=('能谷 m_z=%.2fm0, 态 %d' % (
                        params.m_z[i_lad] / constants.m0, k)) if k == 0 else None)
    ax.set_xlim(0, zmax)
    ax.set_ylim(-0.8, 0.7)
    ax.annotate('SiO₂ 导带底（ΔEc=3.1 eV，图中超出范围）',
                xy=(1.0, 0.65), xytext=(3.5, 0.45), fontsize=7,
                arrowprops=dict(arrowstyle='->', lw=0.8))
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('能量 [eV]（相对 E_i）')
    ax.set_title('导带底与亚带能级 / 波函数（近界面放大）')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.3)

    # (b) 静电势 φ(z) 全器件
    ax = axes[0, 1]
    ax.plot(z_nm, res.phi, lw=1.4, color='tab:green')
    ax.axvline(t_ox_nm, color='red', ls='--', lw=1.0, label='界面')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('φ [V]')
    ax.set_title(f'静电势（Vg = {res.Vg} V）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) 载流子浓度（对数）
    ax = axes[1, 0]
    n_cm3 = np.maximum(res.n * units.M3_TO_CM3, 1e-3)
    p_cm3 = np.maximum(res.p * units.M3_TO_CM3, 1e-3)
    ax.semilogy(z_nm, n_cm3, lw=1.4, color='tab:blue', label='n(z)（量子）')
    ax.semilogy(z_nm, p_cm3, lw=1.4, color='tab:orange', label='p(z)（经典）')
    ax.axhline(params.NA * units.M3_TO_CM3, color='gray', ls=':', lw=0.8,
               label='NA')
    ax.set_ylim(1e8, 1e21)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('浓度 [cm^-3]')
    ax.set_title('载流子浓度（对数坐标）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    # (d) 电荷密度
    ax = axes[1, 1]
    ax.plot(z_nm, res.rho, lw=1.2, color='tab:purple')
    ax.axvline(t_ox_nm, color='red', ls='--', lw=1.0, label='界面')
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('ρ [C/m^3]')
    ax.set_title('空间电荷密度')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_subbands_vs_vg(device, df):
    """面板图 2：亚带能级随 Vg 演化（反型开启）。"""
    params = device.params
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    EF_eV = find_fermi_level(params.n_i, params.NA, 300.0).EF * units.J_TO_EV
    ax.axhline(EF_eV, color='tab:red', ls='--', lw=1.2,
               label=f'EF = {EF_eV * 1e3:.0f} meV')
    ax.axhline(0.5 * params.E_g * units.J_TO_EV, color='gray', ls=':', lw=1.0,
               label='bulk 导带底 E_g/2')
    for col, color, label in [('E0_2fold_eV', 'tab:blue', '二重能谷 m_z=0.91'),
                              ('E1_2fold_eV', 'tab:blue', None),
                              ('E2_2fold_eV', 'tab:blue', None),
                              ('E0_4fold_eV', 'tab:orange', '四重能谷 m_z=0.19'),
                              ('E1_4fold_eV', 'tab:orange', None)]:
        ax.plot(df['Vg_V'], df[col], lw=1.2, color=color,
                label=label)
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('亚带能级 [eV]（相对 E_i）')
    ax.set_title('亚带能级随 Vg 演化：反型开启（穿过 EF）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_ns_and_phisurf(device, df):
    """面板图 3：Ns(Vg) 与 φ_surf(Vg)（弱反型 → 强反型）。"""
    params = device.params
    phi_F_eV = -find_fermi_level(params.n_i, params.NA, 300.0).EF \
        * units.J_TO_EV                     # φ_F = (E_i - EF)/q > 0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ax = axes[0]
    ax.semilogy(df['Vg_V'], np.maximum(df['Ns_total_cm2'], 1e-2), 'o-', ms=4,
                lw=1.3, label='总 Ns')
    ax.semilogy(df['Vg_V'], np.maximum(df['Ns_2fold_cm2'], 1e-2), 's--', ms=3,
                lw=1.0, label='二重能谷（m_z=m_l）')
    ax.semilogy(df['Vg_V'], np.maximum(df['Ns_4fold_cm2'], 1e-2), '^:', ms=3,
                lw=1.0, label='四重能谷（m_z=m_t）')
    ax.axhline(NS_THRESHOLD, color='gray', ls=':', lw=0.8,
               label=f'Ns = {NS_THRESHOLD:.0e} cm^-2')
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('Ns [cm^-2]')
    ax.set_title('电子面密度随 Vg（对数坐标）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ax.plot(df['Vg_V'], df['phi_surf_V'] * 1e3, 'o-', ms=4, lw=1.3,
            label='φ_surf（自洽解）')
    ax.axhline(phi_F_eV * 1e3, color='tab:green', ls=':', lw=1.0,
               label=f'φ_F = {phi_F_eV * 1e3:.0f} mV')
    ax.axhline(2 * phi_F_eV * 1e3, color='tab:red', ls='--', lw=1.0,
               label=f'2φ_F = {2 * phi_F_eV * 1e3:.0f} mV')
    ax.set_xlabel('Vg [V]')
    ax.set_ylabel('φ_surf [mV]')
    ax.set_title('表面势随 Vg：强反型区近似钉扎在 ~2φ_F')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def fig_convergence(device, config):
    """面板图 4：自洽迭代收敛历史（Anderson 混合，直接求解）。"""
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for Vg in [0.8, 1.2, 1.5, 2.0]:
        res = solve_sp(device, float(Vg), config)
        ax.semilogy(np.arange(1, len(res.history) + 1), res.history,
                    lw=1.2, label=f'Vg = {Vg} V（{res.iterations} 轮）')
    ax.set_xlabel('外层迭代轮数')
    ax.set_ylabel('max|G(φ) − φ| [V]')
    ax.set_title('自洽迭代收敛历史（Anderson 混合）')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    config = load_config(os.path.join(ROOT, 'configs', 'default.yaml'))
    device = Device1D(config)
    params = device.params

    res0 = find_fermi_level(params.n_i, params.NA, 300.0)
    print(f'EF    = {res0.EF * units.J_TO_EV * 1e3:.4f} meV（相对 E_i）')
    print(f'EF-Ec = {(res0.EF - 0.5 * params.E_g) * units.J_TO_EV * 1e3:.4f} meV'
          f'（相对 bulk 导带底）')
    print(f'φ_F   = {-res0.EF * units.J_TO_EV * 1e3:.1f} meV')
    print('Vg 扫描（电压连续化）：')
    df = run_sweep(device, config)

    # 反型开启：Ns 首次超过阈值时的 Vg
    above = df[df['Ns_total_cm2'] > NS_THRESHOLD]
    if not above.empty:
        print(f'反型开启（Ns > {NS_THRESHOLD:.0e} cm^-2）：Vg = '
              f'{above.iloc[0]["Vg_V"]:.2f} V')

    # 剖面图（Vg=V_PROFILE 的直接求解；Anderson 混合下无需电压连续化）
    res = solve_sp(device, V_PROFILE, config)
    print(f'\nVg = {V_PROFILE} V 自洽解：')
    print(f'  phi_surf = {res.phi[device.is_si][0] * 1e3:.1f} mV')
    print(f'  Ns_total = {res.Ns_total / 1e4:.4g} cm^-2'
          f'（二重 {np.sum(res.Ns_per_ladder[0]) / 1e4:.3g}，'
          f'四重 {np.sum(res.Ns_per_ladder[1]) / 1e4:.3g}）')
    for i, energies in enumerate(res.subband_energies):
        mz = params.m_z[i] / constants.m0
        print(f'  能谷组 m_z={mz:.2f}m0：束缚态 {energies.size} 个，'
              f'E0 = {energies[0] * units.J_TO_EV * 1e3:.1f} meV')
    print(f'  外层迭代 = {res.iterations} 轮（tol = '
          f'{config["solver"]["tol_V"]:.0e} V）')

    # 输出 CSV
    csv_path = os.path.join(FIG_DIR, 'sp_baseline_Vg_sweep.csv')
    df.to_csv(csv_path, index=False)
    print(f'\n数据已保存：{csv_path}')

    jobs = [
        (fig_profile(device, config, res), 'sp_baseline_profile.png'),
        (fig_subbands_vs_vg(device, df), 'sp_baseline_subbands_vs_Vg.png'),
        (fig_ns_and_phisurf(device, df), 'sp_baseline_Ns_phisurf_vs_Vg.png'),
        (fig_convergence(device, config), 'sp_baseline_convergence.png'),
    ]
    for fig, name in jobs:
        p = os.path.join(FIG_DIR, name)
        fig.savefig(p, dpi=200, bbox_inches='tight')
        fig.savefig(p.rsplit('.', 1)[0] + '.pdf', bbox_inches='tight')
        plt.close(fig)
        print(f'图已保存：{p} (+ .pdf)')


if __name__ == '__main__':
    main()
