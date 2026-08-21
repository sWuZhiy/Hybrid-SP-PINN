"""数值结果图绘制。

仅绘制来自真实计算数据的数值图（材料剖面、网格、求解结果等），
不包含器件结构示意图、物理机制示意图等概念图（由用户另行绘制，
见项目搭建说明 §46）。
"""

import numpy as np
import matplotlib.pyplot as plt

from .constants import m0

# 中文字体（Windows）；无中文字体时回退，避免绘图报错
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun',
                                   'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def mark_interface(ax, t_ox_nm):
    """在横轴 z [nm] 上标注 Si/SiO₂ 界面位置。"""
    ax.axvline(t_ox_nm, color='red', ls='--', lw=1.0,
               label=f'界面 z = {t_ox_nm:.3f} nm')


def plot_device_profiles(device, save_path=None):
    """绘制 Stage 2 的材料剖面数值图。

    面板：eps(z)、ΔEc(z)、各能谷 m_z(z)、网格间距 dz(z)。
    同时输出 PNG（save_path）与同名 PDF（论文用，矢量）。
    """
    z_nm = device.z * 1e9
    t_ox_nm = device.t_ox * 1e9

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # (1) 介电常数 eps(z)
    ax = axes[0, 0]
    ax.plot(z_nm, device.eps, lw=1.5, color='tab:blue')
    mark_interface(ax, t_ox_nm)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('ε [F/m]')
    ax.set_title('介电常数 ε(z)')
    ax.legend()

    # (2) 导带带阶 ΔEc(z)
    ax = axes[0, 1]
    ax.plot(z_nm, device.delta_ec_eV, lw=1.5, color='tab:green')
    mark_interface(ax, t_ox_nm)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('Δ$E_C$ [eV]')
    ax.set_title('导带带阶 Δ$E_C$(z)')
    ax.legend()

    # (3) z 向有效质量 m_z(z)（两组能谷）
    ax = axes[1, 0]
    for i, mz in enumerate(device.mass_z):
        mz_m0 = mz / m0
        m_si = device.params.m_z[i] / m0
        ax.plot(z_nm, mz_m0, lw=1.5,
                label=f'能谷组 {i + 1}（Si 内 $m_z$ = {m_si:.2f} $m_0$）')
    mark_interface(ax, t_ox_nm)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('$m_z$ [$m_0$]')
    ax.set_title('z 向有效质量 $m_z$(z)')
    ax.legend()

    # (4) 网格间距 dz(z)
    ax = axes[1, 1]
    dz = np.diff(device.z) * 1e9
    ax.plot(device.z[:-1] * 1e9, dz, lw=1.5, color='tab:purple')
    mark_interface(ax, t_ox_nm)
    ax.set_xlabel('z [nm]')
    ax.set_ylabel('dz [nm]')
    ax.set_title('网格间距 dz(z)')
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        pdf_path = save_path.rsplit('.', 1)[0] + '.pdf'
        fig.savefig(pdf_path, bbox_inches='tight')
    return fig
