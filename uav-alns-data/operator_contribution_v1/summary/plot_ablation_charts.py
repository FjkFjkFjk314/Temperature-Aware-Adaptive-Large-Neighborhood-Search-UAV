import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from math import pi
import os

# ── 全局样式 ──
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ── 读取数据 ──
data_path = 'overall_metrics_paper_core.csv'
df = pd.read_csv(data_path)

df_plot = df[df['scenario'] == 'real_multi'].copy()
core_groups = ['static_baseline', 'dynamic_only', 'dynamic_plus_repair', 'dynamic_full_baseline']
df_plot = df_plot[df_plot['group'].isin(core_groups)]
df_plot['group'] = pd.Categorical(df_plot['group'], categories=core_groups, ordered=True)
df_plot = df_plot.sort_values('group').reset_index(drop=True)

group_labels = {
    'static_baseline':       'Static Baseline',
    'dynamic_only':          'Dynamic Evaluator Only',
    'dynamic_plus_repair':   'Dynamic + Temp-Repair',
    'dynamic_full_baseline': 'Full Temp-Aware ALNS',
}
df_plot['group_name'] = df_plot['group'].map(group_labels)

os.makedirs('figures', exist_ok=True)

# ==============================================================
# 1. 雷达图 (Radar Chart) — 改进版
# ==============================================================
# 按逻辑顺序：Cost → Distance → Thermal Risk → Compute Time → Best Fitness
metrics       = ['report_cost_3_mean', 'distance_mean', 'thermal_risk_mean', 'time_mean', 'fitness_best']
metric_labels = ['Total Cost', 'Total Distance', 'Thermal Risk', 'Compute Time (s)', 'Best Fitness']

# 记录每个维度的真实原始值区间，用于生成数值刻度标签
df_radar = df_plot[['group_name'] + metrics].copy()
raw_ranges = {}
for m in metrics:
    hi = df_radar[m].max()
    lo = df_radar[m].min()
    raw_ranges[m] = (lo, hi)
    df_radar[m + '_norm'] = (df_radar[m] - lo) / (hi - lo) if hi > lo else 0.5

N      = len(metrics)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig_radar, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
ax.set_facecolor('#f0f4f8')
fig_radar.patch.set_facecolor('white')
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# ── 轴标签（维度名）──
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metric_labels, size=12.5, fontweight='bold', color='#333333')

# ── 添加真实数值刻度（在第一根轴 / 0° 方向显示）──
tick_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
ax.set_rlabel_position(0)   # 刻度标签放在 0° 轴旁（即第一根轴旁边）
ax.set_yticks(tick_levels)

# 为第一个维度（report_cost_3_mean）生成真实刻度标注
lo0, hi0 = raw_ranges[metrics[0]]
tick_real_labels = [f'{lo0 + (hi0 - lo0) * t:.0f}' for t in tick_levels]
ax.set_yticklabels(tick_real_labels, size=8.5, color='#666666')
plt.ylim(0, 1.12)

# ── 极坐标网格美化 ──
ax.grid(color='#cccccc', linestyle='--', linewidth=0.6, alpha=0.8)
ax.spines['polar'].set_color('#cccccc')

# ── 各方法绘制 ──
palette    = ['#e63946', '#adb5bd', '#2a9d8f', '#1d3557']
alphas     = [0.9, 0.45, 0.9, 1.0]
linewidths = [1.8, 1.2, 1.8, 3.0]   # Full ALNS 用更粗的线
fill_alpha = [0.07, 0.04, 0.08, 0.18]  # Full ALNS 填充更明显

for i, (_, row) in enumerate(df_radar.iterrows()):
    vals = row[[m + '_norm' for m in metrics]].tolist() + [row[metrics[0] + '_norm']]
    ax.plot(angles, vals,
            linewidth=linewidths[i], linestyle='solid',
            label=row['group_name'], color=palette[i], alpha=alphas[i],
            zorder=4 + i)
    ax.fill(angles, vals, alpha=fill_alpha[i], color=palette[i], zorder=3 + i)

# Full ALNS 额外加一圈高亮描边，让深蓝色优势区域在视觉上更突出
full_row = df_radar[df_radar['group_name'] == 'Full Temp-Aware ALNS'].iloc[0]
full_vals = full_row[[m + '_norm' for m in metrics]].tolist() + [full_row[metrics[0] + '_norm']]
ax.plot(angles, full_vals, linewidth=5.5, linestyle='solid',
        color='#1d3557', alpha=0.15, zorder=7)  # 外发光效果

# ── 轴标签增加"Real Value Range"注脚 ──
ax.annotate('Tick values shown on 0° axis\nreflect Total Cost range;\nall axes normalized to [0, 1].',
            xy=(0.5, -0.12), xycoords='axes fraction',
            ha='center', fontsize=9, color='#888888', style='italic')

# ── 标题 ──
plt.title('Ablation Study — Multi-dimensional Performance Radar\n(Lower / Inner = Better Performance)',
          size=14, fontweight='bold', y=1.12, color='#1a1a2e')

# ── 图例放在图内左下角，避免占用右侧空间 ──
legend = ax.legend(loc='lower left', bbox_to_anchor=(-0.18, -0.18),
                   fontsize=10.5, framealpha=0.92,
                   edgecolor='#cccccc', handlelength=1.8)

plt.tight_layout()
plt.savefig('figures/radar_chart_ablation.png', dpi=300, bbox_inches='tight')
print('Radar chart saved to figures/radar_chart_ablation.png')


# ==============================================================
# 2. 帕累托前沿图 (Pareto Frontier Scatter Plot)
# ==============================================================
x_metric = 'thermal_risk_mean'
y_metric = 'report_cost_3_mean'

# ── 计算非支配点 ──
pts      = df_plot[[x_metric, y_metric]].values
is_pf    = np.ones(len(pts), dtype=bool)
for i, c in enumerate(pts):
    for j, o in enumerate(pts):
        if i == j:
            continue
        if (o[0] <= c[0] and o[1] <= c[1]) and (o[0] < c[0] or o[1] < c[1]):
            is_pf[i] = False
            break
df_plot['is_pareto'] = is_pf
pareto_df = df_plot[df_plot['is_pareto']].sort_values(by=x_metric)

# ── 样式字典 ──
sty_map = {
    'static_baseline':       {'color': '#e63946', 'marker': 'X', 'alpha': 1.00, 's': 260, 'zorder': 6, 'lw': 1.6},
    'dynamic_only':          {'color': '#adb5bd', 'marker': 's', 'alpha': 0.70, 's': 200, 'zorder': 4, 'lw': 1.2},
    'dynamic_plus_repair':   {'color': '#2a9d8f', 'marker': '^', 'alpha': 1.00, 's': 290, 'zorder': 7, 'lw': 1.6},
    'dynamic_full_baseline': {'color': '#1d3557', 'marker': 'o', 'alpha': 1.00, 's': 290, 'zorder': 7, 'lw': 1.6},
}

# Dynamic Evaluator Only 标注放到图标左侧，其余右侧
annot_cfg = {
    'static_baseline':       {'xytext': ( 12,  10), 'ha': 'left'},
    'dynamic_only':          {'xytext': (-12,   8), 'ha': 'right'},
    'dynamic_plus_repair':   {'xytext': ( 12,  10), 'ha': 'left'},
    'dynamic_full_baseline': {'xytext': ( 12, -18), 'ha': 'left'},
}

fig_pareto, ax2 = plt.subplots(figsize=(11, 7), facecolor='white')
ax2.set_facecolor('#f8f9fa')

# ── 散点 ──
for _, row in df_plot.iterrows():
    g   = row['group']
    sty = sty_map[g]
    ec  = '#333333' if sty['alpha'] == 1.0 else '#999999'

    ax2.scatter(row[x_metric], row[y_metric],
                s=sty['s'], c=[sty['color']], marker=sty['marker'],
                alpha=sty['alpha'], edgecolor=ec, linewidth=sty['lw'],
                label=group_labels[g], zorder=sty['zorder'])

    # 非支配点光晕
    if row['is_pareto']:
        ax2.scatter(row[x_metric], row[y_metric],
                    s=sty['s'] * 2.8, c=[sty['color']], marker=sty['marker'],
                    alpha=0.10, edgecolor='none', zorder=sty['zorder'] - 1)

    # 标注文字 (zorder=10，高于箭头 zorder=3)
    ac = annot_cfg[g]
    ax2.annotate(group_labels[g],
                 (row[x_metric], row[y_metric]),
                 xytext=ac['xytext'], textcoords='offset points',
                 fontsize=10.5, fontweight='bold', ha=ac['ha'],
                 alpha=sty['alpha'],
                 bbox=dict(boxstyle='round,pad=0.3', fc='white',
                           ec='#cccccc', alpha=0.88 * sty['alpha']),
                 zorder=10)

# ── 帕累托前沿线（仅非支配点）──
ax2.plot(pareto_df[x_metric], pareto_df[y_metric],
         linestyle='-', color='#6a0dad', linewidth=2.8,
         zorder=5, label='Pareto Front')

# 前沿标注
if len(pareto_df) > 1:
    mid_x = pareto_df[x_metric].mean()
    mid_y = pareto_df[y_metric].mean()
    ax2.annotate('Pareto Front', xy=(mid_x, mid_y),
                 xytext=(mid_x + 0.4, mid_y + 5),
                 color='#6a0dad', fontweight='bold', fontsize=11, ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#6a0dad', alpha=0.75),
                 zorder=11)

# ── 坐标轴 ──
ax2.set_xlabel('Thermal Risk Penalty  (Lower → Safer)',
               fontsize=13, fontweight='bold', labelpad=10)
ax2.set_ylabel('Total Comprehensive Cost  (Lower → Cheaper)',
               fontsize=13, fontweight='bold', labelpad=10)
ax2.set_title('Cost vs. Safety Pareto Front — Ablation Study\n'
              r'$\longleftarrow$ Lower Risk & Lower Cost is Better',
              fontsize=15, fontweight='bold', pad=14)
ax2.grid(True, linestyle='--', linewidth=0.6, alpha=0.6, color='#cccccc')
for spine in ax2.spines.values():
    spine.set_edgecolor('#cccccc')

# ── 灰色方向箭头（整体右移 10%，zorder=3 在文字/散点之下）──
min_x, max_x = ax2.get_xlim()
min_y, max_y = ax2.get_ylim()
rng_x = max_x - min_x
rng_y = max_y - min_y

ax2.annotate('Pareto Optimal\n(Safer & Cheaper)',
             xy    =(min_x + rng_x * 0.13, min_y + rng_y * 0.08),   # 箭头尖端（右移10%）
             xytext=(min_x + rng_x * 0.35, min_y + rng_y * 0.30),   # 文字位置（右移10%）
             ha='center', va='bottom',
             arrowprops=dict(facecolor='#aaaaaa', edgecolor='#aaaaaa',
                             shrink=0.04, width=2.2, headwidth=11),
             fontsize=12, fontweight='bold', color='#444488',
             zorder=3)   # ← 低于文字层(10)与散点层(4~7)，但高于网格(0)

# ── 图例 ──
legend = ax2.legend(loc='upper right', fontsize=10.5,
                    framealpha=0.92, edgecolor='#cccccc', handlelength=1.8)
for lh in legend.legend_handles:
    lh.set_alpha(1.0)

plt.tight_layout()
plt.savefig('figures/pareto_frontier_ablation.png', dpi=300, bbox_inches='tight')
print('Pareto plot saved to figures/pareto_frontier_ablation.png')
