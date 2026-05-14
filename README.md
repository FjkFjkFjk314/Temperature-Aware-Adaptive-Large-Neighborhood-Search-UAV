# Temperature-Aware Adaptive Large Neighborhood Search for UAV Scheduling (TA-ALNS)

This repository hosts the **companion data and experimental outputs** of the paper *"Temperature-Aware Adaptive Large Neighborhood Search for Multi-Platform UAV Scheduling under Low-Temperature Endurance Degradation"*. It contains the modified Solomon benchmark instances (adapted to a multi-platform UAV setting with time windows and payload constraints) and the raw outputs, aggregated tables, and visualizations from the four core experiment families: main comparison, ablations, operator contribution, Thompson Sampling adaptive selection, and robustness sweeps.

> Note: this is a **data / results release repository**. The TA-ALNS source implementation is not included here.

## Overview

In cold environments, battery efficiency drops sharply with ambient temperature, so the effective range of a UAV varies dynamically over the day. Conventional routing methods treat range as a fixed parameter, producing plans that pass nominal feasibility checks but fail under actual low-temperature execution. This study targets the **multi-platform UAV scheduling problem with time windows and payload constraints** and proposes **Temperature-Aware Adaptive Large Neighborhood Search (TA-ALNS)**, which:

- Converts the temperature-to-capacity relationship into a time-varying effective range and embeds this signal directly into the search-decision layer;
- Verifies endurance segment by segment at each leg's departure time, so that battery state participates in route feasibility evaluation rather than serving merely as a post-hoc correction;
- Introduces three new operators:
  - **Energy-Risk Guided Destroy**
  - **Temperature-Aware Insertion Repair**
  - **Temperature-Aware Platform Migration Repair**
- Adopts a **Thompson Sampling**-based adaptive operator selection scheme, which reduces across-run standard deviation by 46% and mean solution cost by 2.62% compared with the classical adaptive scheme.

## Key Results

Main comparison under the real Shenyang winter temperature trace (2024-01-01):

| Platform     | Paired cases | Wins over static baseline | Mean composite-cost improvement |
|--------------|:-----------:|:-------------------------:|:-------------------------------:|
| Single       | 12          | 12 / 12                   | **19.37%**                      |
| Multi        | 12          | 12 / 12                   | **10.15%**                      |
| All (incl. cosine controls) | 48 | **43 / 48**            | 11.22%                          |

See `数据表格提取_初稿.md` (journal-ready table drafts) and the **Data Organization** section below for full breakdowns.

Main takeaways:

1. Static-baseline solutions exhibit service-failure risks under real low-temperature execution; TA-ALNS actively circumvents high-risk long-distance legs.
2. **Embedding battery-aware information into search decisions** (rather than using it for post-hoc evaluation only) is the key driver of performance: dynamic evaluation alone yields about a 3% reduction, while the full framework achieves 4.5% (ablation study).
3. The benefit is robust across **two additional winter days** and **seven alternative battery-efficiency function families**; the V-shaped M7 model used in the headline comparison sits at the conservative end of the spectrum.
4. TA-ALNS is most effective in **single-platform, large-scale, wide-time-window** scenarios.

## Repository Layout

```
.
├── README.md                          # this file
├── LICENSE
├── 数据表格提取_初稿.md                # journal-ready table drafts (Tables 1–B2)
│
├── C101_25.txt  C101_100.txt          # modified Solomon benchmark instances
├── C102_50.txt  C201_25.txt           # columns: id  x  y  demand  ready  due  service
├── C201_100.txt C202_50.txt
├── R101_50.txt  R102_25.txt
├── R102_100.txt R201_50.txt
├── R202_25.txt  R202_100.txt
│
└── uav-alns-data/                     # all experimental outputs
    ├── 0刻度出发-出发温度时间不同v1/       # main comparison (M2 battery model)
    ├── 0刻度出发-出发温度时间不同v2换用m7模型/ # main comparison (M7 model, used in paper)
    ├── 强化学习改进V1/                    # learning-based operator selection baseline
    ├── ablation_per_flag/                # per-flag ablation
    ├── operator_contribution_v1/         # contribution of the three new operators
    ├── thompson_sampling_ablation/       # Thompson Sampling ablation
    ├── weight_sensitivity_v1/            # objective-weight sensitivity
    ├── battery_multiday_sweep/           # multi-day × multi-model robustness sweep
    ├── 动态6点优势图表_battery_multiday/   # aggregated tables / LaTeX for the sweep
    ├── model_eta_curves/                 # battery-efficiency function curves
    └── v2_final_validation/              # final validation round (V2)
```

## Data Organization

### 1. Solomon instances (repository root)

Twelve instances, named `{family}{id}_{size}.txt`:

- **C family** (clustered customers): `C101`, `C102`, `C201`, `C202`
- **R family** (random customers):    `R101`, `R102`, `R201`, `R202`
- Sizes: 25 / 50 / 100 customers

Each row is one node with columns: `node_id  x  y  demand  ready_time  due_time  service_time`.
Node 0 is the depot platform. In the multi-platform setting, two additional platforms at `(20, 30)` and `(57, 31)` are added on the algorithm side.

### 2. Main comparison (source of paper Table 2)

`uav-alns-data/0刻度出发-出发温度时间不同v2换用m7模型/` is the main directory used in the paper. It contains:

| File | Description |
|------|-------------|
| `record_fitness_{weather}_m7_{platform}_fixed_dispatch_0.mat` | full iteration record (MATLAB) |
| `record_fitness_{weather}_m7_{platform}_fixed_dispatch_0.md`  | convergence log and solution structure (human-readable) |
| `*_best_routes.md`                                            | best routes: node sequence, departure time, ambient temperature, per-leg battery state |
| `metrics_summary.csv`                                         | paired summary metrics (feeds Table 2) |
| `comparison_table.md`                                         | static vs. dynamic paired comparison |
| `all_parsed_results.json`                                     | parsed structured results |
| `航路图_png/`, `挑选的图片/`, `动态6点优势图表/`                 | visualization assets |

Filename conventions:

- `weather ∈ {real, cos}` — `real` is the minute-level smoothed Shenyang trace on 2024-01-01; `cos` is the controlled cosine synthetic weather
- `platform ∈ {single_platform, multi_platform}`
- The `static_` prefix denotes the static low-temperature baseline; no prefix denotes TA-ALNS dynamic planning

`0刻度出发-出发温度时间不同v1/` is the parallel run with the M2 battery model, using the same naming scheme.

### 3. Ablations and operator contribution

- `operator_contribution_v1/`: per-operator on/off results for the three new operators (destroy / insertion / platform migration), including
  `summary/overall_metrics_paper_core.csv`, `operator_usage_statistics_paper_core.csv`,
  the section-level drafts, and the figure-generation script `plot_ablation_charts.py`.
- `ablation_per_flag/raw.csv`: raw results decomposed by feature flags (e.g. `dynamic_eval`, `dynamic_repair`).
- `thompson_sampling_ablation/`: paired comparison between the classical roulette adaptive scheme and Thompson Sampling.

### 4. Robustness and sensitivity

- `battery_multiday_sweep/`:
  - `A_M7_multiday/` — sweep over two additional winter days (2024-01-03, 2024-01-04) with the M7 model fixed;
  - `B_model_sensitivity_jan01/` — sweep over seven alternative battery-efficiency function families with 2024-01-01 fixed.
- `动态6点优势图表_battery_multiday/`: CSV summaries and LaTeX tables (`latex_tables.txt`) for the above sweeps.
- `weight_sensitivity_v1/`: objective-weight sensitivity analysis (includes `sensitivity_curves.png`).
- `model_eta_curves/`: η(T) curves of the seven battery-efficiency families over −20 °C to 0 °C (PNG + PDF).

### 5. Learning-based selection baseline

- `强化学习改进V1/`: comparison against a learning-based operator-selection policy (real / cos × single / multi, 8 cases total).

## Field Cheatsheet

The filename pattern `record_fitness_{[static_]}{weather}_{model}_{platform}_fixed_dispatch_{idx}`:

| Segment | Values | Meaning |
|---------|--------|---------|
| `static_` | present / absent | static low-temperature baseline / TA-ALNS dynamic planning |
| `weather` | `real` / `cos`   | Shenyang real trace / cosine synthetic |
| `model`   | `m2` / `m7`      | battery-efficiency family (paper uses m7) |
| `platform`| `single_platform` / `multi_platform` | single / multi-platform |
| `fixed_dispatch_0` | — | fixed dispatch at 00:00 (reference times 06:00 / 08:00 / 14:00) |

Main experimental parameters (see Tables 1 / A1 in `数据表格提取_初稿.md` for the full list):

- Nominal range = 240, maximum payload = 200, flight speed = 1, service time = 10
- Fixed cost = 500, flight-use cost = 25, unit distance cost = 1
- ALNS: max iterations = 10000, stop threshold = 100, weight smoothing b = 0.8, weight reset N = 50, SA parameter a = 5
- 9 destroy operators + 7 repair operators; base random seed = 20260408; 5 repetitions per case
- M7 parameters: α_L = 0.008, β_L = 1.2, α_H = 0.004, β_H = 1.3, T_opt = 18 °C, lower bound = 0.3

## Reproducing the Tables

1. **Table 2 (main comparison):** read `uav-alns-data/0刻度出发-出发温度时间不同v2换用m7模型/metrics_summary.csv`.
2. **Table B2 (complete paired results):** see Section 8 of `数据表格提取_初稿.md`; raw files live in the same directory.
3. **Ablation figures:** run `uav-alns-data/operator_contribution_v1/summary/plot_ablation_charts.py`.
4. **Multi-day / multi-model robustness:** run `uav-alns-data/动态6点优势图表_battery_multiday/summarize_battery_multiday.py`.

If this data or the underlying method is useful for your research, please cite the paper. The citation block will be updated once the paper is officially published.

## License

Released under the license declared in [LICENSE](./LICENSE). The original Solomon benchmark instances are property of their original authors; this repository only redistributes the UAV-adapted variants together with our experimental outputs.
