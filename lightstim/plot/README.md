# Plot Module

Configurable plotting for QEC simulation results (LER, decoding time, etc.).

## Quick Start

```python
from src.plot import plot_ler_vs_p

# One-liner: LER vs physical error rate, colored by distance
plot_ler_vs_p(df, x_col="p", hue="d")
```

## Input

Pandas DataFrame from `SimulationPipeline.run_batch()` with columns: `shots`, `post_selected_shots`, `errors`, `logical_error_rate`, plus metadata (`d`, `p`, `p1`, `decoder`, etc.).

## API

| Function | Description |
|----------|-------------|
| `plot_ler_vs_p(df, ...)` | LER vs physical error rate |
| `plot_ler_vs_distance(df, ...)` | LER vs code distance |
| `plot_simulation_results(df, x, y, hue, ...)` | Generic x vs y |
| `plot_custom(df, PlotConfig(...))` | Full config-driven plot |

## Config

`PlotConfig` maps DataFrame columns to axes and styling:
- `x`, `y`, `hue` – column names
- `facet_col`, `facet_row` – subplots
- `x_scale`, `y_scale` – `"log"` or `"linear"`
- `palette` – `"distance"` or `Dict` for hue colors

## Layout

```
src/plot/
├── config.py   # PlotConfig dataclass
├── styles.py   # Palettes, theme
├── utils.py    # Error bars, sanitize
└── plotter.py  # Main plotting functions
```
