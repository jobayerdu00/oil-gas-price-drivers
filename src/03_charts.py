"""Step 3: charts for the README (PNG, saved to outputs/figures/)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
R = json.load(open(ROOT / "outputs" / "results.json"))

# palette (validated categorical slots 1-3, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 10.5, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 24, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "lines.linewidth": 2, "figure.dpi": 150,
})


def finish(fig, ax, name, subtitle=None, source="Source: FRED (GASREGW, MCOILWTICO), EIA."):
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, color=INK2, fontsize=10, va="bottom")
    fig.text(0.01, 0.01, source, color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


all_m = pd.read_csv(ROOT / "data/processed/monthly_all.csv", parse_dates=["month"], index_col="month")
d = pd.read_csv(ROOT / "data/processed/monthly_model_features.csv", parse_dates=["month"], index_col="month")

# 1. gasoline vs crude, same unit -----------------------------------------
m = all_m[["gasoline_usd_gal", "wti_usd_bbl"]].dropna().loc["1991":]
crude_gal = m.wti_usd_bbl / 42
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.fill_between(m.index, crude_gal, m.gasoline_usd_gal, color=BLUE, alpha=0.10, linewidth=0)
ax.plot(m.index, m.gasoline_usd_gal, color=BLUE)
ax.plot(m.index, crude_gal, color=ORANGE)
ax.text(m.index[-1], m.gasoline_usd_gal.iloc[-1], "  Retail gasoline", color=INK, va="center", fontsize=10)
ax.text(m.index[-1], crude_gal.iloc[-1], "  Crude oil (WTI)", color=INK, va="center", fontsize=10)
ax.text(pd.Timestamp("1992-01-01"), 2.3, "Shaded gap = refining, distribution,\ntaxes and retail margin",
        color=INK2, fontsize=9, ha="left")
ax.set_ylabel("US$ per gallon")
ax.set_xlim(m.index[0], m.index[-1] + pd.Timedelta(days=1500))
ax.set_title("Gasoline tracks crude oil closely")
finish(fig, ax, "01_gas_vs_crude.png", "Monthly average, 1991 to 2026. Crude converted to $/gallon (1 barrel = 42 gallons).")

# 2. scatter with fit -------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
pre = d.index < "2020-01-01"
ax.scatter(d.crude_usd_gal[pre], d.gasoline_usd_gal[pre], s=22, color=BLUE, alpha=0.75,
           edgecolor=SURFACE, linewidth=0.8, label="2005 to 2019")
ax.scatter(d.crude_usd_gal[~pre], d.gasoline_usd_gal[~pre], s=22, color=ORANGE, alpha=0.85,
           edgecolor=SURFACE, linewidth=0.8, label="2020 to 2026")
xs = np.linspace(d.crude_usd_gal.min(), d.crude_usd_gal.max(), 50)
a, b = R["levels_model"]["crude_only_intercept_usd_gal"], R["levels_model"]["crude_only_slope"]
ax.plot(xs, a + b * xs, color=INK2, linewidth=1.5, linestyle="--")
ax.text(xs[-1], 1.75, f"Fit: gas = ${a:.2f} + {b:.2f} × crude\nR² = {R['levels_model']['r2_crude_only']:.2f}",
        ha="right", color=INK2, fontsize=9)
ax.set_xlabel("Crude oil, US$ per gallon")
ax.set_ylabel("Retail gasoline, US$ per gallon")
ax.legend(frameon=False, loc="upper left")
ax.set_title("Since 2020, gasoline sits higher for the same crude price")
finish(fig, ax, "02_scatter_fit.png", "Each dot is one month, 2005 to 2026.")

# 3. pass-through speed -----------------------------------------------------
cum = list(R["pass_through"]["cumulative"].values())
fig, ax = plt.subplots(figsize=(7.5, 4.4))
labels = ["Same month", "+1 month", "+2 months", "+3 months"]
bars = ax.bar(labels, cum, color=BLUE, width=0.55)
for bar, v in zip(bars, cum):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v*100:.0f}%", ha="center", color=INK, fontsize=10)
ax.axhline(1, color=INK2, linewidth=1, linestyle=":")
ax.text(-0.27, 1.01, "100% pass-through", color=INK2, fontsize=8.5, ha="left", va="bottom")
ax.set_ylim(0, 1.3)
ax.set_ylabel("Share of crude change passed to pump")
ax.set_title("Crude price changes reach the pump within about 2 months")
finish(fig, ax, "03_pass_through_speed.png",
       f"Cumulative effect of a crude change on gasoline (distributed-lag model, {R['pass_through']['n']} months).")

# 4. driver importance ------------------------------------------------------
imp = pd.Series(R["levels_model"]["drop_one_r2_loss"]).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.2))
cols = [ORANGE if k == "Crude oil price" else BLUE for k in imp.index]
ax.barh(imp.index, imp.values * 100, color=cols, height=0.55)
for i, v in enumerate(imp.values):
    ax.text(v * 100 + 0.4, i, f"{v*100:.1f} pts", va="center", color=INK, fontsize=9.5)
ax.set_xlabel("Explanatory power lost if removed (R² percentage points)")
ax.set_xlim(0, imp.max() * 100 * 1.18)
ax.grid(axis="y", visible=False)
ax.set_title("Crude oil is the dominant driver; others add little")
finish(fig, ax, "04_driver_importance.png", f"Full model R² = {R['levels_model']['r2_full']:.2f}; crude alone = {R['levels_model']['r2_crude_only']:.2f}.")

# 5. seasonality ------------------------------------------------------------
s = pd.Series(R["seasonality_usd_gal"]).astype(float)
s.index = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.bar(s.index, s.values * 100, color=[BLUE if v >= 0 else ORANGE for v in s.values], width=0.6)
ax.axhline(0, color=INK2, linewidth=1)
for i, v in enumerate(s.values):
    ax.text(i, v * 100 + (1 if v >= 0 else -1), f"{v*100:+.0f}¢", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
ax.set_ylabel("Cents per gallon vs. yearly average")
ax.set_ylim(-22, 20)
ax.set_title("Gasoline runs about 30 cents higher in May than in January")
finish(fig, ax, "05_seasonality.png", "Gasoline price after removing the crude effect, averaged by calendar month (2005 to 2026).")

# 6. spread by year ---------------------------------------------------------
sp = pd.Series(R["spread"]["by_year"]).astype(float)
sp.index = sp.index.astype(int)
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.bar(sp.index, sp.values, color=[ORANGE if y >= 2022 else BLUE for y in sp.index], width=0.7)
ax.axhline(R["spread"]["mean_2005_2019"], color=INK2, linewidth=1, linestyle=":")
ax.text(2004.6, R["spread"]["mean_2005_2019"] + 0.03, f"2005 to 2019 avg ${R['spread']['mean_2005_2019']:.2f}",
        color=INK2, fontsize=9)
ax.text(2024, sp.max() + 0.06, f"2022+ avg ${R['spread']['mean_2022_on']:.2f}", color=INK, fontsize=9, ha="center")
ax.set_ylim(0, sp.max() + 0.3)
ax.set_ylabel("Gasoline minus crude, $/gal")
ax.set_title("The gap between pump and crude widened by about 50 cents after 2021")
finish(fig, ax, "06_spread_by_year.png", "Annual average spread (covers refining, distribution, taxes and retail margin).")

# 7. backtest ---------------------------------------------------------------
bt = pd.read_csv(ROOT / "outputs/backtest_predictions.csv", parse_dates=["month"], index_col="month")
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(bt.index, bt.actual, color=BLUE)
ax.plot(bt.index, bt.predicted_crude_only, color=ORANGE, linestyle="--")
ax.text(bt.index[-1], bt.actual.iloc[-1] + 0.08, "  Actual", color=INK, fontsize=10)
ax.text(bt.index[-1], bt.predicted_crude_only.iloc[-1] - 0.05, "  Model (trained\n  on 2005 to 2019)", color=INK, fontsize=10, va="top")
ax.set_xlim(bt.index[0], bt.index[-1] + pd.Timedelta(days=500))
ax.set_ylabel("US$ per gallon")
ax.set_title("A model trained before 2020 under-predicts today's prices")
finish(fig, ax, "07_backtest.png",
       f"Out-of-sample test, 2020 to 2026: average error ${R['backtest']['mae_crude_only']:.2f}/gal, "
       f"almost all of it the wider post-2021 spread.")
print("charts written:", sorted(p.name for p in FIG.glob("*.png")))
