"""Step 2: what drives U.S. gasoline prices?

Questions answered
  Q1. How much of gasoline's price is explained by crude oil vs. inventories,
      production and drilling activity?            (levels model + drop-one R^2)
  Q2. When crude moves $1/bbl, how much does gasoline move, and how fast?
                                                     (distributed-lag model on monthly changes)
  Q3. Do pump prices rise faster than they fall?    ("rockets and feathers" asymmetry test)
  Q4. Is there a seasonal (summer) premium?
  Q5. Does the model hold up on data it never saw?  (train 2005-2019, test 2020+)
  Q6. What happens to gasoline if crude jumps 20%?  (scenario table)

All regressions are plain OLS written with numpy, with Newey-West (HAC)
standard errors because monthly price data are autocorrelated.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
GAL_PER_BBL = 42.0


# ---------------------------------------------------------------- helpers
def ols(y, X, names, hac_lags=0):
    """OLS with optional Newey-West standard errors. Returns dict."""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    if hac_lags > 0:
        S = (X * e[:, None]).T @ (X * e[:, None])
        for L in range(1, hac_lags + 1):
            w = 1 - L / (hac_lags + 1)
            G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
            S += w * (G + G.T)
        V = XtX_inv @ S @ XtX_inv * n / (n - k)
    else:
        V = XtX_inv * (e @ e) / (n - k)
    se = np.sqrt(np.diag(V))
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return {"b": pd.Series(b, names), "se": pd.Series(se, names), "V": pd.DataFrame(V, names, names),
            "t": pd.Series(b / se, names), "r2": r2, "n": n, "resid": e}


def wald_sum(res, names_a, names_b):
    """t-stat for H0: sum(coef in a) == sum(coef in b)."""
    idx = res["b"].index
    c = np.array([(1 if i in names_a else 0) - (1 if i in names_b else 0) for i in idx], float)
    diff = c @ res["b"].values
    se = np.sqrt(c @ res["V"].values @ c)
    return diff, se, diff / se


def month_dummies(index):
    d = pd.get_dummies(index.month, prefix="m", drop_first=True).astype(float)
    d.index = index
    return d


# ---------------------------------------------------------------- data
d = pd.read_csv(DATA / "monthly_model.csv", parse_dates=["month"], index_col="month")
d["crude_usd_gal"] = d.wti_usd_bbl / GAL_PER_BBL                 # crude in the same unit as gasoline
d["stocks_bn_bbl"] = d.crude_stocks_kbbl / 1e6
d["production_mbd"] = d.crude_production_kbd / 1e3
d["rigs_100"] = d.oil_rigs / 100
d["indpro"] = d.industrial_production / 10          # coefficient = effect of +10 index points
results = {"window": [f"{d.index.min():%Y-%m}", f"{d.index.max():%Y-%m}"], "n_months": len(d)}

# ---------------------------------------------------------------- Q1 levels model
drivers = {"Crude oil price": "crude_usd_gal", "Crude inventories": "stocks_bn_bbl",
           "Crude production": "production_mbd", "Oil rig count": "rigs_100",
           "Industrial production (demand)": "indpro"}
md = month_dummies(d.index)
Xcols = list(drivers.values())
X = pd.concat([pd.Series(1.0, d.index, name="const"), d[Xcols], md], axis=1)
lev = ols(d.gasoline_usd_gal, X, X.columns, hac_lags=12)

# simple model: crude only
X1 = pd.concat([pd.Series(1.0, d.index, name="const"), d[["crude_usd_gal"]]], axis=1)
simple = ols(d.gasoline_usd_gal, X1, X1.columns, hac_lags=12)

# drop-one importance: how much R^2 is lost if a driver is removed
importance = {}
for label, col in drivers.items():
    Xd = X.drop(columns=col)
    importance[label] = lev["r2"] - ols(d.gasoline_usd_gal, Xd, Xd.columns)["r2"]
Xs = X.drop(columns=md.columns)
importance["Seasonality (month)"] = lev["r2"] - ols(d.gasoline_usd_gal, Xs, Xs.columns)["r2"]

results["levels_model"] = {
    "r2_crude_only": round(simple["r2"], 3),
    "r2_full": round(lev["r2"], 3),
    "crude_only_intercept_usd_gal": round(simple["b"]["const"], 3),
    "crude_only_slope": round(simple["b"]["crude_usd_gal"], 3),
    "coefficients": {k: {"coef": round(lev["b"][v], 4), "se": round(lev["se"][v], 4),
                         "t": round(lev["t"][v], 2)} for k, v in drivers.items()},
    "drop_one_r2_loss": {k: round(v, 4) for k, v in importance.items()},
}
results["correlations"] = d[["gasoline_usd_gal", "wti_usd_bbl", "crude_stocks_kbbl",
                             "crude_production_kbd", "oil_rigs", "industrial_production"]].corr().round(2).to_dict()

# ---------------------------------------------------------------- Q2 pass-through speed
LAGS = 3
c = d.diff()
dl = pd.DataFrame({"d_gas": c.gasoline_usd_gal})
for L in range(LAGS + 1):
    dl[f"d_crude_L{L}"] = c.crude_usd_gal.shift(L)
dl["d_stocks"] = c.stocks_bn_bbl
dl["d_prod"] = c.production_mbd
dl["d_rigs"] = c.rigs_100
dl["d_indpro"] = c.indpro
dl = dl.dropna()
Xdl = pd.concat([pd.Series(1.0, dl.index, name="const"), dl.drop(columns="d_gas"),
                 month_dummies(dl.index)], axis=1)
pt = ols(dl.d_gas, Xdl, Xdl.columns, hac_lags=3)
lag_names = [f"d_crude_L{L}" for L in range(LAGS + 1)]
lag_coefs = pt["b"][lag_names]
cum = lag_coefs.cumsum()
lr_diff, lr_se, _ = wald_sum(pt, lag_names, [])
results["pass_through"] = {
    "lag_coefs": {f"month_{L}": round(v, 3) for L, v in enumerate(lag_coefs)},
    "cumulative": {f"month_{L}": round(v, 3) for L, v in enumerate(cum)},
    "long_run": round(lr_diff, 3), "long_run_se": round(lr_se, 3),
    "cents_per_gal_per_usd_bbl": round(100 * lr_diff / GAL_PER_BBL, 2),
    "share_passed_by_month_1": round(cum.iloc[1] / cum.iloc[-1], 3),
    "r2": round(pt["r2"], 3), "n": pt["n"],
    "other_drivers_monthly_changes": {k: {"coef": round(pt["b"][k], 4), "t": round(pt["t"][k], 2)}
                                      for k in ["d_stocks", "d_prod", "d_rigs", "d_indpro"]},
}

# ---------------------------------------------------------------- Q3 rockets & feathers
up = c.crude_usd_gal.clip(lower=0)
dn = c.crude_usd_gal.clip(upper=0)
af = pd.DataFrame({"d_gas": c.gasoline_usd_gal})
for L in range(LAGS + 1):
    af[f"up_L{L}"] = up.shift(L)
    af[f"dn_L{L}"] = dn.shift(L)
af = af.dropna()
Xaf = pd.concat([pd.Series(1.0, af.index, name="const"), af.drop(columns="d_gas"),
                 month_dummies(af.index)], axis=1)
asy = ols(af.d_gas, Xaf, Xaf.columns, hac_lags=3)
up_n = [f"up_L{L}" for L in range(LAGS + 1)]
dn_n = [f"dn_L{L}" for L in range(LAGS + 1)]
diff0, se0, t0 = wald_sum(asy, ["up_L0"], ["dn_L0"])
difflr, selr, tlr = wald_sum(asy, up_n, dn_n)
results["asymmetry"] = {
    "up_cumulative": {f"month_{L}": round(v, 3) for L, v in enumerate(asy["b"][up_n].cumsum())},
    "down_cumulative": {f"month_{L}": round(v, 3) for L, v in enumerate(asy["b"][dn_n].cumsum())},
    "same_month_gap": round(diff0, 3), "same_month_t": round(t0, 2),
    "long_run_gap": round(difflr, 3), "long_run_t": round(tlr, 2),
}

# ---------------------------------------------------------------- Q4 seasonality
# month effect after controlling for crude: average residual from crude-only model by month
seas = pd.Series(simple["resid"], d.index).groupby(d.index.month).mean()
seas = seas - seas.mean()
results["seasonality_usd_gal"] = {int(k): round(v, 3) for k, v in seas.items()}

# retail spread over crude (refining + distribution + taxes + margin)
d["spread_usd_gal"] = d.gasoline_usd_gal - d.crude_usd_gal
results["spread"] = {
    "mean": round(d.spread_usd_gal.mean(), 2),
    "mean_2005_2019": round(d.spread_usd_gal[:"2019"].mean(), 2),
    "mean_2022_on": round(d.spread_usd_gal["2022":].mean(), 2),
    "by_year": {int(k): round(v, 2) for k, v in d.spread_usd_gal.groupby(d.index.year).mean().items()},
}

# ---------------------------------------------------------------- Q5 out-of-sample backtest
train = d.index < "2020-01-01"
Xtr, Xte = X[train], X[~train]
bt = ols(d.gasoline_usd_gal[train], Xtr, X.columns)
pred = Xte.values @ bt["b"].values
err = d.gasoline_usd_gal[~train].values - pred
bt1 = ols(d.gasoline_usd_gal[train], X1[train], X1.columns)
pred1 = X1[~train].values @ bt1["b"].values
naive = d.gasoline_usd_gal.shift(1)[~train].values               # "same as last month"
results["backtest"] = {
    "train": "2005-01 to 2019-12", "test": f"2020-01 to {d.index.max():%Y-%m}",
    "mae_full_model": round(np.mean(np.abs(err)), 3),
    "mae_crude_only": round(np.mean(np.abs(d.gasoline_usd_gal[~train].values - pred1)), 3),
    "mae_naive_last_month": round(np.nanmean(np.abs(d.gasoline_usd_gal[~train].values - naive)), 3),
    "mean_bias_full_model": round(np.mean(err), 3),
    "mean_bias_crude_only": round(np.mean(d.gasoline_usd_gal[~train].values - pred1), 3),
}
pd.DataFrame({"actual": d.gasoline_usd_gal[~train], "predicted_full": pred,
              "predicted_crude_only": pred1}).to_csv(OUT / "backtest_predictions.csv")

# ---------------------------------------------------------------- dollar index (short window)
all_m = pd.read_csv(DATA / "monthly_all.csv", parse_dates=["month"], index_col="month")
dw = all_m[["wti_usd_bbl", "dollar_index", "gasoline_usd_gal"]].dropna()
r = np.log(dw).diff().dropna()
results["dollar"] = {
    "window": [f"{dw.index.min():%Y-%m}", f"{dw.index.max():%Y-%m}"], "n_months": len(dw),
    "corr_monthly_pct_change_dollar_vs_wti": round(r.dollar_index.corr(r.wti_usd_bbl), 2),
    "corr_monthly_pct_change_dollar_vs_gas": round(r.dollar_index.corr(r.gasoline_usd_gal), 2),
}

# ---------------------------------------------------------------- Q6 scenarios
latest = all_m[["wti_usd_bbl", "gasoline_usd_gal"]].dropna().iloc[-1]
latest_month = all_m[["wti_usd_bbl", "gasoline_usd_gal"]].dropna().index[-1]
# Asymmetry was not statistically significant (Q3), so scenarios use the
# symmetric pass-through path from Q2.
path_coef = cum.values
rows = []
for pct in [-30, -20, -10, 10, 20, 30]:
    d_crude_gal = latest.wti_usd_bbl * pct / 100 / GAL_PER_BBL
    path = path_coef * d_crude_gal
    rows.append({"crude_change_pct": pct,
                 "new_wti_usd_bbl": round(latest.wti_usd_bbl * (1 + pct / 100), 2),
                 "gas_change_month0": round(path[0], 3),
                 "gas_change_month1": round(path[1], 3),
                 "gas_change_full": round(path[-1], 3),
                 "new_gas_usd_gal": round(latest.gasoline_usd_gal + path[-1], 2)})
scen = pd.DataFrame(rows)
scen.to_csv(OUT / "scenarios.csv", index=False)
results["scenario_base"] = {"month": f"{latest_month:%Y-%m}", "wti_usd_bbl": round(latest.wti_usd_bbl, 2),
                            "gasoline_usd_gal": round(latest.gasoline_usd_gal, 3)}
results["scenarios"] = rows
# parameters the interactive scenario page needs
results["scenario_params"] = {"cumulative_pass_through": [round(v, 4) for v in path_coef]}

d.to_csv(DATA / "monthly_model_features.csv")
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)
print(json.dumps({k: v for k, v in results.items() if k not in ("correlations", "spread")}, indent=1,
                 default=float))
