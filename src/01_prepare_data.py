"""Step 1: load raw FRED + EIA files and build one clean monthly table.

Raw files live in data/raw/. EIA .xls files were converted to .xlsx with
LibreOffice (`soffice --headless --convert-to xlsx`) because the old .xls
format needs an extra library.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def read_fred(name):
    df = pd.read_csv(RAW / f"{name}.csv", parse_dates=["observation_date"])
    df[name] = pd.to_numeric(df[name], errors="coerce")
    s = df.set_index("observation_date")[name].dropna()
    return s.resample("MS").mean()          # monthly average, month-start index


def read_eia(name, col):
    df = pd.read_excel(RAW / f"{name}.xlsx", sheet_name="Data 1", skiprows=3, header=None,
                       names=["date", col])
    df["date"] = pd.to_datetime(df["date"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    s = df.set_index("date")[col].dropna()
    s.index = s.index.to_period("M").to_timestamp()   # EIA uses mid-month dates
    return s


def main():
    m = pd.concat({
        "gasoline_usd_gal": read_fred("GASREGW"),
        "wti_usd_bbl": read_fred("MCOILWTICO"),
        "dollar_index": read_fred("DTWEXBGS"),
        "industrial_production": read_fred("INDPRO"),
        "crude_stocks_kbbl": read_eia("MCRSTUS1m", "x"),
        "crude_production_kbd": read_eia("MCRFPUS2m", "x"),
        "oil_rigs": read_eia("E_ERTRRO_XR0_NUS_Cm", "x"),
    }, axis=1, sort=True)
    m.index.name = "month"
    full = m.loc["1986":]
    full.to_csv(OUT / "monthly_all.csv")

    core = m.dropna(subset=["gasoline_usd_gal", "wti_usd_bbl", "crude_stocks_kbbl",
                            "crude_production_kbd", "oil_rigs",
                            "industrial_production"])
    core.to_csv(OUT / "monthly_model.csv")
    print(f"Model window: {core.index.min():%Y-%m} to {core.index.max():%Y-%m}, {len(core)} months")
    print(core.describe().round(2).T)


if __name__ == "__main__":
    main()
