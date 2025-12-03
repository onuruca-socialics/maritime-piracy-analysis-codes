import sys
import argparse
from pathlib import Path

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import plotly.express as px
from geodatasets import get_path

import plots
from utils import cramers_corrected_stat
import constants as _c


def load(fn, verbose=True):
    fn = Path(fn)
    if not fn.exists():
        raise RuntimeError(f"File {fn} does not exist.")

    df = pd.read_csv(fn)
    if verbose:
        print(df)

    return df


def corr(fn):
    df = load(fn, verbose=False)
    df.drop(columns=["vessel_name"], inplace=True)
    corr_mtx = df.corr(method="pearson")
    print(corr_mtx)
    scatter_matrix(df)
    plt.show()


def related_cols(fn):
    font = {
        "weight": "bold",
        "size": 16,
    }
    matplotlib.rc("font", **font)

    df = load(fn, verbose=False)
    xdf = pd.crosstab(df["year"], df["region"])
    plots.bar_plot(
        xdf,
        _c.regions[1:],
        _c.years[1:],
        "Years",
    )
    plt.savefig("years_vs_region.png", bbox_inches="tight")
    plots.vbar_plot(xdf, _c.years[1:], "Years")
    plt.savefig("attack_counts_vs_years.png", bbox_inches="tight")

    xdf = pd.crosstab(df["vessel_type"], df["region"])
    plots.bar_plot(
        xdf,
        _c.regions[1:],
        _c.vessel_types[1:],
        "Vessel types",
    )
    plt.savefig("vessel_type_vs_region.png", bbox_inches="tight")
    plots.vbar_plot(xdf, _c.vessel_types[1:], "Vessel types")
    plt.savefig("attack_counts_vs_vessel_types.png", bbox_inches="tight")

    xdf = pd.crosstab(df["attack_type"], df["region"])
    plots.bar_plot(
        xdf,
        _c.regions[1:],
        _c.attack_types[1:],
        "Attack types",
    )
    plt.savefig("attack_type_vs_region.png", bbox_inches="tight")
    plots.vbar_plot(xdf, _c.attack_types[1:], "Attack types")
    plt.savefig("attack_counts_vs_attack_types.png", bbox_inches="tight")


def chi_test(fn):
    df = load(fn, verbose=False)
    df.drop(columns=["vessel_name"], inplace=True)

    col_count = len(df.columns)
    cv_corr = np.zeros((col_count, col_count))

    for i, col1 in enumerate(df):
        for j, col2 in enumerate(df):
            cv = cramers_corrected_stat(df[col1], df[col2])
            cv_corr[i, j] = cv

    cv_corr = pd.DataFrame(cv_corr, df.columns, df.columns)
    print(cv_corr)
    plots.chi_test_plot(cv_corr)


def gis_iqr(df):
    years = range(2016, 2025)
    percent = 0.25
    fig, axs = plt.subplots(3, 3, figsize=(18.5, 10.5))
    mean_vals = [0.0] * len(years)
    median_vals = [0.0] * len(years)
    std_vals = [0.0] * len(years)
    for i, (year, ax) in enumerate(zip(years, axs.reshape(-1))):
        dfi = df.loc[df["year"] == year]
        q1 = dfi["distance"].quantile(percent)
        q3 = dfi["distance"].quantile(1.0 - percent)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        idx = np.where(
            (dfi["distance"] <= lower).values | (dfi["distance"] >= upper).values
        )[0]
        idx += dfi.index[0]
        mean_vals[i] = dfi["distance"].mean()
        median_vals[i] = dfi["distance"].median()
        std_vals[i] = dfi["distance"].std()
        df_cleaned = dfi.drop(index=idx)
        mean_vals[i] = df_cleaned["distance"].mean()
        median_vals[i] = df_cleaned["distance"].median()
        std_vals[i] = df_cleaned["distance"].std()
        df_cleaned["distance"].hist(bins=10, ax=ax)
        ax.set_title(f"{year}")
        ax.tick_params(labelsize="x-large")
        if i > 5:
            ax.set_xlabel("Distance (km)", size="xx-large")
        if i % 3 == 0:
            ax.set_ylabel("Attack count", size="xx-large")

    plt.savefig("years_vs_attack_dist.png", bbox_inches="tight")

    fig, axs = plt.subplots(2, 1, figsize=(18.5 * 0.66, 10.5 * 0.66))
    axs[0].plot(years, mean_vals, linewidth=3)
    axs[1].plot(years, std_vals, linewidth=3)
    axs[0].set_ylabel("Mean (km)", size="xx-large")
    axs[0].tick_params(labelsize="x-large")
    axs[0].grid(True)
    axs[1].set_ylabel("STD (km)", size="xx-large")
    axs[1].set_xlabel("Years", size="xx-large")
    axs[1].tick_params(labelsize="x-large")
    axs[1].grid(True)
    plt.savefig("attack_dist_mean_med.png", bbox_inches="tight")

    q1 = df["distance"].quantile(percent)
    q3 = df["distance"].quantile(1.0 - percent)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    idx = np.where(
        (df["distance"] <= lower).values | (df["distance"] >= upper).values
    )[0]
    df_cleaned = df.drop(index=idx)
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    df_cleaned["distance"].hist(bins=20, ax=ax)
    ax.tick_params(labelsize="x-large")
    ax.set_xlabel("Distance (km)", size="xx-large")
    ax.set_ylabel("Attack count", size="xx-large")
    plt.savefig("attack_dist.png", bbox_inches="tight")


def gis_geopandas(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"], crs="EPSG:4326"))
    world = gpd.read_file(get_path("naturalearth.land"))
    fig = px.density_mapbox(
        gdf, lat=gdf.geometry.y, lon=gdf.geometry.x, radius=15,
        width=1800, height=900, center={"lat": 0.0, "lon": 0.0},
        opacity=0.75,
    ).update_layout(
        mapbox={
            "style": "carto-positron",
            "zoom": 2,
            "layers": [
                {
                    "source": world.geometry.__geo_interface__,
                    "type": "line",
                },
            ],
        },
        margin={"l":0,"r":0,"t":0,"b":0},
    )
    fig.write_image("attack_map.png")


def gis_analysis(fn):
    df = load(fn, verbose=False)
    df_tmp = df.drop(
        columns=["vessel_name", "vessel_type", "attack_type", "place", "coord"],
    )
    gis_iqr(df_tmp)
    gis_geopandas(df_tmp)


def main():
    parser = argparse.ArgumentParser(
        prog="ProgramName",  # TODO: Change prog name
        description="What the program does",
        epilog="Text at the bottom of help",
    )
    subparsers = parser.add_subparsers(help="Subcommand help", dest="command")

    load_cmd = subparsers.add_parser("load")
    load_cmd.add_argument("filename")

    corr_cmd = subparsers.add_parser("corr")
    corr_cmd.add_argument("filename")

    rel_cmd = subparsers.add_parser("rel")
    rel_cmd.add_argument("filename")

    chi_cmd = subparsers.add_parser("chi")
    chi_cmd.add_argument("filename")

    gis_cmd = subparsers.add_parser("gis")
    gis_cmd.add_argument("filename")

    help_cmd = subparsers.add_parser("help")
    help_cmd.add_argument("subject")

    args = parser.parse_args()

    match args.command:
        case "load":
            load(args.filename)
        case "corr":
            corr(args.filename)
        case "rel":
            related_cols(args.filename)
        case "chi":
            chi_test(args.filename)
        case "gis":
            gis_analysis(args.filename)
        case _:
            print("Provide a subcommand")

    return 0


if __name__ == "__main__":
    sys.exit(main())
