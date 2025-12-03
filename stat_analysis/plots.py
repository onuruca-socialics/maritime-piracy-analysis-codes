import numpy as np
import matplotlib.pyplot as plt

import constants as _c


def bar_plot(df, col_labels, row_labels, ylabel):
    total = df.sum(axis=1)
    df = df.div(total.values, axis=0) * 100
    df_cum = df.cumsum(axis=1)

    category_colors = plt.colormaps["RdBu"](np.linspace(0.15, 0.85, df.shape[1]))

    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, df.sum(axis=1).max())

    for i, (clabel, color) in enumerate(zip(col_labels, category_colors)):
        widths = df.values[:, i]
        starts = df_cum.values[:, i] - widths
        rects = ax.barh(
            row_labels,
            widths,
            left=starts,
            height=0.5,
            label=clabel,
            color=color,
        )
        r, g, b, _ = color
        text_color = "white" if r * g * b < 0.5 else "darkgrey"
        ax.bar_label(rects, label_type="center", color=text_color, fmt="%d")

    ax.legend(
        ncols=len(col_labels),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="medium",
    )
    ax.set_ylabel(ylabel)


def vbar_plot(df, col_labels, xlabel):
    total = df.sum(axis=1)
    fig, ax = plt.subplots(figsize=(18.5, 10.5))
    rects = ax.bar(col_labels, total)
    ax.bar_label(rects, label_type="center", color="white", fmt="%d")
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Attack count")
    ax.tick_params(axis="x", labelrotation=75)


def chi_test_plot(df):
    labels = ["Year", "Vessel type", "Attack type", "Place", "Region"]
    fig, ax = plt.subplots(figsize=(18.5 * 0.66, 10.5 * 0.66))
    im = ax.imshow(df)
    ax.set_xticks(
        range(len(labels)), labels=labels,
        rotation=45, ha="right", rotation_mode="anchor", size="xx-large"
    )
    ax.set_yticks(
        range(len(labels)), labels=labels, size="xx-large"
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(
                j, i, round(df.values[i, j], 2), ha="center", va="center", color="w",
                size="xx-large",
            )

    plt.savefig("chi_test_plot.png", bbox_inches="tight")

