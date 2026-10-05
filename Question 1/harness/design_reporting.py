"""Numbered display and export helpers for the Design notebook."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display


def comparison(study, names, region="validation"):
    """Wide station/product table with physical-unit errors and separate resource columns."""
    overall = study.model_table(region)
    grouped = study.grouped_table(names, region)
    rows = []
    for name in names:
        row = {"model": name}
        for population in ("established", "held-out"):
            match = overall[(overall["model"] == name) & (overall["population"] == population)].iloc[0]
            label = "established stations" if population == "established" else "held-out station 4"
            row[f"{label} RMSE (m/s²)"] = match["RMSE (m/s²)"]
            row[f"{label} MSE (m/s²)²"] = match.MSE
            for product in study.world.product_names:
                value = grouped[(grouped["model"] == name) & (grouped["population"] == population)
                                & (grouped["product"] == product)]["RMSE (m/s²)"].iloc[0]
                row[f"{label}, {product} RMSE"] = value
        resources = overall[overall.model == name].iloc[0]
        row["parameters"] = int(resources.parameters)
        row["fit seconds"] = float(resources["fit seconds"])
        rows.append(row)
    return pd.DataFrame(rows)


def deployment_summary(study, region="validation"):
    """Compact, validation-only evidence for the two Part 4 deployment choices."""
    table = study.model_table(region)
    rows = []
    for model in table["model"].drop_duplicates():
        entries = table[table.model == model].set_index("population")
        established, held_out = entries.loc["established"], entries.loc["held-out"]
        rows.append({
            "model": model,
            "established validation RMSE (m/s²)": established["RMSE (m/s²)"],
            "established validation MSE (m/s²)²": established.MSE,
            "held-out validation RMSE (m/s²)": held_out["RMSE (m/s²)"],
            "held-out validation MSE (m/s²)²": held_out.MSE,
            "parameters": int(established.parameters),
            "fit seconds": established["fit seconds"],
        })
    return pd.DataFrame(rows)


def publish(number, table=None, directory="results/design"):
    """Display and export one authoritative output identifier."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Output {number}")
    if table is not None:
        table = table.copy()
        table.to_csv(destination / f"{number}.csv", index=False)
        table.to_latex(destination / f"{number}.tex", index=False, escape=True,
                       float_format="%.3f", na_rep="unavailable")
        display(table.round(3).replace({np.nan: "unavailable"}))
    for index, figure_id in enumerate(plt.get_fignums(), 1):
        figure = plt.figure(figure_id)
        figure.savefig(destination / f"{number}-{index}.pdf", bbox_inches="tight")
        display(figure)
        plt.close(figure)
