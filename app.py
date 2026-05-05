from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
MONTH_NAME_MAP = {index + 1: name for index, name in enumerate(MONTH_LABELS)}

VALUE_CONFIG = {
    "Solarproduktion": {"column": "Solarproduktion", "category": "solar"},
    "Solarproduktion Tracker 1": {"column": "Solarproduktion-Tracker 1", "category": "solar"},
    "Solarproduktion Tracker 2": {"column": "Solarproduktion-Tracker 2", "category": "solar"},
    "Batterie Laden": {"column": "Batterie Laden", "category": "battery"},
    "Batterie Entladen": {"column": "Batterie Entladen", "category": "battery"},
    "Netzeinspeisung": {"column": "Netzeinspeisung", "category": "grid"},
    "Netzbezug": {"column": "Netzbezug", "category": "grid"},
    "Hausverbrauch": {"column": "Hausverbrauch", "category": "consumption"},
    "Summe Wallbox Laden": {"column": "Summe Wallbox Laden", "category": "consumption"},
    "Wallbox Solarladeleistung": {"column": "Wallbox Solarladeleistung", "category": "solar"},
    "Summe Verbrauch": {"column": "Summe Verbrauch", "category": "consumption"},
}

DEFAULT_VALUES = {"Solarproduktion", "Netzbezug"}

ALL_DATA_COLUMNS = {
    config["column"] for config in VALUE_CONFIG.values()
}.union({"Netzeinspeisung", "Netzbezug", "Hausverbrauch", "Solarproduktion"})

VALUE_COLORS = {
    "Solarproduktion": "#F59E0B",
    "Solarproduktion Tracker 1": "#FBBF24",
    "Solarproduktion Tracker 2": "#F97316",
    "Batterie Laden": "#8B5CF6",
    "Batterie Entladen": "#A78BFA",
    "Netzeinspeisung": "#38BDF8",
    "Netzbezug": "#0EA5E9",
    "Hausverbrauch": "#EF4444",
    "Summe Wallbox Laden": "#F43F5E",
    "Wallbox Solarladeleistung": "#14B8A6",
    "Summe Verbrauch": "#B91C1C",
}

VALUE_ORDER = list(VALUE_CONFIG.keys())


@dataclass
class Dataset:
    label: str
    year: Optional[int]
    raw: pd.DataFrame
    monthly: pd.DataFrame
    warnings: list[str]

    @property
    def display_year(self) -> str:
        return str(self.year) if self.year is not None else self.label


def read_csv_robust(content: bytes) -> pd.DataFrame:
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    seps = [";", ",", "\t"]

    for encoding in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(BytesIO(content), sep=sep, encoding=encoding, dtype=str)
            except Exception:
                continue
            if df.shape[1] > 1:
                return df

    for encoding in encodings:
        try:
            df = pd.read_csv(BytesIO(content), sep=None, engine="python", encoding=encoding, dtype=str)
        except Exception:
            continue
        if df.shape[1] > 1:
            return df

    return pd.read_csv(BytesIO(content), sep=";", encoding="utf-8-sig", dtype=str)


def parse_german_csv(content: bytes) -> pd.DataFrame:
    df = read_csv_robust(content)
    df.columns = [col.strip() for col in df.columns]

    if "Zeitstempel" in df.columns:
        df["Zeitstempel"] = pd.to_datetime(
            df["Zeitstempel"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
        )

    for col in df.columns:
        if col == "Zeitstempel":
            continue
        series = df[col].astype(str).str.strip()
        series = series.str.replace(" ", "", regex=False)
        series = series.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        df[col] = pd.to_numeric(series, errors="coerce")

    return df


def format_number_de(value: float, decimals: int = 1) -> str:
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "X").replace(".", ",").replace("X", ".")


def format_kwh(value_wh: Optional[float]) -> str:
    if value_wh is None or pd.isna(value_wh):
        return "k.A."
    return f"{format_number_de(value_wh / 1000)} kWh"


def format_percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "k.A."
    return f"{format_number_de(value, 1)} %"


def get_value_color(value_name: str) -> str:
    return VALUE_COLORS.get(value_name, "#6B7280")


def load_dataset(name: str, content: bytes) -> Dataset:
    warnings: list[str] = []
    df = parse_german_csv(content)

    if "Zeitstempel" not in df.columns:
        warnings.append("Spalte 'Zeitstempel' fehlt.")
        df["Zeitstempel"] = pd.NaT

    missing_columns = [col for col in sorted(ALL_DATA_COLUMNS) if col not in df.columns]
    for col in missing_columns:
        df[col] = pd.NA
    if missing_columns:
        warnings.append("Fehlende Spalten: " + ", ".join(missing_columns))

    wallbox_fallbacks = {
        "Summe Wallbox Laden": "Summe Wallbox Laden ID: 0",
        "Wallbox Solarladeleistung": "Wallbox Solarladeleistung ID: 0",
    }
    for primary, fallback in wallbox_fallbacks.items():
        if primary in df.columns and fallback in df.columns:
            df[primary] = df[primary].combine_first(df[fallback])

    year = None
    timestamps = df["Zeitstempel"].dropna()
    if not timestamps.empty:
        year = int(timestamps.iloc[0].year)
    else:
        warnings.append("Jahr konnte nicht erkannt werden.")

    df = df.sort_values("Zeitstempel")
    df["Monat"] = df["Zeitstempel"].dt.month

    numeric_cols = [col for col in df.columns if col not in ("Zeitstempel", "Monat")]
    present_numeric_cols = [col for col in numeric_cols if col not in missing_columns]

    df_monthly = (
        df.groupby("Monat")[numeric_cols]
        .sum(min_count=1)
        .reindex(range(1, 13))
    )

    months_found = sorted(int(m) for m in df["Monat"].dropna().unique())
    if len(months_found) != 12:
        missing_months = [MONTH_NAME_MAP[m] for m in range(1, 13) if m not in months_found]
        warnings.append("Es fehlen Monate: " + ", ".join(missing_months))

    if present_numeric_cols:
        if (df_monthly[present_numeric_cols] < 0).any().any():
            warnings.append("Es gibt negative Werte, bitte pruefen.")
        if df_monthly[present_numeric_cols].isna().any().any():
            warnings.append("Nicht-numerische oder fehlende Werte gefunden.")

    return Dataset(label=name, year=year, raw=df, monthly=df_monthly, warnings=warnings)


def build_kpis(dataset: Dataset) -> dict[str, Optional[float]]:
    def sum_wh(column: str) -> Optional[float]:
        if column not in dataset.monthly.columns:
            return None
        return dataset.monthly[column].sum(min_count=1)

    solar_total = sum_wh("Solarproduktion")
    netzeinspeisung_total = sum_wh("Netzeinspeisung")
    netzbezug_total = sum_wh("Netzbezug")
    hausverbrauch_total = sum_wh("Hausverbrauch")

    eigenverbrauch = None
    if solar_total is not None and netzeinspeisung_total is not None and solar_total > 0:
        eigenverbrauch = (solar_total - netzeinspeisung_total) / solar_total * 100

    autarkie = None
    if hausverbrauch_total is not None and netzbezug_total is not None and hausverbrauch_total > 0:
        autarkie = (hausverbrauch_total - netzbezug_total) / hausverbrauch_total * 100

    return {
        "solar_total": solar_total,
        "eigenverbrauch": eigenverbrauch,
        "autarkie": autarkie,
        "netzbezug_total": netzbezug_total,
    }


def main() -> None:
    st.set_page_config(page_title="E3DC Jahresvergleich", layout="wide")
    st.title("E3DC Jahresvergleich")

    st.sidebar.header("Dateien")
    uploads = st.sidebar.file_uploader(
        "CSV-Dateien hochladen",
        type=["csv"],
        accept_multiple_files=True,
    )

    datasets_by_label: dict[str, Dataset] = {}
    data_dir = Path(__file__).resolve().parent
    local_files = sorted(data_dir.glob("*_All_in*.csv"))
    for path in local_files:
        datasets_by_label[path.name] = load_dataset(path.name, path.read_bytes())

    if uploads:
        for upload in uploads:
            datasets_by_label[upload.name] = load_dataset(upload.name, upload.getvalue())

    datasets = list(datasets_by_label.values())

    if not datasets:
        st.info("Bitte eine oder mehrere CSV-Dateien hochladen oder CSVs im Ordner ablegen.")
        return

    datasets = sorted(
        datasets,
        key=lambda item: (item.year is None, item.year or 0, item.label),
    )

    st.sidebar.subheader("Jahr-Auswahl")
    selected_labels: list[str] = []
    for dataset in datasets:
        if st.sidebar.checkbox(dataset.display_year, value=True, key=f"year_{dataset.label}"):
            selected_labels.append(dataset.label)

    st.sidebar.subheader("Werte-Auswahl")
    selected_values: list[str] = []
    for value_name in VALUE_ORDER:
        if st.sidebar.checkbox(
            value_name,
            value=value_name in DEFAULT_VALUES,
            key=f"value_{value_name}",
        ):
            selected_values.append(value_name)

    selected_datasets = [item for item in datasets if item.label in selected_labels]

    tabs = st.tabs(
        [
            "Jahr gesamt",
            "Monatliche Aufschluesselung",
            "Gesamt (alle Jahre)",
            "Rohdaten",
        ]
    )

    with tabs[0]:
        st.subheader("Jahr gesamt")
        if not selected_values:
            st.info("Bitte mindestens einen Wert auswaehlen.")
        elif not selected_datasets:
            st.info("Bitte mindestens ein Jahr auswaehlen.")
        else:
            fig_total = go.Figure()
            shown_values: set[str] = set()
            for dataset in selected_datasets:
                for value_name in selected_values:
                    column = VALUE_CONFIG[value_name]["column"]
                    total_wh = dataset.monthly[column].sum(min_count=1)
                    total_kwh = None if total_wh is None or pd.isna(total_wh) else total_wh / 1000
                    color = get_value_color(value_name)
                    showlegend = value_name not in shown_values
                    fig_total.add_trace(
                        go.Bar(
                            x=[dataset.display_year],
                            y=[total_kwh],
                            name=value_name,
                            legendgroup=value_name,
                            showlegend=showlegend,
                            marker=dict(color=color),
                            hovertemplate=(
                                f"{dataset.display_year}<br>{value_name}: "
                                "%{y:.1f} kWh<extra></extra>"
                            ),
                        )
                    )
                    if showlegend:
                        shown_values.add(value_name)

            fig_total.update_layout(
                barmode="group",
                legend_title_text="Jahr und Wert",
                xaxis_title="Jahr",
                yaxis_title="kWh",
                bargap=0.18,
                bargroupgap=0.08,
                template="plotly_white",
            )
            st.plotly_chart(fig_total, use_container_width=True)

            totals_table = pd.DataFrame(
                {
                    f"{dataset.display_year}": {
                        f"{value_name} (kWh)": (
                            None
                            if (total := dataset.monthly[VALUE_CONFIG[value_name]["column"]].sum(min_count=1)) is None
                            or pd.isna(total)
                            else total / 1000
                        )
                        for value_name in selected_values
                    }
                    for dataset in selected_datasets
                }
            ).T
            st.dataframe(totals_table, use_container_width=True)

    with tabs[1]:
        st.subheader("Monatliche Aufschluesselung")
        if not selected_values:
            st.info("Bitte mindestens einen Wert auswaehlen.")
        elif not selected_datasets:
            st.info("Bitte mindestens ein Jahr auswaehlen.")
        else:
            fig = go.Figure()
            shown_values: set[str] = set()
            for dataset in selected_datasets:
                for value_name in selected_values:
                    column = VALUE_CONFIG[value_name]["column"]
                    color = get_value_color(value_name)
                    y_values = dataset.monthly[column] / 1000
                    showlegend = value_name not in shown_values
                    fig.add_trace(
                        go.Bar(
                            x=MONTH_LABELS,
                            y=y_values,
                            name=value_name,
                            legendgroup=value_name,
                            showlegend=showlegend,
                            marker=dict(color=color),
                            hovertemplate=(
                                f"%{{x}} {dataset.display_year}<br>{value_name}: "
                                "%{y:.1f} kWh<extra></extra>"
                            ),
                        )
                    )
                    if showlegend:
                        shown_values.add(value_name)

            fig.update_layout(
                barmode="group",
                legend_title_text="Jahr und Wert",
                xaxis_title="Monat",
                yaxis_title="kWh",
                bargap=0.18,
                bargroupgap=0.08,
                template="plotly_white",
            )
            fig.update_xaxes(categoryorder="array", categoryarray=MONTH_LABELS)
            st.plotly_chart(fig, use_container_width=True)

            table = pd.DataFrame(index=MONTH_LABELS)
            for dataset in selected_datasets:
                for value_name in selected_values:
                    column = VALUE_CONFIG[value_name]["column"]
                    series = dataset.monthly[column] / 1000
                    table[f"{dataset.display_year} {value_name}"] = series.values
            st.dataframe(table, use_container_width=True)

    with tabs[2]:
        st.subheader("Gesamt (alle Jahre)")
        if not selected_values:
            st.info("Bitte mindestens einen Wert auswaehlen.")
        elif not selected_datasets:
            st.info("Bitte mindestens ein Jahr auswaehlen.")
        else:
            totals_by_value: dict[str, Optional[float]] = {}
            for value_name in selected_values:
                column = VALUE_CONFIG[value_name]["column"]
                total_wh = 0.0
                has_value = False
                for dataset in selected_datasets:
                    value = dataset.monthly[column].sum(min_count=1)
                    if value is None or pd.isna(value):
                        continue
                    total_wh += float(value)
                    has_value = True
                totals_by_value[value_name] = total_wh if has_value else None

            fig_all = go.Figure()
            for value_name in selected_values:
                total_wh = totals_by_value.get(value_name)
                total_kwh = None if total_wh is None else total_wh / 1000
                fig_all.add_trace(
                    go.Bar(
                        x=[value_name],
                        y=[total_kwh],
                        name=value_name,
                        marker=dict(color=get_value_color(value_name)),
                        hovertemplate=(
                            f"{value_name}: "
                            "%{y:.1f} kWh<extra></extra>"
                        ),
                    )
                )

            fig_all.update_layout(
                barmode="group",
                legend_title_text="Wert",
                xaxis_title="Wert",
                yaxis_title="kWh",
                bargap=0.18,
                bargroupgap=0.08,
                template="plotly_white",
            )
            st.plotly_chart(fig_all, use_container_width=True)

            totals_table = pd.DataFrame(
                {
                    "Summe (kWh)": {
                        value_name: (
                            None if totals_by_value[value_name] is None
                            else totals_by_value[value_name] / 1000
                        )
                        for value_name in selected_values
                    }
                }
            )
            st.dataframe(totals_table, use_container_width=True)

    with tabs[3]:
        for dataset in datasets:
            st.markdown(f"**{dataset.display_year}**")
            st.dataframe(dataset.raw, use_container_width=True)


if __name__ == "__main__":
    main()
