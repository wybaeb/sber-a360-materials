from __future__ import annotations

from pathlib import Path
from textwrap import fill
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


PALETTE = {
    "green": "#21A038",
    "green_dark": "#177A2B",
    "green_light": "#E7F6EB",
    "blue": "#0F5E9C",
    "blue_light": "#E7F1F8",
    "orange": "#F59E0B",
    "orange_light": "#FFF5DE",
    "red": "#D64545",
    "red_light": "#FCE8E8",
    "gray": "#5B6575",
    "gray_light": "#EEF2F5",
    "text": "#1F2937",
    "border": "#D7DEE6",
}

NUMERIC_COLUMNS = [
    "заявки",
    "одобрено",
    "выдано",
    "средний_чек_тыс_руб",
    "расход_на_маркетинг_тыс_руб",
    "доля_цифровых_заявок_проц",
    "просрочка_30дн_проц",
    "чистый_доход_млн_руб",
]

OUTLIER_COLUMNS = [
    "расход_на_маркетинг_тыс_руб",
    "средний_чек_тыс_руб",
    "чистый_доход_млн_руб",
]

REGRESSION_SLICE = {
    "продукт": "Кредитная карта",
    "регион": "Центр",
    "сегмент": "Массовый",
}

FORECAST_SCENARIOS = pd.DataFrame(
    {
        "Сценарий": ["Осторожный", "Базовый", "Усиленный"],
        "Маркетинг, тыс. руб.": [3300, 3900, 4500],
    }
)


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "axes.edgecolor": PALETTE["border"],
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "grid.color": "#DDE5EB",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.85,
    }
)


def synthesize_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2024-01-01", periods=24, freq="MS")

    products = {
        "Кредитная карта": {
            "base_applications": 560,
            "base_marketing": 3400,
            "approval": 71,
            "issue": 79,
            "ticket": 118,
            "margin": 0.064,
            "delinquency": 3.7,
            "digital_share": 72,
        },
        "Потребительский кредит": {
            "base_applications": 430,
            "base_marketing": 3900,
            "approval": 63,
            "issue": 73,
            "ticket": 420,
            "margin": 0.039,
            "delinquency": 4.4,
            "digital_share": 57,
        },
        "Ипотека": {
            "base_applications": 140,
            "base_marketing": 5200,
            "approval": 51,
            "issue": 67,
            "ticket": 4100,
            "margin": 0.014,
            "delinquency": 1.7,
            "digital_share": 45,
        },
        "Вклад": {
            "base_applications": 350,
            "base_marketing": 1600,
            "approval": 88,
            "issue": 91,
            "ticket": 910,
            "margin": 0.009,
            "delinquency": 0.6,
            "digital_share": 64,
        },
    }

    regions = {
        "Центр": {"volume": 1.16, "ticket": 1.08, "digital": 4},
        "Северо-Запад": {"volume": 0.92, "ticket": 1.02, "digital": 2},
        "Поволжье": {"volume": 0.98, "ticket": 0.96, "digital": -1},
        "Урал": {"volume": 0.88, "ticket": 1.01, "digital": 1},
        "Сибирь": {"volume": 0.82, "ticket": 0.94, "digital": -2},
    }

    segments = {
        "Массовый": {"volume": 1.00, "ticket": 0.96, "risk": 0.6, "digital": 1},
        "Зарплатный": {"volume": 0.80, "ticket": 1.05, "risk": -0.3, "digital": 4},
        "Премиум": {"volume": 0.42, "ticket": 1.48, "risk": -0.9, "digital": -2},
    }

    channels = ["Офис", "Мобильное приложение", "Партнерский канал"]
    channel_weights = np.array([0.34, 0.48, 0.18])

    rows: list[dict[str, Any]] = []

    for month in months:
        month_wave = 1 + 0.08 * np.sin((month.month - 1) / 12 * 2 * np.pi)
        quarter_wave = 1 + 0.05 * np.cos((month.quarter - 1) / 4 * 2 * np.pi)

        for product_name, product_cfg in products.items():
            if product_name == "Ипотека":
                product_wave = 1 + 0.18 * np.sin((month.month - 3) / 12 * 2 * np.pi)
            elif product_name == "Кредитная карта":
                product_wave = 1 + 0.12 * np.sin((month.month + 2) / 12 * 2 * np.pi)
            elif product_name == "Вклад":
                product_wave = 1 + 0.10 * np.cos((month.month - 1) / 12 * 2 * np.pi)
            else:
                product_wave = 1 + 0.06 * np.sin((month.month + 4) / 12 * 2 * np.pi)

            for region_name, region_cfg in regions.items():
                for segment_name, segment_cfg in segments.items():
                    channel = rng.choice(channels, p=channel_weights)

                    marketing = (
                        product_cfg["base_marketing"]
                        * region_cfg["volume"]
                        * (0.85 + 0.30 * segment_cfg["volume"])
                        * month_wave
                        * (1 + rng.normal(0, 0.08))
                    )
                    marketing = max(650, marketing)

                    applications = (
                        product_cfg["base_applications"]
                        * region_cfg["volume"]
                        * segment_cfg["volume"]
                        * month_wave
                        * quarter_wave
                        * product_wave
                        + marketing * 0.06
                        + rng.normal(0, 28)
                    )
                    applications = max(40, round(applications))

                    digital_share = np.clip(
                        product_cfg["digital_share"]
                        + region_cfg["digital"]
                        + segment_cfg["digital"]
                        + rng.normal(0, 3.5),
                        18,
                        96,
                    )

                    approval_rate = np.clip(
                        product_cfg["approval"]
                        - segment_cfg["risk"] * 1.7
                        + (digital_share - 55) * 0.08
                        + rng.normal(0, 2.0),
                        34,
                        96,
                    )
                    approved = max(20, round(applications * approval_rate / 100))

                    issue_rate = np.clip(
                        product_cfg["issue"] + (digital_share - 60) * 0.05 + rng.normal(0, 1.8),
                        50,
                        96,
                    )
                    issued = max(10, round(approved * issue_rate / 100))

                    avg_ticket = (
                        product_cfg["ticket"]
                        * region_cfg["ticket"]
                        * segment_cfg["ticket"]
                        * (1 + rng.normal(0, 0.07))
                    )
                    avg_ticket = max(35, avg_ticket)

                    delinquency = np.clip(
                        product_cfg["delinquency"]
                        + segment_cfg["risk"]
                        + rng.normal(0, 0.35),
                        0.2,
                        8.5,
                    )

                    income_mln = (
                        issued * avg_ticket * 1000 * product_cfg["margin"] / 1_000_000
                        - marketing * 0.00055
                        - delinquency * 0.08
                        + rng.normal(0, 0.18)
                    )

                    rows.append(
                        {
                            "месяц": month.strftime("%Y-%m-%d"),
                            "продукт": product_name,
                            "регион": region_name,
                            "сегмент": segment_name,
                            "канал": channel,
                            "заявки": int(applications),
                            "одобрено": int(approved),
                            "выдано": int(issued),
                            "средний_чек_тыс_руб": round(float(avg_ticket), 1),
                            "расход_на_маркетинг_тыс_руб": round(float(marketing), 1),
                            "доля_цифровых_заявок_проц": round(float(digital_share), 1),
                            "просрочка_30дн_проц": round(float(delinquency), 2),
                            "чистый_доход_млн_руб": round(float(income_mln), 3),
                        }
                    )

    df = pd.DataFrame(rows).sort_values(["месяц", "продукт", "регион", "сегмент"]).reset_index(drop=True)

    regression_mask = np.ones(len(df), dtype=bool)
    for key, value in REGRESSION_SLICE.items():
        regression_mask &= df[key] == value

    regression_index = df.index[regression_mask]
    marketing = df.loc[regression_index, "расход_на_маркетинг_тыс_руб"].to_numpy()
    seasonal_adjustment = np.linspace(-18, 22, len(regression_index))
    issued = 302 + 0.052 * marketing + seasonal_adjustment + rng.normal(0, 10, len(regression_index))
    approved = issued / 0.79 + rng.normal(0, 7, len(regression_index))
    applications = approved / 0.72 + rng.normal(0, 10, len(regression_index))
    income = (
        issued * df.loc[regression_index, "средний_чек_тыс_руб"].to_numpy() * 1000 * 0.064 / 1_000_000
        - marketing * 0.00055
        - df.loc[regression_index, "просрочка_30дн_проц"].to_numpy() * 0.08
        + rng.normal(0, 0.08, len(regression_index))
    )
    df.loc[regression_index, "выдано"] = np.round(issued).astype(int)
    df.loc[regression_index, "одобрено"] = np.round(approved).astype(int)
    df.loc[regression_index, "заявки"] = np.round(applications).astype(int)
    df.loc[regression_index, "чистый_доход_млн_руб"] = np.round(income, 3)

    safe_index = df.index[~regression_mask].to_numpy()

    for column, share in {
        "средний_чек_тыс_руб": 0.055,
        "расход_на_маркетинг_тыс_руб": 0.037,
        "просрочка_30дн_проц": 0.028,
    }.items():
        count = int(len(df) * share)
        idx = rng.choice(safe_index, size=count, replace=False)
        df.loc[idx, column] = np.nan

    outlier_idx = rng.choice(safe_index, size=12, replace=False)
    df.loc[outlier_idx[:5], "расход_на_маркетинг_тыс_руб"] = (
        df.loc[outlier_idx[:5], "расход_на_маркетинг_тыс_руб"].fillna(0) * rng.uniform(2.3, 3.0, 5)
    ).round(1)
    df.loc[outlier_idx[5:9], "средний_чек_тыс_руб"] = (
        df.loc[outlier_idx[5:9], "средний_чек_тыс_руб"].fillna(0) * rng.uniform(1.6, 2.1, 4)
    ).round(1)
    df.loc[outlier_idx[9:], "чистый_доход_млн_руб"] = (
        df.loc[outlier_idx[9:], "чистый_доход_млн_руб"].fillna(0) * rng.uniform(1.8, 2.5, 3)
    ).round(3)

    return df


def load_dataset(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df["месяц"] = pd.to_datetime(df["месяц"])
    return df.sort_values(["месяц", "продукт", "регион", "сегмент"]).reset_index(drop=True)


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "products": int(df["продукт"].nunique()),
        "regions": int(df["регион"].nunique()),
        "segments": int(df["сегмент"].nunique()),
        "start_month": df["месяц"].min().strftime("%Y-%m"),
        "end_month": df["месяц"].max().strftime("%Y-%m"),
    }


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = (
        df.isna()
        .sum()
        .rename("число_пропусков")
        .to_frame()
        .assign(доля_пропусков_проц=lambda x: (x["число_пропусков"] / len(df) * 100).round(2))
        .reset_index()
        .rename(columns={"index": "поле"})
        .sort_values(["число_пропусков", "поле"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return missing


def count_outliers(df: pd.DataFrame, column: str) -> int:
    total = 0
    for _, group in df.groupby(["продукт", "сегмент"]):
        series = group[column].dropna()
        if len(series) < 8:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        total += int(((series < lower) | (series > upper)).sum())
    return total


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in OUTLIER_COLUMNS:
        rows.append({"поле": column, "подозрительных_наблюдений": count_outliers(df, column)})
    return pd.DataFrame(rows).sort_values("подозрительных_наблюдений", ascending=False).reset_index(drop=True)


def prepare_working_copy(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    for column in ["средний_чек_тыс_руб", "расход_на_маркетинг_тыс_руб", "просрочка_30дн_проц"]:
        working[column] = working.groupby("продукт")[column].transform(lambda s: s.fillna(s.median()))
        working[column] = working[column].fillna(working[column].median())
    return working


def build_pivots(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    income_pivot = pd.pivot_table(
        df,
        index="продукт",
        columns="регион",
        values="чистый_доход_млн_руб",
        aggfunc="sum",
    ).round(1)
    issue_pivot = pd.pivot_table(
        df,
        index="сегмент",
        columns="продукт",
        values="выдано",
        aggfunc="sum",
    ).round(0)
    return {"доход_по_продукту_и_региону": income_pivot, "выдачи_по_сегментам": issue_pivot}


def fit_regression(df: pd.DataFrame) -> dict[str, Any]:
    mask = np.ones(len(df), dtype=bool)
    for key, value in REGRESSION_SLICE.items():
        mask &= df[key] == value

    slice_df = df.loc[mask, ["месяц", "расход_на_маркетинг_тыс_руб", "выдано"]].copy()
    slice_df = slice_df.sort_values("месяц").reset_index(drop=True)

    q1 = slice_df["расход_на_маркетинг_тыс_руб"].quantile(0.25)
    q3 = slice_df["расход_на_маркетинг_тыс_руб"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    slice_df = slice_df.loc[slice_df["расход_на_маркетинг_тыс_руб"] <= upper].reset_index(drop=True)

    model = LinearRegression()
    model.fit(slice_df[["расход_на_маркетинг_тыс_руб"]], slice_df["выдано"])

    slice_df["прогноз"] = model.predict(slice_df[["расход_на_маркетинг_тыс_руб"]]).round(1)
    slice_df["остаток"] = (slice_df["выдано"] - slice_df["прогноз"]).round(1)
    residual_view = slice_df.tail(8).copy()

    forecast = FORECAST_SCENARIOS.copy()
    forecast_input = forecast.rename(columns={"Маркетинг, тыс. руб.": "расход_на_маркетинг_тыс_руб"})
    forecast["Прогноз, шт."] = model.predict(forecast_input[["расход_на_маркетинг_тыс_руб"]]).round(0).astype(int)

    limits = pd.DataFrame(
        [
            {
                "Что ограничивает": "Объем выборки",
                "Почему важно": "Модель обучена только на 24 месяцах одного среза и чувствительна к единичным всплескам.",
            },
            {
                "Что ограничивает": "Сезонность",
                "Почему важно": "Месячные акции и календарные пики могут смещать линию и завышать эффект маркетинга.",
            },
            {
                "Что ограничивает": "Переносимость",
                "Почему важно": "Коэффициент нельзя без проверки переносить на ипотеку, вклады и другие регионы.",
            },
        ]
    )

    return {
        "history": slice_df,
        "residual_view": residual_view,
        "model": model,
        "forecast": forecast,
        "limits": limits,
        "slope_per_1000": round(float(model.coef_[0]) * 1000, 1),
        "intercept": round(float(model.intercept_), 1),
        "r2_value": round(float(r2_score(slice_df["выдано"], slice_df["прогноз"])), 3),
        "mae_value": round(float(mean_absolute_error(slice_df["выдано"], slice_df["прогноз"])), 1),
        "sample_size": int(len(slice_df)),
        "slice_label": REGRESSION_SLICE.copy(),
    }


def generate_management_conclusions(df: pd.DataFrame, regression: dict[str, Any]) -> pd.DataFrame:
    miss = missing_summary(df)
    outliers = outlier_summary(df)
    pivots = build_pivots(df)

    income_pivot = pivots["доход_по_продукту_и_региону"]
    top_pair = income_pivot.stack().sort_values(ascending=False).index[0]
    top_pair_value = float(income_pivot.loc[top_pair[0], top_pair[1]])
    total_income = float(df["чистый_доход_млн_руб"].sum())

    segment_risk = (
        df.groupby("сегмент", as_index=False)["просрочка_30дн_проц"]
        .mean()
        .sort_values("просрочка_30дн_проц", ascending=False)
        .iloc[0]
    )

    issue_share = (
        df.groupby("продукт", as_index=False)["выдано"]
        .sum()
        .sort_values("выдано", ascending=False)
        .iloc[0]
    )
    total_issued = float(df["выдано"].sum())

    conclusions = pd.DataFrame(
        [
            {
                "Приоритет": "Высокий",
                "Наблюдение": f"Связка «{top_pair[0]} / {top_pair[1]}» дает самый большой накопленный доход.",
                "Основание": f"{top_pair_value:.1f} млн руб., это {(top_pair_value / total_income * 100):.1f}% суммарного дохода набора.",
                "Действие": "Разобрать практики продаж этого направления и проверить, какие из них можно перенести на соседние регионы.",
            },
            {
                "Приоритет": "Высокий",
                "Наблюдение": f"Сегмент «{segment_risk['сегмент']}» имеет максимальный средний уровень просрочки.",
                "Основание": f"Средняя просрочка 30+ дней: {segment_risk['просрочка_30дн_проц']:.2f}%.",
                "Действие": "Развести продажи и риск: отдельно уточнить лимиты, скоринг и сценарии повторного предложения.",
            },
            {
                "Приоритет": "Средний",
                "Наблюдение": "В срезе для регрессии маркетинговый бюджет заметно связан с выдачами.",
                "Основание": (
                    f"Дополнительные 1 000 тыс. руб. маркетинга связаны в среднем с "
                    f"+{regression['slope_per_1000']:.1f} выдачами; R² по срезу = {regression['r2_value']:.2f}."
                ),
                "Действие": "Использовать модель только как быстрый ориентир для сценарного плана, а итоговый бюджет подтверждать расширенной проверкой.",
            },
            {
                "Приоритет": "Средний",
                "Наблюдение": f"Наиболее массовым продуктом остается «{issue_share['продукт']}».",
                "Основание": f"{int(issue_share['выдано']):,} выдач, это {(issue_share['выдано'] / total_issued * 100):.1f}% всех выдач.".replace(",", " "),
                "Действие": "По этому продукту особенно важно держать качество данных и отдельную витрину для регулярного мониторинга.",
            },
            {
                "Приоритет": "Средний",
                "Наблюдение": f"Больше всего пропусков и аномалий у поля «{miss.iloc[0]['поле']}».",
                "Основание": (
                    f"Пропусков: {int(miss.iloc[0]['число_пропусков'])}; "
                    f"подозрительных наблюдений в ключевом поле-лидере: {int(outliers.iloc[0]['подозрительных_наблюдений'])}."
                ),
                "Действие": "До регулярного прогноза закрепить правило заполнения поля и контроль всплесков перед расчётом средних.",
            },
        ]
    )

    return conclusions


def build_case_summary(df: pd.DataFrame, regression: dict[str, Any], conclusions: pd.DataFrame) -> dict[str, Any]:
    miss = missing_summary(df)
    outliers = outlier_summary(df)
    pivots = build_pivots(df)

    income_pivot = pivots["доход_по_продукту_и_региону"]
    top_pair = income_pivot.stack().sort_values(ascending=False).index[0]
    segment_leader = (
        df.groupby("сегмент", as_index=False)["выдано"].sum().sort_values("выдано", ascending=False).iloc[0]
    )

    return {
        **dataset_summary(df),
        "top_missing_field": str(miss.iloc[0]["поле"]),
        "top_missing_share": float(miss.iloc[0]["доля_пропусков_проц"]),
        "marketing_outliers": int(
            outliers.loc[outliers["поле"] == "расход_на_маркетинг_тыс_руб", "подозрительных_наблюдений"].iloc[0]
        ),
        "income_outliers": int(
            outliers.loc[outliers["поле"] == "чистый_доход_млн_руб", "подозрительных_наблюдений"].iloc[0]
        ),
        "top_income_product": top_pair[0],
        "top_income_region": top_pair[1],
        "top_segment": str(segment_leader["сегмент"]),
        "regression_slope_per_1000": float(regression["slope_per_1000"]),
        "regression_r2": float(regression["r2_value"]),
        "regression_mae": float(regression["mae_value"]),
        "forecast_base": int(
            regression["forecast"].loc[regression["forecast"]["Сценарий"] == "Базовый", "Прогноз, шт."].iloc[0]
        ),
        "forecast_high": int(
            regression["forecast"].loc[regression["forecast"]["Сценарий"] == "Усиленный", "Прогноз, шт."].iloc[0]
        ),
        "first_conclusion": str(conclusions.iloc[0]["Наблюдение"]),
    }


def _save_figure(fig: plt.Figure, output_path: Path | None = None) -> plt.Figure:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180)
    return fig


def _format_heatmap_labels(value: float) -> str:
    return f"{value:.1f}"


def build_overview_figure(df: pd.DataFrame, output_path: Path | None = None) -> plt.Figure:
    summary = dataset_summary(df)
    monthly = (
        df.groupby(["месяц", "продукт"], as_index=False)["чистый_доход_млн_руб"]
        .sum()
        .pivot(index="месяц", columns="продукт", values="чистый_доход_млн_руб")
    )
    product_totals = df.groupby("продукт", as_index=False)["выдано"].sum().sort_values("выдано", ascending=False)

    fig = plt.figure(figsize=(13.2, 7.2))
    gs = GridSpec(2, 2, width_ratios=[2.3, 1.1], height_ratios=[2.2, 1.2], figure=fig)
    ax_line = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, 1])

    colors = [PALETTE["green"], PALETTE["blue"], PALETTE["orange"], PALETTE["gray"]]
    for color, column in zip(colors, monthly.columns):
        ax_line.plot(monthly.index, monthly[column], linewidth=2.4, color=color, label=column)
    ax_line.set_title("Динамика чистого дохода по продуктам")
    ax_line.set_ylabel("млн руб.")
    ax_line.grid(True, axis="y")
    ax_line.legend(loc="upper left", frameon=False)

    ax_bar.barh(product_totals["продукт"], product_totals["выдано"], color=PALETTE["green"])
    ax_bar.set_title("Выдачи за весь период")
    ax_bar.set_xlabel("количество")
    ax_bar.grid(True, axis="x")

    ax_table.axis("off")
    table_rows = pd.DataFrame(
        [
            ["Строк", f"{summary['rows']:,}".replace(",", " ")],
            ["Полей", str(summary["columns"])],
            ["Период", f"{summary['start_month']} - {summary['end_month']}"],
            ["Продуктов", str(summary["products"])],
            ["Регионов", str(summary["regions"])],
            ["Сегментов", str(summary["segments"])],
        ],
        columns=["Параметр", "Значение"],
    )
    table = ax_table.table(
        cellText=table_rows.values,
        colLabels=table_rows.columns,
        colColours=[PALETTE["green_light"], PALETTE["green_light"]],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])

    fig.suptitle("Учебный набор данных по банковским продуктам", x=0.02, y=1.02, ha="left", fontsize=16, fontweight="bold")
    return _save_figure(fig, output_path)


def build_missing_and_outliers_figure(df: pd.DataFrame, output_path: Path | None = None) -> plt.Figure:
    miss = missing_summary(df)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.2), gridspec_kw={"width_ratios": [1.0, 1.2]})

    miss_plot = miss.loc[miss["число_пропусков"] > 0].copy()
    axes[0].barh(miss_plot["поле"], miss_plot["доля_пропусков_проц"], color=PALETTE["blue"])
    axes[0].set_title("Доля пропусков по полям")
    axes[0].set_xlabel("% строк")
    axes[0].invert_yaxis()
    axes[0].grid(True, axis="x")

    box_data = [df[column].dropna() for column in OUTLIER_COLUMNS]
    axes[1].boxplot(
        box_data,
        patch_artist=True,
        boxprops={"facecolor": PALETTE["green_light"], "edgecolor": PALETTE["green_dark"]},
        medianprops={"color": PALETTE["green_dark"], "linewidth": 2},
        whiskerprops={"color": PALETTE["gray"]},
        capprops={"color": PALETTE["gray"]},
        flierprops={"markerfacecolor": PALETTE["red"], "markeredgecolor": PALETTE["red"], "markersize": 5, "alpha": 0.65},
    )
    axes[1].set_xticklabels(
        [
            "Маркетинг,\nтыс. руб.",
            "Средний чек,\nтыс. руб.",
            "Чистый доход,\nмлн руб.",
        ]
    )
    axes[1].set_title("Подозрительные выбросы в ключевых метриках")
    axes[1].grid(True, axis="y")

    fig.suptitle("Первичный осмотр: пропуски и выбросы", x=0.02, y=1.02, ha="left", fontsize=16, fontweight="bold")
    return _save_figure(fig, output_path)


def build_cuts_figure(df: pd.DataFrame, output_path: Path | None = None) -> plt.Figure:
    pivots = build_pivots(df)
    income_pivot = pivots["доход_по_продукту_и_региону"]
    issue_pivot = pivots["выдачи_по_сегментам"]

    fig = plt.figure(figsize=(13.2, 7.2))
    gs = GridSpec(1, 2, width_ratios=[1.25, 1.0], figure=fig)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])

    image = ax_heat.imshow(income_pivot.values, cmap="Greens")
    ax_heat.set_xticks(np.arange(len(income_pivot.columns)))
    ax_heat.set_xticklabels(income_pivot.columns, rotation=20, ha="right")
    ax_heat.set_yticks(np.arange(len(income_pivot.index)))
    ax_heat.set_yticklabels(income_pivot.index)
    ax_heat.set_title("Сводная таблица: чистый доход, млн руб.")
    for i in range(income_pivot.shape[0]):
        for j in range(income_pivot.shape[1]):
            ax_heat.text(j, i, _format_heatmap_labels(income_pivot.iloc[i, j]), ha="center", va="center", color=PALETTE["text"], fontsize=10)
    fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.03)

    issue_pivot.T.plot(kind="bar", ax=ax_bar, color=[PALETTE["green"], PALETTE["blue"], PALETTE["orange"]], width=0.78)
    ax_bar.set_title("Выдачи по продуктам и сегментам")
    ax_bar.set_ylabel("количество")
    ax_bar.grid(True, axis="y")
    ax_bar.legend(title="Сегмент", frameon=False)
    ax_bar.tick_params(axis="x", rotation=20)

    fig.suptitle("Разрезы по продуктам, регионам и сегментам", x=0.02, y=1.02, ha="left", fontsize=16, fontweight="bold")
    return _save_figure(fig, output_path)


def build_regression_figure(regression: dict[str, Any], output_path: Path | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = GridSpec(1, 2, width_ratios=[1.45, 0.95], figure=fig)
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1])

    history = regression["history"]
    earlier = history.iloc[:-6]
    recent = history.tail(6)
    model = regression["model"]
    all_x = regression["history"]["расход_на_маркетинг_тыс_руб"]
    x_line = np.linspace(all_x.min() * 0.95, all_x.max() * 1.05, 100)
    x_line_df = pd.DataFrame({"расход_на_маркетинг_тыс_руб": x_line})
    y_line = model.predict(x_line_df)

    ax_plot.scatter(
        earlier["расход_на_маркетинг_тыс_руб"],
        earlier["выдано"],
        color=PALETTE["green"],
        s=65,
        label="Более ранние месяцы",
    )
    ax_plot.scatter(
        recent["расход_на_маркетинг_тыс_руб"],
        recent["выдано"],
        color=PALETTE["orange"],
        s=65,
        label="Последние месяцы",
    )
    ax_plot.plot(x_line, y_line, color=PALETTE["blue"], linewidth=2.5, label="Линия регрессии")
    ax_plot.set_title("Маркетинг и выдачи в выбранном срезе")
    ax_plot.set_xlabel("Маркетинг, тыс. руб.")
    ax_plot.set_ylabel("Выдачи, шт.")
    ax_plot.grid(True)
    ax_plot.legend(frameon=False)

    ax_table.axis("off")
    forecast = regression["forecast"].copy()
    table = ax_table.table(
        cellText=forecast.values,
        colLabels=forecast.columns,
        colColours=[PALETTE["blue_light"]] * len(forecast.columns),
        cellLoc="left",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.65)
    for (_, _), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])

    ax_table.text(
        0.02,
        0.38,
        (
            f"Срез: {regression['slice_label']['продукт']}, "
            f"{regression['slice_label']['регион']}, "
            f"{regression['slice_label']['сегмент']}\n\n"
            f"Каждые дополнительные 1 000 тыс. руб.\n"
            f"связаны в среднем с +{regression['slope_per_1000']:.1f} выдачами."
        ),
        fontsize=11,
        color=PALETTE["text"],
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": PALETTE["green_light"],
            "edgecolor": PALETTE["border"],
        },
    )

    fig.suptitle("Простая линейная регрессия и сценарный прогноз", x=0.02, y=1.02, ha="left", fontsize=16, fontweight="bold")
    return _save_figure(fig, output_path)


def build_interpretation_figure(regression: dict[str, Any], output_path: Path | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = GridSpec(2, 2, width_ratios=[1.05, 1.05], height_ratios=[1.35, 1.0], figure=fig)
    ax_res = fig.add_subplot(gs[0, :])
    ax_metrics = fig.add_subplot(gs[1, 0])
    ax_limits = fig.add_subplot(gs[1, 1])

    residual_view = regression["residual_view"].copy()
    labels = [pd.Timestamp(value).strftime("%Y-%m") for value in residual_view["месяц"]]
    ax_res.bar(
        labels,
        residual_view["остаток"],
        color=[PALETTE["green"] if value >= 0 else PALETTE["red"] for value in residual_view["остаток"]],
    )
    ax_res.axhline(0, color=PALETTE["gray"], linewidth=1.2)
    ax_res.set_title("Остатки по последним месяцам")
    ax_res.set_ylabel("факт - прогноз")
    ax_res.grid(True, axis="y")

    metrics_df = pd.DataFrame(
        [
            ["Коэффициент на 1 000 тыс. руб.", f"+{regression['slope_per_1000']:.1f} выдач"],
            ["R² по выбранному срезу", f"{regression['r2_value']:.2f}"],
            ["Средняя абсолютная ошибка", f"{regression['mae_value']:.1f} выдач"],
            ["Размер среза", f"{regression['sample_size']} месяцев"],
        ],
        columns=["Метрика", "Значение"],
    )

    ax_metrics.axis("off")
    metrics_table = ax_metrics.table(
        cellText=metrics_df.values,
        colLabels=metrics_df.columns,
        colColours=[PALETTE["green_light"], PALETTE["green_light"]],
        cellLoc="left",
        loc="center",
    )
    metrics_table.auto_set_font_size(False)
    metrics_table.set_fontsize(10)
    metrics_table.scale(1.0, 1.6)
    for (_, _), cell in metrics_table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])

    ax_limits.axis("off")
    limits_view = regression["limits"].copy()
    limits_view["Что ограничивает"] = limits_view["Что ограничивает"].map(lambda value: fill(str(value), width=16))
    limits_view["Почему важно"] = limits_view["Почему важно"].map(lambda value: fill(str(value), width=32))
    limits_table = ax_limits.table(
        cellText=limits_view.values,
        colLabels=limits_view.columns,
        colColours=[PALETTE["orange_light"], PALETTE["orange_light"]],
        cellLoc="left",
        loc="center",
    )
    limits_table.auto_set_font_size(False)
    limits_table.set_fontsize(9.2)
    limits_table.scale(1.0, 1.7)
    for (_, _), cell in limits_table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])

    fig.suptitle("Интерпретация результата и границы применимости", x=0.02, y=1.02, ha="left", fontsize=16, fontweight="bold")
    return _save_figure(fig, output_path)


def build_conclusions_figure(conclusions: pd.DataFrame, output_path: Path | None = None) -> plt.Figure:
    compact = conclusions.head(3)[["Приоритет", "Наблюдение", "Действие"]].copy()
    compact["Наблюдение"] = compact["Наблюдение"].map(lambda value: fill(str(value), width=32))
    compact["Действие"] = compact["Действие"].map(lambda value: fill(str(value), width=34))

    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.axis("off")

    table = ax.table(
        cellText=compact.values,
        colLabels=compact.columns,
        colColours=[PALETTE["green_light"]] * len(compact.columns),
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.6)
    table.scale(1.08, 2.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(PALETTE["border"])
        if row == 0:
            cell.set_text_props(weight="bold", color=PALETTE["text"])
        if row > 0 and col == 0:
            priority = compact.iloc[row - 1, 0]
            if priority == "Высокий":
                cell.set_facecolor(PALETTE["red_light"])
            elif priority == "Средний":
                cell.set_facecolor(PALETTE["orange_light"])
            else:
                cell.set_facecolor(PALETTE["gray_light"])

    fig.suptitle("Автоматически сформированные управленческие выводы", x=0.02, y=1.02, ha="left", fontsize=16, fontweight="bold")
    return _save_figure(fig, output_path)
