from __future__ import annotations

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import pandas as pd

from case_logic import (
    build_case_summary,
    build_conclusions_figure,
    build_cuts_figure,
    build_interpretation_figure,
    build_missing_and_outliers_figure,
    build_overview_figure,
    build_regression_figure,
    fit_regression,
    generate_management_conclusions,
    load_dataset,
    missing_summary,
    outlier_summary,
    prepare_working_copy,
    synthesize_dataset,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
NOTEBOOK_PATH = BASE_DIR / "bank_products_basic_methods_walkthrough.ipynb"
DATA_PATH = DATA_DIR / "bank_products_sales.csv"
CONCLUSIONS_PATH = DATA_DIR / "management_conclusions.csv"
SUMMARY_PATH = ARTIFACTS_DIR / "case_summary.json"


def write_notebook() -> None:
    notebook = {
        "cells": [
            markdown_cell(
                """
                # Практика 4. Базовые методы анализа данных

                Эта тетрадь проходит полный путь по учебному набору данных о банковских продуктах:
                первичный осмотр, пропуски и выбросы, сводные таблицы, линейная регрессия,
                интерпретация результата и автоматические управленческие выводы.
                """
            ),
            markdown_cell(
                """
                ## Контекст задачи

                Нам нужен короткий и наглядный учебный пример для руководителя.
                На одном наборе данных смотрим продажи банковских продуктов по месяцам,
                регионам и сегментам, а затем проверяем, можно ли использовать простую
                линейную регрессию для быстрого прогноза выдач.
                """
            ),
            code_cell(
                """
                from pathlib import Path
                import sys

                import pandas as pd
                from IPython.display import Image, display

                BASE_DIR = Path.cwd()
                if not (BASE_DIR / "data").exists():
                    for root in [Path.cwd(), *Path.cwd().parents]:
                        if (root / "data").exists() and (root / "case_logic.py").exists():
                            BASE_DIR = root
                            break
                        candidate = root / "bank_products_basic_methods"
                        if (candidate / "data").exists() and (candidate / "case_logic.py").exists():
                            BASE_DIR = candidate
                            break
                    else:
                        raise FileNotFoundError("Не удалось найти папку кейса с данными и логикой.")

                sys.path.append(str(BASE_DIR))

                from case_logic import (
                    build_conclusions_figure,
                    build_cuts_figure,
                    build_interpretation_figure,
                    build_missing_and_outliers_figure,
                    build_overview_figure,
                    build_regression_figure,
                    fit_regression,
                    generate_management_conclusions,
                    load_dataset,
                    missing_summary,
                    outlier_summary,
                    prepare_working_copy,
                )

                DATA_PATH = BASE_DIR / "data" / "bank_products_sales.csv"
                ARTIFACTS_DIR = BASE_DIR / "artifacts"
                """
            ),
            code_cell(
                """
                df = load_dataset(DATA_PATH)
                print(f"Файл: {DATA_PATH}")
                print(f"Строк: {len(df):,}".replace(",", " "))
                print(f"Колонок: {df.shape[1]}")
                display(df.head())
                """
            ),
            markdown_cell(
                """
                ## Шаг 1. Посмотрим на структуру набора данных

                Сначала убеждаемся, что у нас есть все ключевые измерения:
                месяц, продукт, регион, сегмент и основные показатели результата.
                """
            ),
            code_cell(
                """
                overview_path = ARTIFACTS_DIR / "01_dataset_overview.png"
                build_overview_figure(df, overview_path)
                display(Image(filename=str(overview_path)))
                """
            ),
            markdown_cell(
                """
                ## Шаг 2. Проверяем пропуски и выбросы

                До построения разрезов и прогноза нужно понять, какие поля заполнены не полностью
                и где есть значения, которые могут искажать средние.
                """
            ),
            code_cell(
                """
                missing = missing_summary(df)
                outliers = outlier_summary(df)

                display(missing.loc[missing["число_пропусков"] > 0])
                display(outliers)

                quality_path = ARTIFACTS_DIR / "02_missing_and_outliers.png"
                build_missing_and_outliers_figure(df, quality_path)
                display(Image(filename=str(quality_path)))
                """
            ),
            markdown_cell(
                """
                ## Шаг 3. Готовим рабочую копию

                Для упражнения достаточно простой стратегии:
                пропуски восстанавливаем медианой внутри продукта и продолжаем анализ.
                """
            ),
            code_cell(
                """
                working_df = prepare_working_copy(df)
                display(working_df.head())
                """
            ),
            markdown_cell(
                """
                ## Шаг 4. Строим сводные таблицы и разрезы

                Теперь можно посмотреть, какие продукты, регионы и сегменты дают наибольший доход
                и где сосредоточен объем выдач.
                """
            ),
            code_cell(
                """
                cuts_path = ARTIFACTS_DIR / "03_cuts_and_pivots.png"
                build_cuts_figure(working_df, cuts_path)
                display(Image(filename=str(cuts_path)))

                income_pivot = pd.pivot_table(
                    working_df,
                    index="продукт",
                    columns="регион",
                    values="чистый_доход_млн_руб",
                    aggfunc="sum",
                ).round(1)
                display(income_pivot)
                """
            ),
            markdown_cell(
                """
                ## Шаг 5. Строим простую линейную регрессию

                Для прогноза берем один конкретный срез:
                кредитные карты в регионе «Центр» для массового сегмента.
                Признак-множитель: расходы на маркетинг.
                """
            ),
            code_cell(
                """
                regression = fit_regression(working_df)
                regression_path = ARTIFACTS_DIR / "04_regression_and_forecast.png"
                build_regression_figure(regression, regression_path)
                display(Image(filename=str(regression_path)))
                display(regression["forecast"])
                """
            ),
            markdown_cell(
                """
                ## Шаг 6. Интерпретируем модель и проговариваем границы применимости

                Даже простая модель полезна только тогда, когда мы видим ошибку прогноза,
                остатки и ограничения по выборке.
                """
            ),
            code_cell(
                """
                interpretation_path = ARTIFACTS_DIR / "05_interpretation_and_limits.png"
                build_interpretation_figure(regression, interpretation_path)
                display(Image(filename=str(interpretation_path)))
                display(regression["limits"])
                """
            ),
            markdown_cell(
                """
                ## Шаг 7. Формируем управленческие выводы автоматически

                На последнем шаге переводим найденные сигналы в короткий список действий:
                что усиливать, что проверять в данных и где нужна отдельная модель.
                """
            ),
            code_cell(
                """
                conclusions = generate_management_conclusions(df, regression)
                display(conclusions)

                conclusions_path = ARTIFACTS_DIR / "06_management_conclusions.png"
                build_conclusions_figure(conclusions, conclusions_path)
                display(Image(filename=str(conclusions_path)))
                """
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.9",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code_cell(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines(text)}


def lines(text: str) -> list[str]:
    return [line + "\n" for line in dedent(text).strip().splitlines()]


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = synthesize_dataset()
    df.to_csv(DATA_PATH, index=False, encoding="utf-8")

    loaded_df = load_dataset(DATA_PATH)
    working_df = prepare_working_copy(loaded_df)
    regression = fit_regression(working_df)
    conclusions = generate_management_conclusions(loaded_df, regression)
    summary = build_case_summary(loaded_df, regression, conclusions)

    build_overview_figure(loaded_df, ARTIFACTS_DIR / "01_dataset_overview.png")
    build_missing_and_outliers_figure(loaded_df, ARTIFACTS_DIR / "02_missing_and_outliers.png")
    build_cuts_figure(working_df, ARTIFACTS_DIR / "03_cuts_and_pivots.png")
    build_regression_figure(regression, ARTIFACTS_DIR / "04_regression_and_forecast.png")
    build_interpretation_figure(regression, ARTIFACTS_DIR / "05_interpretation_and_limits.png")
    build_conclusions_figure(conclusions, ARTIFACTS_DIR / "06_management_conclusions.png")

    missing_summary(loaded_df).to_csv(DATA_DIR / "missing_summary.csv", index=False, encoding="utf-8")
    outlier_summary(loaded_df).to_csv(DATA_DIR / "outlier_summary.csv", index=False, encoding="utf-8")
    conclusions.to_csv(CONCLUSIONS_PATH, index=False, encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_notebook()
    print(f"Готово: {DATA_PATH}")
    print(f"Готово: {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
