# Python-кейс: кластеризация клиентов банка

## Что внутри

- `data/bank_client_behavior.csv` - синтетический датасет по клиентам.
- `data/cluster_profiles.csv` - агрегированные портреты кластеров.
- `data/target_clients_for_campaign.csv` - shortlist для коммуникации.
- `artifacts/cluster_report.html` - локальный HTML-отчет.
- `artifacts/cluster_scatter_2d.png` - готовая 2D-визуализация для слайдов.
- `artifacts/cluster_scatter_3d.png` - готовая 3D-визуализация для слайдов.
- `artifacts/cluster_profiles.png` - сравнение портретов кластеров.
- `client_clustering_walkthrough.ipynb` - пошаговая тетрадь для показа в IDE.
- `open_notebook.sh` - быстрый запуск тетради через `jupyter lab`, если он установлен.

## Как показывать на занятии

1. Запустите `./open_notebook.sh` или просто откройте `client_clustering_walkthrough.ipynb` в VS Code / PyCharm / Jupyter.
2. Последовательно запускайте ячейки:
   - загрузка данных;
   - отбор признаков;
   - нормализация;
   - подбор числа кластеров;
   - K-means;
   - 2D и 3D визуализация;
   - выбор целевого кластера;
   - выгрузка списка клиентов для контакта.
3. Для “готового результата” можно параллельно открыть `./run_cluster_case.sh` и показать HTML-отчет.

## Идея кейса

- Простая сегментация по обороту или просрочке здесь недостаточна.
- Часть клиентов тратит много и платит дисциплинированно, но почти не приносит маржу.
- K-means выделяет такой кластер и позволяет выбрать правильный следующий оффер.

## Текущая рекомендация

- Целевой кластер: `0`
- Кампания: премиальный travel-пакет или инвестиционный оффер

Абсолютные пути:

- Dataset: `data/bank_client_behavior.csv`
- Profiles: `data/cluster_profiles.csv`
- Leads: `data/target_clients_for_campaign.csv`
- Notebook: `client_clustering_walkthrough.ipynb`
- Report: `artifacts/cluster_report.html`
