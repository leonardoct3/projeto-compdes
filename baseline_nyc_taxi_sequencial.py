# baseline_nyc_taxi_sequencial.py
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


PICKUP_CANDIDATES = ["tpep_pickup_datetime", "pickup_datetime", "pickup_at"]
DROPOFF_CANDIDATES = ["tpep_dropoff_datetime", "dropoff_datetime", "dropoff_at"]
DISTANCE_CANDIDATES = ["trip_distance", "distance"]
FARE_CANDIDATES = ["fare_amount", "total_amount", "fare"]
LOCATION_CANDIDATES = ["PULocationID", "pickup_location_id", "location_id"]

INPUT_DIR = "data/"
OUTPUT_DIR = "output/"


def pick_column(existing_columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in existing_columns:
            return col
    return None


def process_parquet_sequential(
    parquet_path: str,
    batch_size: int = 250_000,
    output_json: str | None = "results_baseline.json",
) -> dict:
    dataset = ds.dataset(parquet_path, format="parquet")
    schema_cols = list(dataset.schema.names)

    pickup_col = pick_column(schema_cols, PICKUP_CANDIDATES)
    dropoff_col = pick_column(schema_cols, DROPOFF_CANDIDATES)
    distance_col = pick_column(schema_cols, DISTANCE_CANDIDATES)
    fare_col = pick_column(schema_cols, FARE_CANDIDATES)
    location_col = pick_column(schema_cols, LOCATION_CANDIDATES)

    required_cols = [c for c in [pickup_col, dropoff_col, distance_col, fare_col, location_col] if c]

    if pickup_col is None or dropoff_col is None or distance_col is None or fare_col is None:
        raise ValueError(
            "Não encontrei colunas essenciais no parquet. "
            "Esperado algo como pickup/dropoff datetime, trip_distance e fare_amount."
        )

    total_rides = 0
    sum_distance = 0.0
    sum_duration_min = 0.0
    sum_fare = 0.0
    rides_by_hour = defaultdict(int)
    rides_by_location = defaultdict(int)

    start = time.perf_counter()

    scanner = dataset.scanner(columns=required_cols, batch_size=batch_size)

    for batch in scanner.to_batches():
        df = batch.to_pandas()

        # Conversões e limpeza
        df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
        df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")

        df = df.dropna(subset=[pickup_col, dropoff_col, distance_col, fare_col])

        df = df[
            (df[distance_col] >= 0)
            & (df[fare_col] >= 0)
            & (df[dropoff_col] >= df[pickup_col])
        ]

        if df.empty:
            continue

        duration_min = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60.0

        # métricas globais
        total_rides += len(df)
        sum_distance += float(df[distance_col].sum())
        sum_duration_min += float(duration_min.sum())
        sum_fare += float(df[fare_col].sum())

        # distribuição por horário
        hour_counts = df[pickup_col].dt.hour.value_counts()
        for hour, count in hour_counts.items():
            rides_by_hour[int(hour)] += int(count)

        # distribuição por região, se existir
        if location_col is not None:
            loc_counts = df[location_col].value_counts()
            for loc, count in loc_counts.items():
                rides_by_location[int(loc)] += int(count)

    elapsed = time.perf_counter() - start

    if total_rides == 0:
        raise ValueError("Nenhuma linha válida encontrada após a limpeza.")

    results = {
        "total_rides": total_rides,
        "avg_distance": sum_distance / total_rides,
        "avg_duration_min": sum_duration_min / total_rides,
        "avg_fare": sum_fare / total_rides,
        "elapsed_seconds": elapsed,
        "rides_by_hour": dict(sorted(rides_by_hour.items())),
        "rides_by_location_top20": dict(
            sorted(rides_by_location.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
    }

    if output_json:
        Path(output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    return results


def main() -> None:
    parquet_path = INPUT_DIR + "yellow_tripdata_2025-01.parquet"

    results = process_parquet_sequential(
        parquet_path=parquet_path,
        batch_size=250_000,
        output_json=OUTPUT_DIR + "results_baseline.json",
    )

    print("\n=== RESULTADOS BASELINE SEQUENCIAL ===")
    print(f"Total de corridas: {results['total_rides']}")
    print(f"Distância média: {results['avg_distance']:.3f}")
    print(f"Duração média (min): {results['avg_duration_min']:.3f}")
    print(f"Valor médio: {results['avg_fare']:.3f}")
    print(f"Tempo total: {results['elapsed_seconds']:.2f} s")

    hour_df = pd.DataFrame(
        list(results["rides_by_hour"].items()),
        columns=["pickup_hour", "rides"],
    ).sort_values("pickup_hour")

    hour_df.to_csv(OUTPUT_DIR + "rides_by_hour.csv", index=False)


if __name__ == "__main__":
    main()