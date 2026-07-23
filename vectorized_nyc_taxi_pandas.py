# vectorized_nyc_taxi_pandas.py
"""
Versão intermediária de otimização: mesmo hardware, mesmo único core, mesmo
esquema de leitura em batches do baseline (baseline_nyc_taxi_sequencial.py),
mas substituindo o loop Python linha-a-linha (itertuples + pd.isna escalar)
por operações vetorizadas do pandas/numpy dentro de cada batch.

O objetivo desta versão é isolar o ganho de "vetorizar" do ganho de
"paralelizar em múltiplos cores" (esse segundo vem só na versão Dask).
"""
from __future__ import annotations

import cProfile
import json
import pstats
import time
from collections import defaultdict
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


PICKUP_CANDIDATES = ["tpep_pickup_datetime", "pickup_datetime", "pickup_at"]
DROPOFF_CANDIDATES = ["tpep_dropoff_datetime", "dropoff_datetime", "dropoff_at"]
DISTANCE_CANDIDATES = ["trip_distance", "distance"]
FARE_CANDIDATES = ["fare_amount", "total_amount", "fare"]
LOCATION_CANDIDATES = ["PULocationID", "pickup_location_id", "location_id"]

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("output")

# Ajuste aqui para a pasta que contém TODOS os Parquets do ano.
# Exemplo esperado:
# data/yellow_tripdata_2025/
PARQUET_PATH = INPUT_DIR / "yellow_tripdata_2025"

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_column(existing_columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in existing_columns:
            return col
    return None


def process_parquet_vectorized(
    parquet_path: str | Path,
    batch_size: int = 250_000,
    output_json: str | Path | None = OUTPUT_DIR / "results_vectorized.json",
    stage_timing_json: str | Path | None = OUTPUT_DIR / "stage_timings_vectorized.json",
) -> dict:
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Caminho não encontrado: {parquet_path}. "
            "Aponte para a pasta do ano inteiro contendo os arquivos Parquet."
        )

    dataset = ds.dataset(str(parquet_path), format="parquet")
    schema_cols = list(dataset.schema.names)

    pickup_col = pick_column(schema_cols, PICKUP_CANDIDATES)
    dropoff_col = pick_column(schema_cols, DROPOFF_CANDIDATES)
    distance_col = pick_column(schema_cols, DISTANCE_CANDIDATES)
    fare_col = pick_column(schema_cols, FARE_CANDIDATES)
    location_col = pick_column(schema_cols, LOCATION_CANDIDATES)

    required_cols = [
        c for c in [pickup_col, dropoff_col, distance_col, fare_col, location_col] if c
    ]

    if pickup_col is None or dropoff_col is None or distance_col is None or fare_col is None:
        raise ValueError(
            "Não encontrei colunas essenciais no parquet. "
            "Esperado algo como pickup/dropoff datetime, trip_distance e fare_amount."
        )

    total_rides = 0
    rows_seen = 0
    rows_valid = 0

    sum_distance = 0.0
    sum_duration_min = 0.0
    sum_fare = 0.0

    rides_by_hour: dict[int, int] = defaultdict(int)
    rides_by_location: dict[int, int] = defaultdict(int)

    stage_times = defaultdict(float)
    batch_count = 0

    overall_start = time.perf_counter()
    scanner = dataset.scanner(columns=required_cols, batch_size=batch_size)

    for batch in scanner.to_batches():
        batch_count += 1

        t0 = time.perf_counter()
        df = batch.to_pandas()
        stage_times["to_pandas"] += time.perf_counter() - t0

        rows_seen += len(df)

        t0 = time.perf_counter()
        df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
        df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")
        stage_times["datetime_conversion"] += time.perf_counter() - t0

        # ---- Bloco vetorizado: substitui o loop `for row in df.itertuples()`
        # do baseline por máscaras booleanas e operações em array inteiro,
        # que rodam em código C do numpy/pandas em vez de Python puro.
        t0 = time.perf_counter()

        valid_mask = (
            df[pickup_col].notna()
            & df[dropoff_col].notna()
            & df[distance_col].notna()
            & df[fare_col].notna()
            & (df[distance_col] >= 0)
            & (df[fare_col] >= 0)
            & (df[dropoff_col] >= df[pickup_col])
        )

        valid_df = df.loc[valid_mask]
        rows_valid += len(valid_df)
        total_rides += len(valid_df)

        if len(valid_df) > 0:
            duration_min = (
                valid_df[dropoff_col] - valid_df[pickup_col]
            ).dt.total_seconds() / 60.0

            sum_distance += float(valid_df[distance_col].sum())
            sum_duration_min += float(duration_min.sum())
            sum_fare += float(valid_df[fare_col].sum())

            hour_counts = valid_df[pickup_col].dt.hour.value_counts()
            for hour, count in hour_counts.items():
                rides_by_hour[int(hour)] += int(count)

            if location_col is not None:
                loc_series = valid_df[location_col].dropna()
                loc_counts = loc_series.astype(np.int64).value_counts()
                for location, count in loc_counts.items():
                    rides_by_location[int(location)] += int(count)

        stage_times["vectorized_filter_and_aggregate"] += time.perf_counter() - t0

    elapsed = time.perf_counter() - overall_start

    if total_rides == 0:
        raise ValueError("Nenhuma linha válida encontrada após a limpeza.")

    results = {
        "total_rides": total_rides,
        "rows_seen": rows_seen,
        "rows_valid": rows_valid,
        "batches_processed": batch_count,
        "avg_distance": sum_distance / total_rides,
        "avg_duration_min": sum_duration_min / total_rides,
        "avg_fare": sum_fare / total_rides,
        "elapsed_seconds": elapsed,
        "stage_times_seconds": dict(stage_times),
        "rides_by_hour": dict(sorted(rides_by_hour.items())),
        "rides_by_location_top20": dict(
            sorted(rides_by_location.items(), key=lambda x: x[1], reverse=True)[:20]
        ),
    }

    if output_json:
        Path(output_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if stage_timing_json:
        Path(stage_timing_json).write_text(
            json.dumps(dict(stage_times), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return results


def save_cprofile_report(
    profile: cProfile.Profile,
    output_text_path: str | Path = OUTPUT_DIR / "cprofile_vectorized.txt",
    output_prof_path: str | Path = OUTPUT_DIR / "cprofile_vectorized.prof",
    sort_by: str = "cumulative",
    top_n: int = 40,
) -> None:
    output_text_path = Path(output_text_path)
    output_prof_path = Path(output_prof_path)

    profile.dump_stats(str(output_prof_path))

    stream = StringIO()
    stats = pstats.Stats(profile, stream=stream).sort_stats(sort_by)
    stats.print_stats(top_n)
    output_text_path.write_text(stream.getvalue(), encoding="utf-8")


def main() -> None:
    ensure_output_dir()

    profiler = cProfile.Profile()
    profiler.enable()

    results = process_parquet_vectorized(
        parquet_path=PARQUET_PATH,
        batch_size=250_000,
        output_json=OUTPUT_DIR / "results_vectorized.json",
        stage_timing_json=OUTPUT_DIR / "stage_timings_vectorized.json",
    )

    profiler.disable()
    save_cprofile_report(profiler)

    print("\n=== RESULTADOS VETORIZADO (pandas, 1 core) ===")
    print(f"Arquivo/base: {PARQUET_PATH}")
    print(f"Batches processados: {results['batches_processed']}")
    print(f"Linhas lidas: {results['rows_seen']}")
    print(f"Linhas válidas: {results['rows_valid']}")
    print(f"Total de corridas: {results['total_rides']}")
    print(f"Distância média: {results['avg_distance']:.3f}")
    print(f"Duração média (min): {results['avg_duration_min']:.3f}")
    print(f"Valor médio: {results['avg_fare']:.3f}")
    print(f"Tempo total: {results['elapsed_seconds']:.2f} s")

    print("\n=== TEMPOS POR ETAPA ===")
    for stage, seconds in sorted(
        results["stage_times_seconds"].items(), key=lambda x: x[1], reverse=True
    ):
        print(f"{stage}: {seconds:.2f} s")

    hour_df = pd.DataFrame(
        list(results["rides_by_hour"].items()),
        columns=["pickup_hour", "rides"],
    ).sort_values("pickup_hour")

    hour_df.to_csv(OUTPUT_DIR / "rides_by_hour_vectorized.csv", index=False)


if __name__ == "__main__":
    main()
