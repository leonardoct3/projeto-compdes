# dask_nyc_taxi.py
"""
Versão final otimizada: mesma lógica vetorizada de vectorized_nyc_taxi_pandas.py,
mas usando Dask DataFrame para paralelizar o processamento entre os arquivos
Parquet (partições) usando múltiplos cores da CPU local.

Duas otimizações somadas em relação ao baseline sequencial:
  1) vetorização (mesma ideia da versão vectorized_nyc_taxi_pandas.py);
  2) paralelização entre partições/cores via Dask (o que o baseline e a
     versão vetorizada NÃO fazem — ambos rodam em um único core).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import dask
import dask.dataframe as dd
import pandas as pd
from dask.distributed import Client, LocalCluster, performance_report


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


def process_parquet_dask(
    parquet_path: str | Path,
    output_json: str | Path | None = OUTPUT_DIR / "results_dask.json",
    stage_timing_json: str | Path | None = OUTPUT_DIR / "stage_timings_dask.json",
    performance_report_html: str | Path | None = OUTPUT_DIR / "dask_performance_report.html",
    n_workers: int = 1,
    threads_per_worker: int | None = None,
) -> dict:
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Caminho não encontrado: {parquet_path}. "
            "Aponte para a pasta do ano inteiro contendo os arquivos Parquet."
        )

    threads_per_worker = threads_per_worker or (os.cpu_count() or 1)

    stage_times: dict[str, float] = {}
    overall_start = time.perf_counter()

    # processes=False: um único processo com várias threads, compartilhando
    # memória. As operações vetorizadas do pandas/numpy liberam o GIL, então
    # threads paralelizam de verdade aqui, e evitamos duplicar memória entre
    # processos (importante numa máquina com RAM limitada).
    t0 = time.perf_counter()
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=False,
    )
    client = Client(cluster)
    stage_times["cluster_startup"] = time.perf_counter() - t0

    try:
        t0 = time.perf_counter()
        ddf = dd.read_parquet(str(parquet_path), engine="pyarrow")
        schema_cols = list(ddf.columns)

        pickup_col = pick_column(schema_cols, PICKUP_CANDIDATES)
        dropoff_col = pick_column(schema_cols, DROPOFF_CANDIDATES)
        distance_col = pick_column(schema_cols, DISTANCE_CANDIDATES)
        fare_col = pick_column(schema_cols, FARE_CANDIDATES)
        location_col = pick_column(schema_cols, LOCATION_CANDIDATES)

        if pickup_col is None or dropoff_col is None or distance_col is None or fare_col is None:
            raise ValueError(
                "Não encontrei colunas essenciais no parquet. "
                "Esperado algo como pickup/dropoff datetime, trip_distance e fare_amount."
            )

        required_cols = [
            c for c in [pickup_col, dropoff_col, distance_col, fare_col, location_col] if c
        ]
        ddf = ddf[required_cols]
        npartitions = ddf.npartitions
        stage_times["read_parquet_lazy"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if not pd.api.types.is_datetime64_any_dtype(ddf[pickup_col].dtype):
            ddf[pickup_col] = dd.to_datetime(ddf[pickup_col], errors="coerce")
        if not pd.api.types.is_datetime64_any_dtype(ddf[dropoff_col].dtype):
            ddf[dropoff_col] = dd.to_datetime(ddf[dropoff_col], errors="coerce")
        stage_times["datetime_conversion_graph"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        valid_mask = (
            ddf[pickup_col].notna()
            & ddf[dropoff_col].notna()
            & ddf[distance_col].notna()
            & ddf[fare_col].notna()
            & (ddf[distance_col] >= 0)
            & (ddf[fare_col] >= 0)
            & (ddf[dropoff_col] >= ddf[pickup_col])
        )
        valid = ddf[valid_mask]
        valid = valid.assign(
            duration_min=(valid[dropoff_col] - valid[pickup_col]).dt.total_seconds() / 60.0,
            pickup_hour=valid[pickup_col].dt.hour,
        )

        rows_seen_lazy = ddf.shape[0]
        total_rides_lazy = valid.shape[0]
        sum_distance_lazy = valid[distance_col].sum()
        sum_duration_lazy = valid["duration_min"].sum()
        sum_fare_lazy = valid[fare_col].sum()
        rides_by_hour_lazy = valid.groupby("pickup_hour").size()
        if location_col is not None:
            rides_by_location_lazy = valid.groupby(location_col).size()
        else:
            rides_by_location_lazy = None
        stage_times["build_task_graph"] = time.perf_counter() - t0

        # dask.compute(...) executa todos os lazy objects numa única passada,
        # compartilhando entre eles a leitura do parquet e o filtro (o grafo
        # de tarefas é deduplicado), distribuído entre as threads/partições.
        t0 = time.perf_counter()
        report_ctx = (
            performance_report(filename=str(performance_report_html))
            if performance_report_html
            else None
        )
        if report_ctx:
            report_ctx.__enter__()
        try:
            computed = dask.compute(
                rows_seen_lazy,
                total_rides_lazy,
                sum_distance_lazy,
                sum_duration_lazy,
                sum_fare_lazy,
                rides_by_hour_lazy,
                rides_by_location_lazy,
            )
        finally:
            if report_ctx:
                report_ctx.__exit__(None, None, None)
        stage_times["compute_distributed"] = time.perf_counter() - t0

        (
            rows_seen,
            total_rides,
            sum_distance,
            sum_duration_min,
            sum_fare,
            rides_by_hour_series,
            rides_by_location_series,
        ) = computed
    finally:
        client.close()
        cluster.close()

    elapsed = time.perf_counter() - overall_start

    if total_rides == 0:
        raise ValueError("Nenhuma linha válida encontrada após a limpeza.")

    rides_by_hour = {int(h): int(c) for h, c in rides_by_hour_series.items()}

    rides_by_location_top20 = {}
    if rides_by_location_series is not None:
        top20 = rides_by_location_series.sort_values(ascending=False).head(20)
        rides_by_location_top20 = {int(loc): int(c) for loc, c in top20.items()}

    results = {
        "total_rides": int(total_rides),
        "rows_seen": int(rows_seen),
        "rows_valid": int(total_rides),
        "batches_processed": int(npartitions),
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "avg_distance": sum_distance / total_rides,
        "avg_duration_min": sum_duration_min / total_rides,
        "avg_fare": sum_fare / total_rides,
        "elapsed_seconds": elapsed,
        "stage_times_seconds": dict(stage_times),
        "rides_by_hour": dict(sorted(rides_by_hour.items())),
        "rides_by_location_top20": rides_by_location_top20,
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


def main() -> None:
    ensure_output_dir()

    results = process_parquet_dask(
        parquet_path=PARQUET_PATH,
        output_json=OUTPUT_DIR / "results_dask.json",
        stage_timing_json=OUTPUT_DIR / "stage_timings_dask.json",
        performance_report_html=OUTPUT_DIR / "dask_performance_report.html",
    )

    print("\n=== RESULTADOS DASK (multi-core) ===")
    print(f"Arquivo/base: {PARQUET_PATH}")
    print(f"Partições processadas: {results['batches_processed']}")
    print(f"Threads: {results['threads_per_worker']}")
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

    hour_df.to_csv(OUTPUT_DIR / "rides_by_hour_dask.csv", index=False)


if __name__ == "__main__":
    main()
