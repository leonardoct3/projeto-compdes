# baseline_nyc_taxi_sequencial.py
from __future__ import annotations

import cProfile
import json
import pstats
import time
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as ds


PICKUP_CANDIDATES = ["tpep_pickup_datetime", "pickup_datetime", "pickup_at"]
DROPOFF_CANDIDATES = ["tpep_dropoff_datetime", "dropoff_datetime", "dropoff_at"]
DISTANCE_CANDIDATES = ["trip_distance", "distance"]
FARE_CANDIDATES = ["fare_amount", "total_amount", "fare"]
LOCATION_CANDIDATES = ["PULocationID", "pickup_location_id", "location_id"]

INPUT_DIR = Path("data")
OUTPUT_DIR = Path("output")
PARQUET_PATH = INPUT_DIR / "yellow_tripdata_2025"


@dataclass
class MetricsState:
    total_rides: int = 0
    rows_seen: int = 0
    rows_valid: int = 0
    batch_count: int = 0

    sum_distance: float = 0.0
    sum_duration_min: float = 0.0
    sum_fare: float = 0.0

    rides_by_hour: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    rides_by_location: dict[int, int] = field(default_factory=lambda: defaultdict(int))

    stage_times: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_column(existing_columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in existing_columns:
            return col
    return None


def resolve_dataset_and_columns(
    parquet_path: str | Path,
) -> tuple[ds.Dataset, dict[str, str | None], list[str]]:
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Caminho não encontrado: {parquet_path}. "
            "Aponte para a pasta do ano inteiro contendo os arquivos Parquet."
        )

    dataset = ds.dataset(str(parquet_path), format="parquet")
    schema_cols = list(dataset.schema.names)

    columns = {
        "pickup": pick_column(schema_cols, PICKUP_CANDIDATES),
        "dropoff": pick_column(schema_cols, DROPOFF_CANDIDATES),
        "distance": pick_column(schema_cols, DISTANCE_CANDIDATES),
        "fare": pick_column(schema_cols, FARE_CANDIDATES),
        "location": pick_column(schema_cols, LOCATION_CANDIDATES),
    }

    if (
        columns["pickup"] is None
        or columns["dropoff"] is None
        or columns["distance"] is None
        or columns["fare"] is None
    ):
        raise ValueError(
            "Não encontrei colunas essenciais no parquet. "
            "Esperado algo como pickup/dropoff datetime, trip_distance e fare_amount."
        )

    required_cols = [c for c in columns.values() if c is not None]
    return dataset, columns, required_cols


def convert_datetime_columns(
    df: pd.DataFrame, pickup_col: str, dropoff_col: str
) -> None:
    df[pickup_col] = pd.to_datetime(df[pickup_col], errors="coerce")
    df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors="coerce")


def is_valid_row(
    pickup: Any,
    dropoff: Any,
    distance: Any,
    fare: Any,
) -> bool:
    if pd.isna(pickup) or pd.isna(dropoff) or pd.isna(distance) or pd.isna(fare):
        return False
    if distance < 0 or fare < 0:
        return False
    if dropoff < pickup:
        return False
    return True


def process_valid_row(
    state: MetricsState,
    pickup: pd.Timestamp,
    dropoff: pd.Timestamp,
    distance: Any,
    fare: Any,
    location: Any | None,
) -> None:
    state.rows_valid += 1
    state.total_rides += 1

    duration_min = (dropoff - pickup).total_seconds() / 60.0

    state.sum_distance += float(distance)
    state.sum_duration_min += float(duration_min)
    state.sum_fare += float(fare)

    state.rides_by_hour[int(pickup.hour)] += 1

    if location is not None and not pd.isna(location):
        state.rides_by_location[int(location)] += 1


def process_batch(
    df: pd.DataFrame,
    columns: dict[str, str | None],
    state: MetricsState,
) -> None:
    state.rows_seen += len(df)

    pickup_col = columns["pickup"]
    dropoff_col = columns["dropoff"]
    distance_col = columns["distance"]
    fare_col = columns["fare"]
    location_col = columns["location"]

    assert pickup_col is not None
    assert dropoff_col is not None
    assert distance_col is not None
    assert fare_col is not None

    t0 = time.perf_counter()
    convert_datetime_columns(df, pickup_col, dropoff_col)
    state.stage_times["datetime_conversion"] += time.perf_counter() - t0

    col_pos = {name: idx for idx, name in enumerate(df.columns)}
    pickup_idx = col_pos[pickup_col]
    dropoff_idx = col_pos[dropoff_col]
    distance_idx = col_pos[distance_col]
    fare_idx = col_pos[fare_col]
    location_idx = col_pos[location_col] if location_col is not None else None

    t0 = time.perf_counter()
    for row in df.itertuples(index=False, name=None):
        pickup = row[pickup_idx]
        dropoff = row[dropoff_idx]
        distance = row[distance_idx]
        fare = row[fare_idx]
        location = row[location_idx] if location_idx is not None else None

        if is_valid_row(pickup, dropoff, distance, fare):
            process_valid_row(
                state=state,
                pickup=pickup,
                dropoff=dropoff,
                distance=distance,
                fare=fare,
                location=location,
            )

    state.stage_times["row_processing"] += time.perf_counter() - t0


def process_parquet_sequential(
    parquet_path: str | Path,
    batch_size: int = 250_000,
    output_json: str | Path | None = OUTPUT_DIR / "results_baseline.json",
    stage_timing_json: str | Path | None = OUTPUT_DIR / "stage_timings_baseline.json",
) -> dict:
    dataset, columns, required_cols = resolve_dataset_and_columns(parquet_path)
    state = MetricsState()

    overall_start = time.perf_counter()
    scanner = dataset.scanner(columns=required_cols, batch_size=batch_size)

    for batch in scanner.to_batches():
        state.batch_count += 1

        t0 = time.perf_counter()
        df = batch.to_pandas()
        state.stage_times["to_pandas"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        process_batch(df, columns, state)
        state.stage_times["batch_processing"] += time.perf_counter() - t0

    elapsed = time.perf_counter() - overall_start

    if state.total_rides == 0:
        raise ValueError("Nenhuma linha válida encontrada após a limpeza.")

    results = {
        "total_rides": state.total_rides,
        "rows_seen": state.rows_seen,
        "rows_valid": state.rows_valid,
        "batches_processed": state.batch_count,
        "avg_distance": state.sum_distance / state.total_rides,
        "avg_duration_min": state.sum_duration_min / state.total_rides,
        "avg_fare": state.sum_fare / state.total_rides,
        "elapsed_seconds": elapsed,
        "stage_times_seconds": dict(state.stage_times),
        "rides_by_hour": dict(sorted(state.rides_by_hour.items())),
        "rides_by_location_top20": dict(
            sorted(state.rides_by_location.items(), key=lambda x: x[1], reverse=True)[
                :20
            ]
        ),
    }

    if output_json:
        Path(output_json).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if stage_timing_json:
        Path(stage_timing_json).write_text(
            json.dumps(dict(state.stage_times), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return results


def save_cprofile_report(
    profile: cProfile.Profile,
    output_text_path: str | Path = OUTPUT_DIR / "cprofile_baseline.txt",
    output_prof_path: str | Path = OUTPUT_DIR / "cprofile_baseline.prof",
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

    results = process_parquet_sequential(
        parquet_path=PARQUET_PATH,
        batch_size=250_000,
        output_json=OUTPUT_DIR / "results_baseline.json",
        stage_timing_json=OUTPUT_DIR / "stage_timings_baseline.json",
    )

    profiler.disable()
    save_cprofile_report(profiler)

    print("\n=== RESULTADOS BASELINE SEQUENCIAL ===")
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

    hour_df.to_csv(OUTPUT_DIR / "rides_by_hour.csv", index=False)


if __name__ == "__main__":
    main()
