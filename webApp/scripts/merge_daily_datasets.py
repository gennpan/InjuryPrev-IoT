from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _require_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Dipendenza mancante: pandas.\n"
            "Installa con: pip install pandas\n"
            f"Dettagli: {exc}"
        ) from exc
    return pd


def _parse_date_col(series, *, dayfirst: bool):
    pd = _require_pandas()
    dt = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
    return dt.dt.normalize()


def load_objective_csv(path: Path):
    pd = _require_pandas()
    df = pd.read_csv(path)

    missing = {c for c in ["player_id", "date"] if c not in df.columns}
    if missing:
        raise SystemExit(f"Objective CSV senza colonne richieste: {sorted(missing)} ({path})")

    df["date"] = _parse_date_col(df["date"], dayfirst=False)
    if df["date"].isna().any():
        n_bad = int(df["date"].isna().sum())
        raise SystemExit(f"Objective CSV: {n_bad} righe con date non parsabili ({path})")

    dup = df.duplicated(["player_id", "date"]).sum()
    if int(dup) != 0:
        raise SystemExit(
            f"Objective CSV: trovate {int(dup)} chiavi duplicate (player_id,date). "
            "Serve aggregazione prima del merge."
        )

    return df


def load_wellness_wide_csv(path: Path, *, value_name: str):
    pd = _require_pandas()
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise SystemExit(f"Wellness CSV inatteso (meno di 2 colonne): {path}")

    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df.columns = [str(c).strip() for c in df.columns]

    df["date"] = _parse_date_col(df["date"], dayfirst=True)
    df = df.melt(id_vars=["date"], var_name="player_id", value_name=value_name)

    df[value_name] = pd.to_numeric(df[value_name], errors="coerce")
    df = df.dropna(subset=["date", "player_id", value_name])

    dup = df.duplicated(["player_id", "date"]).sum()
    if int(dup) != 0:
        raise SystemExit(
            f"Wellness CSV {path}: trovate {int(dup)} chiavi duplicate (player_id,date) per {value_name}."
        )

    return df


def merge_daily(
    *,
    objective_path: Path,
    fatigue_path: Path,
    soreness_path: Path,
    sleep_quality_path: Path,
    stress_path: Path,
    how: str,
):
    objective = load_objective_csv(objective_path)

    fatigue = load_wellness_wide_csv(fatigue_path, value_name="fatigue")
    soreness = load_wellness_wide_csv(soreness_path, value_name="soreness")
    sleep_quality = load_wellness_wide_csv(sleep_quality_path, value_name="sleep_quality")
    stress = load_wellness_wide_csv(stress_path, value_name="stress")

    merged = objective.merge(fatigue, on=["player_id", "date"], how=how)
    merged = merged.merge(soreness, on=["player_id", "date"], how=how)
    merged = merged.merge(sleep_quality, on=["player_id", "date"], how=how)
    merged = merged.merge(stress, on=["player_id", "date"], how=how)

    key_cols = ["player_id", "date"]
    objective_cols = [c for c in objective.columns if c not in key_cols]
    wellness_cols = ["fatigue", "soreness", "sleep_quality", "stress"]
    merged = merged[key_cols + objective_cols + wellness_cols]

    merged = merged.sort_values(["player_id", "date"], kind="mergesort").reset_index(drop=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge giornaliero (player_id,date) tra objective.csv e wellness (fatigue/soreness/"
            "sleep_quality/stress) in formato long."
        )
    )

    parser.add_argument(
        "--objective",
        default="dataset/subjective/objective.csv",
        type=Path,
        help="Path a objective.csv (long con player_id,date,...)",
    )
    parser.add_argument(
        "--fatigue",
        default="dataset/subjective/wellness/fatigue.csv",
        type=Path,
        help="Path a fatigue.csv (wide: 1 col data + colonne per player_id)",
    )
    parser.add_argument(
        "--soreness",
        default="dataset/subjective/wellness/soreness.csv",
        type=Path,
        help="Path a soreness.csv (wide)",
    )
    parser.add_argument(
        "--sleep-quality",
        dest="sleep_quality",
        default="dataset/subjective/wellness/sleep_quality.csv",
        type=Path,
        help="Path a sleep_quality.csv (wide)",
    )
    parser.add_argument(
        "--stress",
        default="dataset/subjective/wellness/stress.csv",
        type=Path,
        help="Path a stress.csv (wide)",
    )
    parser.add_argument(
        "--how",
        default="left",
        choices=["left", "right", "inner", "outer"],
        help=(
            "Tipo di join sui key (player_id,date). "
            "Default: left (tieni solo giorni con sessione oggettiva)."
        ),
    )
    parser.add_argument(
        "--output",
        default="dataset/processed/daily_merged.csv",
        type=Path,
        help="Dove salvare il CSV merged.",
    )
    return parser


def main(argv: list[str]) -> int:
    args = build_arg_parser().parse_args(argv)

    df = merge_daily(
        objective_path=args.objective,
        fatigue_path=args.fatigue,
        soreness_path=args.soreness,
        sleep_quality_path=args.sleep_quality,
        stress_path=args.stress,
        how=args.how,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"OK: salvato {output_path} ({df.shape[0]} righe, {df.shape[1]} colonne)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
