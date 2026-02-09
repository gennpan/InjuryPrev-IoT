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


def build_rolling_features(
    df,
    *,
    id_col: str,
    date_col: str,
    window: int,
    min_periods: int,
    stats: tuple[str, ...],
    exclude_cols: set[str],
):
    pd = _require_pandas()

    if window <= 0:
        raise ValueError("window deve essere >= 1")
    if min_periods <= 0:
        raise ValueError("min_periods deve essere >= 1")

    missing = {c for c in [id_col, date_col] if c not in df.columns}
    if missing:
        raise ValueError(f"CSV senza colonne richieste: {sorted(missing)}")

    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df[date_col] = _parse_date_col(df[date_col], dayfirst=False)
    df = df.dropna(subset=[id_col, date_col])

    df = df.sort_values([id_col, date_col], kind="mergesort").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError("Nessuna feature numerica trovata (dopo exclude e coercion).")

    g = df.groupby(id_col, group_keys=False)

    rolled_parts = []
    if "mean" in stats:
        part = (
            g[feature_cols]
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
            .add_prefix(f"roll{window}_mean_")
        )
        rolled_parts.append(part)
    if "max" in stats:
        part = (
            g[feature_cols]
            .rolling(window=window, min_periods=min_periods)
            .max()
            .reset_index(level=0, drop=True)
            .add_prefix(f"roll{window}_max_")
        )
        rolled_parts.append(part)
    if "std" in stats:
        part = (
            g[feature_cols]
            .rolling(window=window, min_periods=min_periods)
            .std(ddof=0)
            .reset_index(level=0, drop=True)
            .add_prefix(f"roll{window}_std_")
        )
        rolled_parts.append(part)

    if not rolled_parts:  # pragma: no cover
        raise ValueError("stats vuoto: niente da calcolare.")

    return pd.concat([df, *rolled_parts], axis=1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calcola rolling features per player_id, ordinando per date. "
            "Per default: tutte le colonne numeriche eccetto player_id/date."
        )
    )
    parser.add_argument(
        "--input",
        default="dataset/subjective/objective.csv",
        type=Path,
        help="CSV input (deve contenere player_id e date).",
    )
    parser.add_argument(
        "--output",
        default="dataset/processed/objective_rolling_7d.csv",
        type=Path,
        help="Dove salvare il CSV con le colonne roll* aggiunte.",
    )
    parser.add_argument("--window", type=int, default=7, help="Lunghezza finestra rolling (numero righe).")
    parser.add_argument("--min-periods", type=int, default=1, help="Minimo numero di osservazioni per calcolare.")
    parser.add_argument("--id-col", type=str, default="player_id", help="Nome colonna ID giocatore.")
    parser.add_argument("--date-col", type=str, default="date", help="Nome colonna data.")
    parser.add_argument(
        "--exclude",
        action="append",
        default=["source_file"],
        help="Colonne da escludere (ripetibile). Default: source_file.",
    )
    parser.add_argument(
        "--stats",
        default="mean,max,std",
        help="Statistiche da calcolare tra: mean,max,std (separate da virgola).",
    )
    return parser


def main(argv: list[str]) -> int:
    pd = _require_pandas()
    args = build_arg_parser().parse_args(argv)

    df = pd.read_csv(args.input)

    stats = tuple(s.strip().lower() for s in str(args.stats).split(",") if s.strip())
    allowed = {"mean", "max", "std"}
    unknown = sorted({s for s in stats if s not in allowed})
    if unknown:
        raise SystemExit(f"--stats contiene valori non supportati: {unknown}. Allowed: {sorted(allowed)}")

    exclude_cols = {args.id_col, args.date_col, *(str(x) for x in (args.exclude or []))}

    out = build_rolling_features(
        df,
        id_col=args.id_col,
        date_col=args.date_col,
        window=int(args.window),
        min_periods=int(args.min_periods),
        stats=stats,
        exclude_cols=exclude_cols,
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out[args.date_col] = pd.to_datetime(out[args.date_col]).dt.strftime("%Y-%m-%d")
    out.to_csv(output_path, index=False)

    print(f"OK: salvato {output_path} ({out.shape[0]} righe, {out.shape[1]} colonne)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
