from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


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


def _episode_starts(dates: np.ndarray, *, gap_days: int = 1) -> np.ndarray:
    """
    dates: array datetime64[D] già ordinato e unico per giorno.
    Ritorna solo gli start di episodio: primo giorno e ogni volta che il gap > gap_days.
    """
    if dates.size == 0:
        return dates
    if gap_days < 0:
        raise ValueError("gap_days deve essere >= 0")
    if gap_days == 0:
        return dates

    diffs = (dates[1:] - dates[:-1]).astype("timedelta64[D]").astype(int)
    mask = np.r_[True, diffs > gap_days]
    return dates[mask]


def load_event_dates(
    *,
    injury_path: Path | None,
    illness_path: Path | None,
    event_source: str,
    id_col_events: str,
    date_col_events: str,
    dayfirst: bool,
):
    pd = _require_pandas()

    frames = []
    if event_source in {"injury", "both"}:
        if injury_path is None:
            raise ValueError("injury_path è richiesto con event_source=injury/both")
        inj = pd.read_csv(injury_path)
        if id_col_events not in inj.columns or date_col_events not in inj.columns:
            raise ValueError(
                f"Injury CSV deve contenere {id_col_events!r} e {date_col_events!r}: {injury_path}"
            )
        inj = inj.rename(columns={id_col_events: "player_id", date_col_events: "date"})
        inj["date"] = _parse_date_col(inj["date"], dayfirst=dayfirst)
        inj = inj.dropna(subset=["player_id", "date"])
        frames.append(inj[["player_id", "date"]])

    if event_source in {"illness", "both"}:
        if illness_path is None:
            raise ValueError("illness_path è richiesto con event_source=illness/both")
        ill = pd.read_csv(illness_path)
        if id_col_events not in ill.columns or date_col_events not in ill.columns:
            raise ValueError(
                f"Illness CSV deve contenere {id_col_events!r} e {date_col_events!r}: {illness_path}"
            )
        ill = ill.rename(columns={id_col_events: "player_id", date_col_events: "date"})
        ill["date"] = _parse_date_col(ill["date"], dayfirst=dayfirst)
        ill = ill.dropna(subset=["player_id", "date"])
        frames.append(ill[["player_id", "date"]])

    if not frames:
        raise ValueError("event_source non produce nessun evento (frames vuoto).")

    events = pd.concat(frames, axis=0, ignore_index=True)
    events["player_id"] = events["player_id"].astype(str).str.strip()
    events = events.drop_duplicates(["player_id", "date"]).sort_values(["player_id", "date"], kind="mergesort")
    return events


def build_survival_labels(
    df,
    *,
    events_df,
    id_col: str,
    date_col: str,
    gap_days: int = 0,
    include_same_day: bool = False,
):
    pd = _require_pandas()

    missing = {c for c in [id_col, date_col] if c not in df.columns}
    if missing:
        raise ValueError(f"Daily CSV senza colonne richieste: {sorted(missing)}")

    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df[date_col] = _parse_date_col(df[date_col], dayfirst=False)
    if df[date_col].isna().any():
        n_bad = int(df[date_col].isna().sum())
        raise ValueError(f"Daily CSV: {n_bad} righe con date non parsabili.")

    dup = df.duplicated([id_col, date_col]).sum()
    if int(dup) != 0:
        raise ValueError(
            f"Daily CSV: trovate {int(dup)} chiavi duplicate (player_id,date). "
            "Serve aggregazione prima di costruire T/E."
        )

    df = df.sort_values([id_col, date_col], kind="mergesort").reset_index(drop=True)

    events = events_df.copy()
    if not {"player_id", "date"}.issubset(events.columns):
        raise ValueError("events_df deve contenere colonne 'player_id' e 'date'.")
    events["player_id"] = events["player_id"].astype(str).str.strip()
    events["date"] = pd.to_datetime(events["date"], errors="coerce").dt.normalize()
    events = events.dropna(subset=["player_id", "date"]).drop_duplicates(["player_id", "date"])

    # Censor per player: ultima data disponibile nel daily dataframe
    censor = df.groupby(id_col, sort=False)[date_col].max()

    # Ignora eventi oltre il censor (non osservabili nei dati feature)
    events["censor"] = events["player_id"].map(censor)
    events = events.dropna(subset=["censor"])
    events = events.loc[events["date"] <= events["censor"], ["player_id", "date"]]

    # player_id -> start episodio
    event_map: dict[str, np.ndarray] = {}
    for pid, g in events.groupby("player_id", sort=False):
        d = np.asarray(g["date"].sort_values().unique(), dtype="datetime64[D]")
        event_map[str(pid)] = _episode_starts(d, gap_days=gap_days)

    T = np.empty((len(df),), dtype=np.int64)
    E = np.empty((len(df),), dtype=np.int64)

    dates_all = df[date_col].to_numpy(dtype="datetime64[D]")
    players_all = df[id_col].to_numpy(dtype=str)

    change = np.flatnonzero(players_all[1:] != players_all[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(df)]

    side = "left" if include_same_day else "right"

    for s, e in zip(starts, ends, strict=True):
        pid = str(players_all[int(s)])
        d = dates_all[int(s) : int(e)]
        censor_date = d.max()

        ev = event_map.get(pid)
        if ev is None or ev.size == 0:
            E[int(s) : int(e)] = 0
            T[int(s) : int(e)] = (censor_date - d).astype("timedelta64[D]").astype(np.int64)
            continue

        idx = np.searchsorted(ev, d, side=side)
        has = idx < ev.size

        next_dates = np.empty_like(d)
        next_dates[has] = ev[idx[has]]
        next_dates[~has] = censor_date

        t_event = (next_dates - d).astype("timedelta64[D]").astype(np.int64)
        t_cens = (censor_date - d).astype("timedelta64[D]").astype(np.int64)

        E[int(s) : int(e)] = has.astype(np.int64)
        T[int(s) : int(e)] = np.where(has, t_event, t_cens).astype(np.int64, copy=False)

    df["T"] = T
    df["E"] = E
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Costruisce label survival (T,E) per ogni riga del daily rolling.\n"
            "Per ogni (player_id,date):\n"
            "- E=1 e T=giorni al prossimo evento (injury/illness) dopo la data\n"
            "- E=0 e T=giorni al censor (ultima data osservata per quel player)\n"
            "NB: per default gli eventi oltre il censor vengono ignorati."
        )
    )
    parser.add_argument(
        "--input",
        default="dataset/processed/objective_rolling_7d.csv",
        type=Path,
        help="CSV input con feature (almeno player_id,date).",
    )
    parser.add_argument(
        "--output",
        default="dataset/processed/daily_rolling_7d_labeled.csv",
        type=Path,
        help="Dove salvare il CSV con colonne T,E aggiunte.",
    )
    parser.add_argument(
        "--event-source",
        default="injury",
        choices=["injury", "illness", "both"],
        help="Quali eventi usare per definire l'endpoint (default: injury).",
    )
    parser.add_argument(
        "--injury-csv",
        default="dataset/subjective/injury/injury.csv",
        type=Path,
        help="CSV injury.",
    )
    parser.add_argument(
        "--illness-csv",
        default="dataset/subjective/illness/illness.csv",
        type=Path,
        help="CSV illness.",
    )
    parser.add_argument("--id-col", default="player_id", help="Colonna player id nel daily CSV.")
    parser.add_argument("--date-col", default="date", help="Colonna date nel daily CSV.")

    parser.add_argument(
        "--events-id-col",
        default="player_name",
        help="Colonna player id nei CSV eventi (injury/illness).",
    )
    parser.add_argument(
        "--events-date-col",
        default="timestamp",
        help="Colonna data nei CSV eventi (injury/illness).",
    )
    parser.add_argument(
        "--events-dayfirst",
        action="store_true",
        help="Parse date eventi come formato day-first (es. 20.03.2020).",
    )
    parser.add_argument(
        "--gap-days",
        type=int,
        default=0,
        help=(
            "Se >0, comprime eventi su episodi: giorni con gap <= gap-days sono lo stesso episodio "
            "(si usa solo il primo giorno). Default: 0 (nessuna compressione, ogni giorno evento conta)."
        ),
    )
    parser.add_argument(
        "--include-same-day",
        action="store_true",
        help="Se attivo, considera evento anche se avviene lo stesso giorno della riga (T può essere 0).",
    )
    return parser


def main(argv: list[str]) -> int:
    pd = _require_pandas()
    args = build_arg_parser().parse_args(argv)

    df = pd.read_csv(args.input)

    injury_path = args.injury_csv if args.event_source in {"injury", "both"} else None
    illness_path = args.illness_csv if args.event_source in {"illness", "both"} else None

    events = load_event_dates(
        injury_path=injury_path,
        illness_path=illness_path,
        event_source=args.event_source,
        id_col_events=args.events_id_col,
        date_col_events=args.events_date_col,
        dayfirst=bool(args.events_dayfirst),
    )

    labeled = build_survival_labels(
        df,
        events_df=events,
        id_col=args.id_col,
        date_col=args.date_col,
        gap_days=int(args.gap_days),
        include_same_day=bool(args.include_same_day),
    )

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled[args.date_col] = pd.to_datetime(labeled[args.date_col]).dt.strftime("%Y-%m-%d")
    labeled.to_csv(output_path, index=False)

    n_events = int(labeled["E"].sum())
    print(f"OK: salvato {output_path} ({labeled.shape[0]} righe, {labeled.shape[1]} colonne), E=1: {n_events}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
