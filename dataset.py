from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


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


def _require_torch():
    try:
        import torch  # type: ignore
        from torch.utils.data import Dataset  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Dipendenza mancante: torch (PyTorch).\n"
            "Installa con: pip install torch\n"
            f"Dettagli: {exc}"
        ) from exc
    return torch, Dataset


@dataclass(frozen=True)
class InjuryDatasetSpec:
    csv_path: Path
    horizon_days: int = 14
    id_col: str = "player_id"
    date_col: str = "date"
    label_col: str | None = None
    drop_cols: tuple[str, ...] = ()


def infer_label_col(columns: Iterable[str], *, horizon_days: int) -> str:
    columns = list(columns)
    expected = f"injury_next_{int(horizon_days)}_days"
    if expected in columns:
        return expected

    injury_cols = [c for c in columns if str(c).startswith("injury_next_") and str(c).endswith("_days")]
    if len(injury_cols) == 1:
        return str(injury_cols[0])

    if injury_cols:
        raise ValueError(
            f"label_col non trovato: atteso {expected!r}. "
            f"Colonne candidate: {sorted(map(str, injury_cols))}. "
            "Passa esplicitamente label_col."
        )
    raise ValueError(
        f"label_col non trovato: atteso {expected!r} e nessuna colonna injury_next_*_days presente. "
        "Passa esplicitamente label_col."
    )


class InjuryWithinHDaysDataset:  # torch.utils.data.Dataset (definito runtime)
    """
    Dataset per classificazione binaria: infortunio entro H giorni (0/1).

    CSV atteso (minimo):
    - player_id
    - date
    - colonne feature numeriche
    - label: injury_next_{H}_days (o label_col esplicito)
    """

    def __init__(self, spec: InjuryDatasetSpec):
        pd = _require_pandas()
        torch, Dataset = _require_torch()

        if int(spec.horizon_days) <= 0:
            raise ValueError("horizon_days deve essere > 0")

        csv_path = Path(spec.csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV non trovato: {csv_path}")

        df = pd.read_csv(csv_path)

        missing = {c for c in [spec.id_col, spec.date_col] if c not in df.columns}
        if missing:
            raise ValueError(f"CSV senza colonne richieste: {sorted(missing)} ({csv_path})")

        label_col = spec.label_col or infer_label_col(df.columns, horizon_days=int(spec.horizon_days))
        if label_col not in df.columns:
            raise ValueError(f"CSV senza label_col={label_col!r} ({csv_path})")

        drop_cols = {spec.id_col, spec.date_col, label_col, *(str(c) for c in spec.drop_cols)}
        feature_cols = [c for c in df.columns if c not in drop_cols]
        if not feature_cols:
            raise ValueError("Nessuna feature trovata dopo drop_cols.")

        for c in feature_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        if df[feature_cols].isna().any().any():
            bad = sorted([c for c in feature_cols if df[c].isna().any()])
            raise ValueError(f"Feature con NaN dopo conversione numerica: {bad[:10]}{'...' if len(bad) > 10 else ''}")

        y = pd.to_numeric(df[label_col], errors="coerce")
        if y.isna().any():
            raise ValueError(f"Label contiene valori non numerici/NaN in {label_col!r}.")

        self.spec = spec
        self.csv_path = csv_path
        self.label_col = label_col
        self.feature_names = [str(c) for c in feature_cols]

        self.player_ids = df[spec.id_col].astype(str).to_numpy()
        self.dates = pd.to_datetime(df[spec.date_col], errors="coerce").dt.normalize().to_numpy()
        if pd.isna(self.dates).any():
            raise ValueError(f"Date non parsabili nella colonna {spec.date_col!r}.")

        self.X = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)

        class _TorchDataset(Dataset):
            def __len__(inner_self):
                return int(self.X.shape[0])

            def __getitem__(inner_self, idx: int):
                return self.X[idx], self.y[idx]

        self._dataset = _TorchDataset()

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int):
        return self._dataset[idx]


def make_dataset(
    csv_path: str | Path = "dataset/processed/objective_rolling_7d_with_injury_next_14_days.csv",
    *,
    horizon_days: int = 14,
    label_col: str | None = None,
    id_col: str = "player_id",
    date_col: str = "date",
    drop_cols: Sequence[str] = (),
) -> InjuryWithinHDaysDataset:
    spec = InjuryDatasetSpec(
        csv_path=Path(csv_path),
        horizon_days=int(horizon_days),
        label_col=label_col,
        id_col=id_col,
        date_col=date_col,
        drop_cols=tuple(drop_cols),
    )
    return InjuryWithinHDaysDataset(spec)
