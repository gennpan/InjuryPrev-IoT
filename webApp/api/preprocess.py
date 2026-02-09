from __future__ import annotations
from typing import Dict, List
import math

BASE_FEATURES = [
    "speed_mean", "speed_max", "speed_std",
    "acc_norm_mean", "acc_norm_max", "acc_norm_std",
    "gyro_norm_mean", "gyro_norm_max",
] # feature che manda l'utente in input (manda 7 giorni o sessioni)

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)

def _std(xs: List[float]) -> float:
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def compute_input_vector(last_7_days: List[Dict[str, float]]) -> Dict[str, float]:
    """
    # prende in ingresso una lista di 7 giorni del tipo {speedmean : 0.5,speedstd 0.7},{speedmean : 0.5,...} ... x altre 5 volte
    # quello che caccia è un singolo dizionario di 32 elementi (input della rete)
    """
    if len(last_7_days) != 7:
        raise ValueError("Servono esattamente 7 giorni di sessioni.")

    # controlli: ogni giorno deve avere tutte le base features
    for i, day in enumerate(last_7_days):
        for f in BASE_FEATURES:
            if f not in day:
                raise ValueError(f"Giorno {i}: manca la feature '{f}'.")

    out: Dict[str, float] = {}

    current = last_7_days[-1]
    for f in BASE_FEATURES:
        out[f] = float(current[f])

    #rolling su 7 giorni per ciascuna base feature
    for f in BASE_FEATURES:
        series = [float(day[f]) for day in last_7_days]
        out[f"roll7_mean_{f}"] = float(_mean(series))
        out[f"roll7_max_{f}"] = float(max(series))
        out[f"roll7_std_{f}"] = float(_std(series))

    return out

def feature_order() -> List[str]:
    order: List[str] = []
    order += BASE_FEATURES
    order += [f"roll7_mean_{f}" for f in BASE_FEATURES]
    order += [f"roll7_max_{f}" for f in BASE_FEATURES]
    order += [f"roll7_std_{f}" for f in BASE_FEATURES]
    return order
