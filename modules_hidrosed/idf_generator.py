from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def generate_idf_power_law(
    a: float,
    b: float,
    c: float,
    d: float,
    durations_min: Iterable[float],
    return_periods: Iterable[float],
) -> pd.DataFrame:
    """IDF genérica: i = a * Tr^b / (t + c)^d, con i en mm/h y t en minutos."""
    rows = []
    for tr in return_periods:
        for t in durations_min:
            t = float(t); tr = float(tr)
            intensity = float(a) * (tr ** float(b)) / ((t + float(c)) ** float(d))
            rows.append({'Tr_anios': tr, 'Duracion_min': t, 'Intensidad_mm_h': intensity, 'Precipitacion_mm': intensity * t / 60.0})
    return pd.DataFrame(rows)


def generate_idf_from_p24(P24_T: dict[float, float], durations_min: Iterable[float]) -> pd.DataFrame:
    """Curvas sintéticas desde P24 por período, usando distribución temporal simple con exponente 0.33.
    Es útil para pre-diseño cuando no existen coeficientes IDF calibrados.
    """
    rows = []
    for tr, p24 in P24_T.items():
        for t in durations_min:
            t = float(t)
            # P_t = P24 * (t/1440)^0.33, i = P_t/(t/60). Acotada para evitar intensidades absurdas.
            p_t = max(0.0, float(p24)) * ((max(t, 1.0) / 1440.0) ** 0.33)
            intensity = p_t * 60.0 / max(t, 1e-9)
            rows.append({'Tr_anios': float(tr), 'Duracion_min': t, 'Intensidad_mm_h': intensity, 'Precipitacion_mm': p_t})
    return pd.DataFrame(rows)


def parse_p24_text(text: str) -> dict[float, float]:
    out = {}
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        line = line.replace('\t', ';').replace(',', '.')
        parts = [p.strip() for p in line.split(';')]
        if len(parts) < 2:
            continue
        try:
            tr = float(parts[0]); p = float(parts[1])
        except Exception:
            continue
        if tr > 0 and p >= 0:
            out[tr] = p
    if not out:
        raise ValueError('No se pudo leer tabla P24. Use formato Tr;P24_mm.')
    return out


def export_idf_excel(idf_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        idf_df.to_excel(writer, index=False, sheet_name='IDF_Formato_Largo')
        if not idf_df.empty:
            piv_i = idf_df.pivot_table(index='Duracion_min', columns='Tr_anios', values='Intensidad_mm_h', aggfunc='mean')
            piv_p = idf_df.pivot_table(index='Duracion_min', columns='Tr_anios', values='Precipitacion_mm', aggfunc='mean')
            piv_i.to_excel(writer, sheet_name='Intensidad_mm_h')
            piv_p.to_excel(writer, sheet_name='Precipitacion_mm')
    return output_path
