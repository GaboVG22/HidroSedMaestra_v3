"""Cálculo hidrológico preliminar de crecidas.

Este módulo entrega un cálculo base tipo racional y una clasificación técnica.
Debe complementarse con metodología DGA/Manual de Carreteras según información
pluviométrica disponible y escala de cuenca.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


RETURN_PERIODS = [2, 5, 10, 25, 50, 100, 200]


@dataclass
class WatershedParams:
    area_km2: float
    main_channel_length_km: float
    mean_slope_percent: float
    runoff_coefficient: float


def kirpich_tc_minutes(length_km: float, slope_percent: float) -> float:
    """Tiempo de concentración por Kirpich aproximado.

    Tc[min] = 0.01947 * L[m]^0.77 * S[m/m]^-0.385
    """
    if length_km <= 0:
        raise ValueError("La longitud de cauce debe ser mayor que cero.")
    slope = slope_percent / 100.0
    if slope <= 0:
        raise ValueError("La pendiente debe ser mayor que cero.")
    length_m = length_km * 1000.0
    return 0.01947 * (length_m ** 0.77) * (slope ** -0.385)


def recommend_method(area_km2: float) -> str:
    if area_km2 <= 25:
        return "Cuenca pequeña: método racional como primera aproximación; contrastar con curvas IDF y criterio DGA/Manual de Carreteras."
    if area_km2 <= 200:
        return "Cuenca mediana: usar método racional solo como control; preferir hidrograma unitario, regionalización DGA o HEC-HMS."
    return "Cuenca grande: no usar racional como método principal; usar modelación hidrológica distribuida/semi-distribuida, regionalización y calibración fluviométrica si existe."


def rational_peak_flow(area_km2: float, runoff_coefficient: float, intensity_mm_h: float) -> float:
    """Q[m3/s] = 0.278 * C * I[mm/h] * A[km2]."""
    return 0.278 * runoff_coefficient * intensity_mm_h * area_km2


def compute_rational_flows(
    params: WatershedParams,
    intensities_by_return_period: Mapping[int, float],
) -> pd.DataFrame:
    rows = []
    tc_min = kirpich_tc_minutes(params.main_channel_length_km, params.mean_slope_percent)
    for tr in RETURN_PERIODS:
        intensity = float(intensities_by_return_period.get(tr, 0.0))
        q = rational_peak_flow(params.area_km2, params.runoff_coefficient, intensity)
        rows.append(
            {
                "T_ret_años": tr,
                "Intensidad_mm_h": intensity,
                "Coef_escorrentia_C": params.runoff_coefficient,
                "Area_km2": params.area_km2,
                "Tc_min_Kirpich": tc_min,
                "Q_m3_s_racional": q,
            }
        )
    return pd.DataFrame(rows)
