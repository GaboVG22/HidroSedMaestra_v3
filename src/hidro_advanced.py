"""Módulos avanzados para reforzar la memoria hidrológica.

Incluye criterios derivados de memorias hidrológicas revisadas: pendiente de
Mociornita, múltiples tiempos de concentración, recomendación de caudal adoptado
y matrices de consistencia para lluvia/estaciones.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from shapely.geometry.base import BaseGeometry

from .hidro_kmz_core import project_geom, contour_elevation, rational_q, scs_effective_precip


def _finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def compute_mociornita_slope_pct(basin_geom: BaseGeometry, contours: list, epsg: int) -> Dict[str, Any]:
    """Pendiente media de cuenca según fórmula de Mociornita.

    S = Δh/A · (l0/2 + Σ li + ln/2)

    Δh en m, A en m² y longitudes de curvas en m. Devuelve pendiente en %.
    Es una aproximación cartográfica porque depende de la calidad de las curvas
    de nivel y de la correcta intersección con el polígono de cuenca.
    """
    out = {"S_mociornita_pct": np.nan, "delta_h_m": np.nan, "area_m2": np.nan, "longitud_total_curvas_m": np.nan, "n_niveles": 0, "estado": "No calculado"}
    if basin_geom is None or not contours:
        out["estado"] = "Faltan polígono o curvas de nivel"
        return out
    try:
        basin_m = project_geom(basin_geom, epsg)
        area_m2 = float(basin_m.area)
        by_level: Dict[float, float] = {}
        for c in contours:
            elev = contour_elevation(c)
            if elev is None or not _finite(elev):
                continue
            try:
                g_m = project_geom(c.geometry, epsg)
                inter = g_m.intersection(basin_m)
                if inter.is_empty:
                    continue
                length = float(inter.length)
                if length <= 0:
                    continue
                key = round(float(elev), 3)
                by_level[key] = by_level.get(key, 0.0) + length
            except Exception:
                continue
        levels = sorted(by_level.keys())
        if area_m2 <= 0 or len(levels) < 2:
            out["estado"] = "Curvas insuficientes dentro de la cuenca"
            return out
        diffs = np.diff(levels)
        dh = float(np.nanmedian(diffs)) if len(diffs) else np.nan
        lengths = [by_level[z] for z in levels]
        weighted = 0.5 * lengths[0] + sum(lengths[1:-1]) + 0.5 * lengths[-1]
        S = dh * weighted / area_m2
        out.update({
            "S_mociornita_pct": float(S * 100.0),
            "delta_h_m": dh,
            "area_m2": area_m2,
            "longitud_total_curvas_m": float(sum(lengths)),
            "n_niveles": int(len(levels)),
            "estado": "Calculado",
        })
        return out
    except Exception as e:
        out["estado"] = f"Error: {e}"
        return out


def _interp_velocity_table(slope_pct: float, table: list[tuple[float, float, float]]) -> float:
    """Pick velocity from range table [(smin,smax,v), ...]."""
    if not _finite(slope_pct):
        return np.nan
    S = float(slope_pct)
    for smin, smax, v in table:
        if S >= smin and S <= smax:
            return float(v)
    if S < table[0][0]:
        return float(table[0][2])
    return float(table[-1][2])


def us_navy_velocity_m_s(slope_pct: float) -> float:
    return _interp_velocity_table(slope_pct, [(1, 2, 0.6), (2, 4, 0.9), (4, 6, 1.2), (6, 10, 1.5), (10, 999, 1.8)])


def texas_velocity_m_s(slope_pct: float, condition: str = "Poca vegetación") -> float:
    cond = (condition or "").lower()
    # Table from El Espino-style method: slope ranges and velocities by cover/cauce condition.
    if "bosque" in cond:
        table = [(0, 3, 0.30), (4, 7, 0.60), (8, 11, 0.90), (12, 15, 1.05), (15, 999, 1.20)]
    elif "cauce" in cond:
        table = [(0, 3, 0.30), (4, 7, 0.90), (8, 11, 1.50), (12, 15, 2.40), (15, 999, 2.70)]
    else:
        table = [(0, 3, 0.45), (4, 7, 0.90), (8, 11, 1.20), (12, 15, 1.35), (15, 999, 1.50)]
    return _interp_velocity_table(slope_pct, table)


def time_concentration_methods(A_km2: float, L_km: float, H_m: float, slope_pct: float, texas_condition: str = "Poca vegetación") -> pd.DataFrame:
    """Return comparative Tc methods in minutes.

    Formulas are implemented as operational estimators and must be reviewed by
    a specialist before design. L is channel length in km, H is hydraulic drop in m.
    """
    rows = []
    A = float(A_km2) if _finite(A_km2) else np.nan
    L = float(L_km) if _finite(L_km) else np.nan
    H = float(H_m) if _finite(H_m) else np.nan
    S_pct = float(slope_pct) if _finite(slope_pct) else np.nan
    L_m = L * 1000.0 if _finite(L) else np.nan
    S_mm = S_pct / 100.0 if _finite(S_pct) else np.nan

    def add(name, tc_min, formula, rango=""):
        rows.append({"metodo_Tc": name, "Tc_min": tc_min, "Tc_h": tc_min / 60.0 if _finite(tc_min) else np.nan, "formula_base": formula, "comentario": rango})

    # Kirpich/SCS, common in app and manuals.
    if _finite(L_m) and _finite(S_mm) and L_m > 0 and S_mm > 0:
        add("Kirpich / SCS", 0.0195 * (L_m ** 0.77) * (S_mm ** -0.385), "0,0195·L^0,77·S^-0,385", "L en m; S=m/m")
    else:
        add("Kirpich / SCS", np.nan, "0,0195·L^0,77·S^-0,385", "Faltan L o S")

    # California culvert style: Tc hours = (0.87 L^3 / H)^0.385, L km, H m.
    if _finite(L) and _finite(H) and L > 0 and H > 0:
        add("California", 60.0 * ((0.87 * (L ** 3) / H) ** 0.385), "(0,87·L³/H)^0,385", "L km; H m")
    else:
        add("California", np.nan, "(0,87·L³/H)^0,385", "Faltan L o H")

    # Giandotti standard operational form: hours = (4 sqrt(A)+1.5 L)/(0.8 sqrt(H)).
    if _finite(A) and _finite(L) and _finite(H) and A > 0 and L > 0 and H > 0:
        add("Giandotti", 60.0 * ((4.0 * math.sqrt(A) + 1.5 * L) / (0.8 * math.sqrt(H))), "(4√A + 1,5L)/(0,8√H)", "A km²; L km; H m")
    else:
        add("Giandotti", np.nan, "(4√A + 1,5L)/(0,8√H)", "Faltan A, L o H")

    # US Navy velocity method.
    if _finite(L_m) and _finite(S_pct) and L_m > 0:
        v = us_navy_velocity_m_s(S_pct)
        add("US Navy velocidad", L_m / v / 60.0, "Tc=L/(v·60)", f"v={v:.2f} m/s según pendiente")
    else:
        add("US Navy velocidad", np.nan, "Tc=L/(v·60)", "Faltan L o pendiente")

    # Texas velocity method.
    if _finite(L_m) and _finite(S_pct) and L_m > 0:
        v = texas_velocity_m_s(S_pct, texas_condition)
        add("Velocidad Texas", L_m / v / 60.0, "Tc=L/(v·60)", f"v={v:.2f} m/s; condición={texas_condition}")
    else:
        add("Velocidad Texas", np.nan, "Tc=L/(v·60)", "Faltan L o pendiente")

    df = pd.DataFrame(rows)
    valid = pd.to_numeric(df["Tc_min"], errors="coerce").dropna()
    if len(valid) >= 2:
        # Robust adoption: median of valid estimates, with min/max for traceability.
        adopted = float(valid.median())
        df.loc[len(df)] = {"metodo_Tc": "Adoptado automático", "Tc_min": adopted, "Tc_h": adopted / 60.0, "formula_base": "mediana de métodos válidos", "comentario": f"rango {valid.min():.1f}–{valid.max():.1f} min"}
    return df


def design_flow_recommendation(results_df: pd.DataFrame, area_km2: float) -> pd.DataFrame:
    """Suggest an adopted design flow by return period.

    It does not replace engineering judgement. It flags preferred method families
    depending on the area and uses a conservative median/high envelope rule.
    """
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    df = df[pd.to_numeric(df.get("Q_m3s"), errors="coerce").notna()].copy()
    if df.empty:
        return pd.DataFrame()
    A = float(area_km2) if _finite(area_km2) else np.nan
    rows = []
    for T, g in df.groupby("T_anios"):
        gg = g.copy()
        if _finite(A) and A >= 20:
            pref = gg[gg["metodo"].str.contains("DGA-AC|Verni-King|HUS", case=False, regex=True, na=False)]
            criterio = "Cuenca ≥20 km²: preferir DGA-AC/Verni-King/HUS; racional como contraste."
        elif _finite(A) and A <= 3:
            pref = gg[gg["metodo"].str.contains("Racional|HUS|SCS", case=False, regex=True, na=False)]
            criterio = "Cuenca pequeña: racional puede ser principal, contrastar con HUS/SCS."
        else:
            pref = gg[gg["metodo"].str.contains("Racional|HUS|SCS|DGA-AC|Verni-King", case=False, regex=True, na=False)]
            criterio = "Cuenca intermedia: adoptar con comparación multimétodo."
        if pref.empty:
            pref = gg
        vals = pd.to_numeric(pref["Q_m3s"], errors="coerce").dropna()
        if vals.empty:
            continue
        q_med = float(vals.median())
        q_p75 = float(vals.quantile(0.75)) if len(vals) >= 3 else float(vals.max())
        q_max = float(vals.max())
        # Adopt a conservative but not blindly maximum value: p75; when only 1-2 methods, use max.
        q_adopt = q_p75 if len(vals) >= 3 else q_max
        rows.append({
            "T_anios": float(T),
            "Q_recomendado_m3s": q_adopt,
            "Q_mediana_m3s": q_med,
            "Q_max_m3s": q_max,
            "n_metodos_preferentes": int(len(vals)),
            "metodos_considerados": ", ".join(pref["metodo"].astype(str).unique()),
            "criterio": criterio,
            "advertencia": "Valor automático de apoyo; el especialista debe adoptar y justificar el caudal definitivo.",
        })
    return pd.DataFrame(rows)


def rainfall_crosscheck_summary(p10d_isohyet: float, pmax24_freq_df: pd.DataFrame) -> pd.DataFrame:
    """Compare isohyet P10D with nearest station frequency values when available."""
    rows = []
    if pmax24_freq_df is None or pmax24_freq_df.empty:
        return pd.DataFrame(rows)
    df = pmax24_freq_df.copy()
    df10 = df[pd.to_numeric(df.get("T_anios"), errors="coerce") == 10.0]
    for _, r in df10.iterrows():
        vals = {c: r.get(c) for c in df10.columns if str(c).startswith("P24_") and pd.notna(r.get(c))}
        for method, val in vals.items():
            diff = float(val) - float(p10d_isohyet) if _finite(p10d_isohyet) and _finite(val) else np.nan
            rows.append({
                "codigo": r.get("codigo"),
                "estacion": r.get("estacion"),
                "distancia_km": r.get("distancia_km"),
                "metodo_frecuencia": method.replace("P24_", ""),
                "P24_T10_estacion_mm": val,
                "P10D_isoyeta_mm": p10d_isohyet,
                "diferencia_mm": diff,
                "diferencia_pct": 100.0 * diff / p10d_isohyet if _finite(p10d_isohyet) and float(p10d_isohyet) > 0 and _finite(diff) else np.nan,
            })
    return pd.DataFrame(rows)
