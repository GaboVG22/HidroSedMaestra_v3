"""Módulos de diseño hidrológico-hidráulico complementarios.

Versión v2.4: incorpora criterios usados en Manual de Carreteras y memorias
hidrológicas/hidráulicas: riesgo de falla, área pluvial por línea de nieve,
verificación hidráulica de canales trapezoidales por Manning y matrices de
aplicabilidad/advertencias.

Todos los resultados son de apoyo técnico; el especialista debe revisar datos,
coeficientes y aplicabilidad antes de emitir diseño definitivo.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


def _finite(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Riesgo, vida útil y período de retorno
# -----------------------------------------------------------------------------

def risk_of_failure(T_years: float, life_years: float) -> float:
    """Riesgo de excedencia durante la vida útil: R = 1 - (1 - 1/T)^n."""
    if not _finite(T_years) or not _finite(life_years):
        return float("nan")
    T = float(T_years)
    n = float(life_years)
    if T <= 1 or n <= 0:
        return float("nan")
    return float(1.0 - (1.0 - 1.0 / T) ** n)


def return_period_for_risk(risk: float, life_years: float) -> float:
    """Período de retorno requerido para un riesgo R y vida útil n."""
    if not _finite(risk) or not _finite(life_years):
        return float("nan")
    R = float(risk)
    n = float(life_years)
    if R <= 0 or R >= 1 or n <= 0:
        return float("nan")
    return float(1.0 / (1.0 - (1.0 - R) ** (1.0 / n)))


def risk_table(return_periods: Iterable[float], life_years: float) -> pd.DataFrame:
    rows = []
    for T in return_periods:
        rows.append({"T_anios": float(T), "vida_util_anios": float(life_years), "riesgo_excedencia": risk_of_failure(T, life_years)})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Línea de nieve / área pluvial efectiva
# -----------------------------------------------------------------------------

def pluvial_area_linear(area_km2: float, z_min_m: float, z_max_m: float, snowline_m: float) -> Dict[str, float | str]:
    """Aproximación de área pluvial efectiva a partir de una curva hipsométrica lineal.

    Si la línea de nieve está sobre la cota máxima, toda la cuenca se considera pluvial.
    Si está bajo la cota mínima, no hay área pluvial líquida. Entre ambas cotas se
    aproxima la fracción areal linealmente.
    """
    out = {"area_total_km2": area_km2, "linea_nieve_m": snowline_m, "fraccion_pluvial": np.nan, "area_pluvial_km2": np.nan, "estado": "No calculado"}
    if not (_finite(area_km2) and _finite(z_min_m) and _finite(z_max_m) and _finite(snowline_m)):
        out["estado"] = "Faltan área, cotas o línea de nieve"
        return out
    A = float(area_km2)
    zmin = float(z_min_m)
    zmax = float(z_max_m)
    snow = float(snowline_m)
    if A <= 0 or zmax <= zmin:
        out["estado"] = "Cotas insuficientes"
        return out
    if snow >= zmax:
        f = 1.0
        estado = "Toda la cuenca bajo línea de nieve: área pluvial completa"
    elif snow <= zmin:
        f = 0.0
        estado = "Línea de nieve bajo salida: área pluvial líquida nula/preliminar"
    else:
        f = (snow - zmin) / (zmax - zmin)
        estado = "Área pluvial estimada por hipsometría lineal entre cota mínima y máxima"
    out.update({"fraccion_pluvial": float(max(0.0, min(1.0, f))), "area_pluvial_km2": float(A * max(0.0, min(1.0, f))), "estado": estado})
    return out


# -----------------------------------------------------------------------------
# Manning para canal trapezoidal / verificación hidráulica
# -----------------------------------------------------------------------------

def trapezoid_geometry(b_m: float, y_m: float, z_hv: float) -> Dict[str, float]:
    """Geometría de canal trapezoidal con talud z:1 H:V."""
    b = float(b_m)
    y = float(y_m)
    z = float(z_hv)
    A = y * (b + z * y)
    P = b + 2.0 * y * math.sqrt(1.0 + z * z)
    T = b + 2.0 * z * y
    R = A / P if P > 0 else float("nan")
    D = A / T if T > 0 else float("nan")
    return {"area_m2": A, "perimetro_mojado_m": P, "ancho_superior_m": T, "radio_hidraulico_m": R, "profundidad_hidraulica_m": D}


def manning_discharge(b_m: float, y_m: float, z_hv: float, n: float, slope_m_m: float) -> float:
    if min(float(b_m), float(y_m), float(n), float(slope_m_m)) <= 0:
        return float("nan")
    g = trapezoid_geometry(b_m, y_m, z_hv)
    return float((1.0 / float(n)) * g["area_m2"] * (g["radio_hidraulico_m"] ** (2.0 / 3.0)) * (float(slope_m_m) ** 0.5))


def solve_normal_depth(Q_m3s: float, b_m: float, z_hv: float, n: float, slope_m_m: float, y_min: float = 0.02, y_max: float = 20.0) -> float:
    """Resuelve y normal por bisección para flujo uniforme Manning."""
    if not all(_finite(v) for v in [Q_m3s, b_m, z_hv, n, slope_m_m]) or float(Q_m3s) <= 0:
        return float("nan")
    lo, hi = float(y_min), float(y_max)
    for _ in range(80):
        if manning_discharge(b_m, hi, z_hv, n, slope_m_m) >= float(Q_m3s):
            break
        hi *= 1.5
        if hi > 200:
            return float("nan")
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        qmid = manning_discharge(b_m, mid, z_hv, n, slope_m_m)
        if not _finite(qmid):
            return float("nan")
        if qmid < float(Q_m3s):
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def hydraulic_verification_table(flow_df: pd.DataFrame, b_m: float, z_hv: float, n: float, slope_pct: float, freeboard_m: float = 0.30, design_depth_m: Optional[float] = None) -> pd.DataFrame:
    """Verifica una sección trapezoidal para una tabla de caudales.

    flow_df puede venir de caudal recomendado (Q_recomendado_m3s) o de resultados por método (Q_m3s).
    """
    if flow_df is None or flow_df.empty:
        return pd.DataFrame()
    q_col = "Q_recomendado_m3s" if "Q_recomendado_m3s" in flow_df.columns else "Q_m3s"
    if q_col not in flow_df.columns:
        return pd.DataFrame()
    S = float(slope_pct) / 100.0
    rows = []
    for _, r in flow_df.iterrows():
        Q = pd.to_numeric(pd.Series([r.get(q_col)]), errors="coerce").iloc[0]
        if pd.isna(Q) or Q <= 0:
            continue
        y = solve_normal_depth(float(Q), b_m, z_hv, n, S)
        geom = trapezoid_geometry(b_m, y, z_hv) if _finite(y) else {"area_m2": np.nan, "perimetro_mojado_m": np.nan, "ancho_superior_m": np.nan, "radio_hidraulico_m": np.nan, "profundidad_hidraulica_m": np.nan}
        V = float(Q) / geom["area_m2"] if _finite(geom.get("area_m2")) and geom["area_m2"] > 0 else np.nan
        Fr = V / math.sqrt(9.81 * geom["profundidad_hidraulica_m"]) if _finite(V) and _finite(geom.get("profundidad_hidraulica_m")) and geom["profundidad_hidraulica_m"] > 0 else np.nan
        tau = 1000.0 * 9.81 * geom["radio_hidraulico_m"] * S if _finite(geom.get("radio_hidraulico_m")) and S > 0 else np.nan
        h_total = y + float(freeboard_m) if _finite(y) else np.nan
        capacidad_con_altura = np.nan
        estado_altura = "No evaluado"
        if design_depth_m is not None and _finite(design_depth_m) and float(design_depth_m) > 0:
            capacidad_con_altura = manning_discharge(b_m, max(float(design_depth_m) - float(freeboard_m), 0.02), z_hv, n, S)
            estado_altura = "OK" if capacidad_con_altura >= float(Q) else "Insuficiente"
        rows.append({
            "T_anios": r.get("T_anios", np.nan),
            "metodo_origen": r.get("metodo", "Caudal recomendado"),
            "Q_m3s": float(Q),
            "b_m": float(b_m),
            "talud_HV": float(z_hv),
            "n_Manning": float(n),
            "pendiente_pct": float(slope_pct),
            "y_normal_m": y,
            "revancha_m": float(freeboard_m),
            "altura_total_requerida_m": h_total,
            "area_m2": geom["area_m2"],
            "radio_hidraulico_m": geom["radio_hidraulico_m"],
            "ancho_superior_m": geom["ancho_superior_m"],
            "velocidad_m_s": V,
            "Froude": Fr,
            "tipo_regimen": "Supercrítico" if _finite(Fr) and Fr > 1 else ("Subcrítico" if _finite(Fr) else "No calculado"),
            "esfuerzo_corte_Pa": tau,
            "capacidad_con_altura_m3s": capacidad_con_altura,
            "estado_altura_diseno": estado_altura,
            "advertencia": "Revisar transiciones, curvas, pérdida singular, erosión y arrastre de sólidos; Manning uniforme no reemplaza modelación hidráulica detallada.",
        })
    return pd.DataFrame(rows)


def objective_design_guidance(objective: str) -> str:
    txt = (objective or "").lower()
    if "alcantarilla" in txt or "atravieso" in txt:
        return "Para atraviesos/alcantarillas conviene reportar caudal de diseño y caudal de verificación, revancha, régimen de escurrimiento, control de entrada/salida y riesgo de obstrucción por sólidos."
    if "canal" in txt:
        return "Para canales de contorno se debe verificar capacidad por Manning, revancha, estabilidad del escurrimiento, velocidades, transiciones, curvas y capacidad de arrastre/retención de sólidos."
    if "embalse" in txt:
        return "Para regulación/embalses pequeños se requiere hidrograma, volumen de escorrentía, volumen detrítico y revisión de vertedero/obra de descarga."
    if "urban" in txt or "aguas lluvias" in txt:
        return "Para aguas lluvias urbanas se debe reforzar IDF, tiempo de concentración, coeficientes de escorrentía, porcentaje impermeable y sensibilidad."
    return "Para quebradas naturales se recomienda cálculo multimétodo, análisis de incertidumbre, potencial detrítico, volumen de sedimentos y validación de terreno."
