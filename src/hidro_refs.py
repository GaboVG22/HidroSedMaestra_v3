"""Catalogos hidrologicos chilenos incorporados a la app.

Los valores incluidos son presets editables para apoyar la automatizacion.
Deben ser revisados por el especialista antes de emitir un diseno definitivo.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _interp_factor(table: Dict[float, float], x: float, log_x: bool = True) -> float:
    clean = {float(k): float(v) for k, v in (table or {}).items() if v is not None and not pd.isna(v)}
    if not clean:
        return float("nan")
    x = float(x)
    if x in clean:
        return clean[x]
    xs = np.array(sorted(clean.keys()), dtype=float)
    ys = np.array([clean[k] for k in xs], dtype=float)
    if x <= xs.min():
        return float(ys[0])
    if x >= xs.max():
        return float(ys[-1])
    if log_x and x > 0 and xs.min() > 0:
        return float(np.interp(np.log(x), np.log(xs), ys))
    return float(np.interp(x, xs, ys))


# Coeficientes de duracion y frecuencia extraidos del Manual de Carreteras Vol. 3,
# Tabla 3.702.403.A/B para estaciones representativas de la Region de Coquimbo.
# Duraciones en horas; coeficientes adimensionales.
IDF_REFERENCE_STATIONS = {
    "Manual": {"cd": {}, "cf": {}, "intensities": {}, "source": "Usuario"},
    "Rivadavia - Elqui": {
        "cd": {1:0.12, 2:0.21, 4:0.35, 6:0.48, 8:0.59, 10:0.68, 12:0.75, 14:0.78, 18:0.87, 24:1.00},
        "cf": {2:0.49, 5:0.80, 10:1.00, 20:1.19, 25:1.26, 50:1.44, 100:1.63, 200:1.82},
        "intensities": {
            10:{1:9.73,2:8.35,4:7.10,6:6.49,8:6.03,10:5.55,12:5.05,14:4.53,18:3.92,24:3.38},
            25:{1:11.97,2:10.30,4:8.78,6:8.08,8:7.55,10:6.96,12:6.35,14:5.70,18:4.96,24:4.29},
            50:{1:13.62,2:11.74,4:10.03,6:9.26,8:8.68,10:8.01,12:7.32,14:6.57,18:5.73,24:4.96},
            100:{1:15.27,2:13.18,4:11.28,6:10.43,8:9.80,10:9.06,12:8.28,14:7.44,18:6.49,24:5.63},
        },
        "source": "Manual de Carreteras Vol. 3, tablas 3.702.402.A y 3.702.403.A/B",
    },
    "La Paloma - Limari": {
        "cd": {1:0.15, 2:0.25, 4:0.41, 6:0.55, 8:0.65, 10:0.74, 12:0.80, 14:0.84, 18:0.92, 24:1.00},
        "cf": {2:0.48, 5:0.79, 10:1.00, 20:1.20, 25:1.26, 50:1.46, 100:1.65, 200:1.84},
        "intensities": {
            10:{1:11.65,2:9.87,4:8.04,6:7.16,8:6.31,10:5.73,12:5.18,14:4.69,18:3.96,24:3.24},
            25:{1:14.34,2:12.13,4:9.96,6:8.95,8:7.92,10:7.22,12:6.55,14:5.94,18:5.04,24:4.16},
            50:{1:16.33,2:13.81,4:11.39,6:10.28,8:9.11,10:8.33,12:7.57,14:6.87,18:5.85,24:4.83},
            100:{1:18.31,2:15.47,4:12.80,6:11.59,8:10.30,10:9.44,12:8.58,14:7.79,18:6.64,24:5.51},
        },
        "source": "Manual de Carreteras Vol. 3, tablas 3.702.402.A y 3.702.403.A/B",
    },
    "Illapel - Choapa": {
        "cd": {1:0.14, 2:0.25, 4:0.41, 6:0.54, 8:0.66, 10:0.73, 12:0.79, 14:0.84, 18:0.92, 24:1.00},
        "cf": {2:0.50, 5:0.80, 10:1.00, 20:1.19, 25:1.25, 50:1.44, 100:1.62, 200:1.81},
        "intensities": {
            10:{1:10.46,2:9.21,4:7.68,6:6.73,8:6.12,10:5.46,12:4.88,14:4.44,18:3.79,24:3.09},
            25:{1:12.90,2:11.34,4:9.46,6:8.35,8:7.65,10:6.83,12:6.11,14:5.57,18:4.78,24:3.92},
            50:{1:14.71,2:12.93,4:10.78,6:9.56,8:8.78,10:7.85,12:7.02,14:6.41,18:5.51,24:4.53},
            100:{1:16.51,2:14.50,4:12.09,6:10.76,8:9.90,10:8.86,12:7.93,14:7.24,18:6.23,24:5.14},
        },
        "source": "Manual de Carreteras Vol. 3, tablas 3.702.402.A y 3.702.403.A/B",
    },
}


def duration_coefficient(station: str, duration_h: float) -> float:
    data = IDF_REFERENCE_STATIONS.get(station, IDF_REFERENCE_STATIONS["Manual"])
    return _interp_factor(data.get("cd", {}), float(duration_h), log_x=True)


def frequency_coefficient(station: str, T_years: float) -> float:
    data = IDF_REFERENCE_STATIONS.get(station, IDF_REFERENCE_STATIONS["Manual"])
    return _interp_factor(data.get("cf", {}), float(T_years), log_x=True)


def intensity_from_p10d(station: str, p10d_mm: float, T_years: float, duration_h: float, k_24h: float = 1.10, area_reduction: float = 1.0) -> float:
    """Estimate intensity from daily 10-year rainfall using CD/CF/K.

    P_t^T = P10D * CD_t * CF_T * K * CA; I = P_t^T / t
    """
    if p10d_mm is None or pd.isna(p10d_mm) or p10d_mm <= 0 or duration_h <= 0:
        return float("nan")
    cd = duration_coefficient(station, duration_h)
    cf = frequency_coefficient(station, T_years)
    if pd.isna(cd) or pd.isna(cf):
        return float("nan")
    return float(p10d_mm) * float(cd) * float(cf) * float(k_24h) * float(area_reduction) / float(duration_h)


def p24_from_p10d(station: str, p10d_mm: float, T_years: float, k_24h: float = 1.10, area_reduction: float = 1.0) -> float:
    if p10d_mm is None or pd.isna(p10d_mm) or p10d_mm <= 0:
        return float("nan")
    cf = frequency_coefficient(station, T_years)
    if pd.isna(cf):
        return float("nan")
    # CD_24 = 1.0
    return float(p10d_mm) * float(cf) * float(k_24h) * float(area_reduction)


def direct_idf_intensity(station: str, T_years: float, duration_h: float) -> float:
    data = IDF_REFERENCE_STATIONS.get(station, {})
    intens = data.get("intensities", {})
    if not intens:
        return float("nan")
    # Interpolate in duration for tabulated return periods. If T is not tabulated
    # (e.g., 2, 5 or 200 years), scale the T=10 IDF by CF_T/CF_10 when CF is available.
    T_req = float(T_years)
    vals_by_T = {}
    for T, durtab in intens.items():
        vals_by_T[float(T)] = _interp_factor(durtab, float(duration_h), log_x=True)
    if T_req in vals_by_T:
        return float(vals_by_T[T_req])
    cf_req = frequency_coefficient(station, T_req)
    cf_10 = frequency_coefficient(station, 10.0)
    if 10.0 in vals_by_T and not pd.isna(cf_req) and not pd.isna(cf_10) and cf_10 > 0:
        return float(vals_by_T[10.0]) * float(cf_req) / float(cf_10)
    return _interp_factor(vals_by_T, T_req, log_x=True)


def rainfall_reference_dataframe(station: str) -> pd.DataFrame:
    data = IDF_REFERENCE_STATIONS.get(station, {})
    rows = []
    for d, cd in (data.get("cd", {}) or {}).items():
        rows.append({"tipo": "CD", "periodo_o_duracion": d, "valor": cd})
    for T, cf in (data.get("cf", {}) or {}).items():
        rows.append({"tipo": "CF", "periodo_o_duracion": T, "valor": cf})
    return pd.DataFrame(rows)


# Hidrograma unitario sintético adimensional del Manual de Carreteras Vol. 2.
HUS_DISTRIBUTION = pd.DataFrame({
    "t_tp": [0, 0.30, 0.50, 0.60, 0.75, 1.00, 1.30, 1.50, 1.80, 2.30, 2.70],
    "q_qp": [0, 0.20, 0.40, 0.60, 0.80, 1.00, 0.80, 0.60, 0.40, 0.20, 0.10],
})


METHODOLOGY_REFERENCES = pd.DataFrame([
    {"documento": "DGA-HUMED32", "aporte_app": "P24,10, coeficientes de duración/frecuencia, zonas homogéneas e isoyetas para estimar lluvia de diseño."},
    {"documento": "FLU398 DGA 1995", "aporte_app": "DGA-AC, Verni-King modificado, racional, HUS Linsley/Gray, CN y límites de uso en cuencas sin información fluviométrica."},
    {"documento": "Manual de Carreteras Vol. 3", "aporte_app": "Curvas IDF, CD/CF, K=1,1, fórmula racional, coeficientes de escorrentía y recomendación de métodos DGA."},
    {"documento": "FLU590", "aporte_app": "Validación de eventos extremos, selección de estaciones BNA, análisis regional y respaldo de coeficientes empíricos."},
    {"documento": "DGA041 / CRH 93-81 Antofagasta 1991", "aporte_app": "Módulo de caudal líquido, caudal detrítico, concentraciones volumétricas, verificación hidráulico-hidrológica y estimación de volúmenes de detritos."},
    {"documento": "Flujos detríticos - Tamburrino", "aporte_app": "Clasificación reológica, régimen macroviscoso/inercial, relaciones empíricas Qp=f(M), productividad de cuencas y cauces, y advertencias por alta dispersión."},
    {"documento": "Caudal detrítico / IDIEM-FLO-2D", "aporte_app": "Conversión QD=QL+QS, QD=QL/(1-Cv), factor de carga, coeficiente K, categorías por concentración volumétrica y ecuación de Takahashi."},
    {"documento": "Estudio Hidrología e Hidráulica Ruta 41-CH", "aporte_app": "Secuencia profesional: estaciones, curvas dobles acumuladas, Weibull, IDF, racional, DGA-AC, Verni-King y verificación hidráulica de obras."},
    {"documento": "Anexo M - Análisis de Crecidas", "aporte_app": "Uso integrado de isoyetas DGA, P24,10, DGA-AC, zona homogénea Kp, factores de frecuencia, factor instantáneo y HEC-RAS como verificación hidráulica."},
    {"documento": "El Espino / Obras en quebradas", "aporte_app": "Pendiente de Mociornita, tiempos de concentración California, Giandotti, US Navy y Velocidad Texas; adopción del Tc por comparación de métodos."},
    {"documento": "Estudio Hidrológico Canal Buzeta", "aporte_app": "Relleno de series, correlaciones entre estaciones, curvas de variación estacional, disponibilidad en bocatoma y curvas IDF."},
    {"documento": "Manual de Carreteras Vol. 2 y Vol. 3", "aporte_app": "Análisis de frecuencia, métodos gráficos y analíticos, Gumbel/Log-Pearson/Lognormal, criterios de crecida de diseño y drenaje vial."},
    {"documento": "Curvas IDF Pizarro et al.", "aporte_app": "Definición formal de curvas IDF, uso de Gumbel para máximos, y metodología de extensión por coeficientes generalizados de duración y frecuencia cuando sólo hay lluvia diaria."},
    {"documento": "Tesis Verni-King CR2MET/CAMELS-CL", "aporte_app": "Refuerza cálculo por área pluvial, línea de nieve, uso de precipitación media diaria distribuida, comparación de métodos de línea de nieve y análisis de frecuencia de P24 por cuenca."},
    {"documento": "Manual de Carreteras Vol. 2 edición 2025", "aporte_app": "Riesgo de falla R=1-(1-1/T)^n, uso de varios métodos independientes, análisis de sensibilidad, métodos directos/regionales/indirectos y verificación de calidad de datos."},
    {"documento": "Manual de Carreteras Vol. 3 edición 2025", "aporte_app": "Tiempos de concentración, curvas IDF, coeficientes de escurrimiento, método racional, SCS-CN y criterios de diseño/verificación para obras de drenaje y puentes."},
    {"documento": "El Espino Diseño Canales de Contorno", "aporte_app": "Módulo hidráulico preliminar para canales: Manning, rugosidad n, taludes, revancha, estabilidad, velocidades, transiciones, curvas y verificación de arrastre de sólidos."},
])
