"""Motor avanzado de hidráulica fluvial y sedimentos para HidroSed Maestra.

Este módulo agrega una capa de revisión tipo HEC-RAS/experta sobre la geometría
transferida desde v13:
- cálculo hidráulico con geometría transversal real (tabla 04_Puntos_Seccion),
  no solo sección trapecial simplificada;
- solución automática de tirante normal por Manning;
- estimación de calado crítico, régimen y energía específica;
- predictores de transporte de sedimentos de fondo y total;
- rugosidad de gravas por varios criterios;
- auditoría de calidad geométrica, hidráulica, estabilidad y sensibilidad.

No pretende reemplazar una modelación 2D calibrada cuando el problema exige
bidimensionalidad, pero sí obliga a una trazabilidad más estricta que un cálculo
manual de secciones aisladas.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable, Any

import numpy as np
import pandas as pd

G = 9.81
EPS = 1e-9


def _sf(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", ".").strip()
            if value == "":
                return default
        val = float(value)
        if not math.isfinite(val):
            return default
        return val
    except Exception:
        return default


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        key = str(name).strip().lower()
        if key in lookup:
            return lookup[key]
    return None


def water_density_kgm3(temp_c: float = 20.0) -> float:
    """Densidad del agua dulce en kg/m3, ecuación empírica usual 0-40 °C."""
    T = _sf(temp_c, 20.0)
    return 1000.0 * (1.0 - ((T + 288.9414) / (508929.2 * (T + 68.12963))) * ((T - 3.9863) ** 2))


def water_dynamic_viscosity_pa_s(temp_c: float = 20.0) -> float:
    """Viscosidad dinámica aproximada del agua dulce, Pa*s."""
    T = max(-5.0, min(60.0, _sf(temp_c, 20.0)))
    return 2.414e-5 * 10 ** (247.8 / (T + 133.15))


def roughness_estimators(d50_mm: float, d75_mm: float | None = None, d90_mm: float | None = None) -> dict[str, float]:
    """Rugosidad Manning para lechos granulares, D en metros.

    Incluye expresiones resumidas en el documento de transporte de sedimentos:
    Chow, Garde & Raju, Anderson, Simons-Li y Meyer-Peter-Müller.
    """
    d50 = max(_sf(d50_mm, 0.0) / 1000.0, EPS)
    d75 = max(_sf(d75_mm, d50_mm) / 1000.0, EPS)
    d90 = max(_sf(d90_mm, d50_mm) / 1000.0, EPS)
    vals = {
        "n_Chow_D50": 0.040 * d50 ** (1.0 / 6.0),
        "n_Garde_Raju_D50": 0.047 * d50 ** (1.0 / 6.0),
        "n_Anderson_D90": (39.3701 * d90) ** (1.0 / 6.0) / 44.4,
        "n_Simons_Li_D75": (39.3701 * d75) ** (1.0 / 6.0) / 39.0,
        "n_MPM_D90": 0.038 * d90 ** (1.0 / 6.0),
    }
    vals["n_recomendado_grano_mediana"] = float(np.nanmedian(list(vals.values())))
    return vals


def _normalize_points(df_puntos: pd.DataFrame) -> pd.DataFrame:
    pts = pd.DataFrame(df_puntos).copy()
    if pts.empty:
        return pts
    rename = {}
    mapping = {
        "id_seccion": ["id_seccion", "ID_Seccion", "ID", "section_id", "Nombre_Seccion"],
        "pk_m": ["pk_m", "PK_m", "Distancia_m"],
        "distancia_transversal_m": ["distancia_transversal_m", "Distancia_Transversal_m", "estacion_m", "station_m"],
        "x_utm": ["x_utm", "X_UTM", "x", "este_m"],
        "y_utm": ["y_utm", "Y_UTM", "y", "norte_m"],
        "z_m": ["z_m", "Z_m", "cota_m", "elevacion_m", "Cota_m"],
        "ribera": ["ribera", "Ribera"],
        "tipo_punto": ["tipo_punto", "Tipo_Punto"],
    }
    lower_cols = {str(c).strip().lower(): c for c in pts.columns}
    for out, candidates in mapping.items():
        for c in candidates:
            if c in pts.columns:
                rename[c] = out
                break
            if c.lower() in lower_cols:
                rename[lower_cols[c.lower()]] = out
                break
    pts = pts.rename(columns=rename)
    for col in ["pk_m", "distancia_transversal_m", "x_utm", "y_utm", "z_m"]:
        if col in pts.columns:
            pts[col] = pd.to_numeric(pts[col], errors="coerce")
    if "id_seccion" in pts.columns:
        pts["id_seccion"] = pts["id_seccion"].astype(str)
    if "distancia_transversal_m" in pts.columns:
        pts = pts.sort_values(["id_seccion", "distancia_transversal_m"] if "id_seccion" in pts.columns else ["distancia_transversal_m"])
    return pts


def _normalize_sections(df_secciones: pd.DataFrame) -> pd.DataFrame:
    sec = pd.DataFrame(df_secciones).copy()
    if sec.empty:
        return sec
    mapping = {
        "id_seccion": ["ID_Seccion", "id_seccion", "ID", "section_id"],
        "nombre_seccion": ["Nombre_Seccion", "nombre_seccion"],
        "pk_m": ["PK_m", "pk_m", "Distancia_m"],
        "pk_km": ["PK_km", "pk_km"],
        "q_m3s": ["Caudal_m3s", "Q_m3s", "q_m3s"],
        "slope": ["Pendiente_m_m", "S_m_m", "slope", "pendiente_local"],
        "manning_n": ["Manning_n", "n", "manning_n"],
        "ancho_m": ["Ancho_Cauce_m", "B_m", "bottom_width_m", "ancho_superior"],
        "cota_fondo_m": ["Cota_Fondo_m", "Cota_fondo_m", "bed_elevation_m", "cota_fondo"],
        "profundidad_m": ["Profundidad_Max_m", "y_m", "depth_m", "profundidad_max"],
        "d50_mm": ["D50_mm", "d50_mm"],
        "d84_mm": ["D84_mm", "d84_mm"],
        "d90_mm": ["D90_mm", "d90_mm"],
        "dm_mm": ["Dm_mm", "dm_mm"],
    }
    lower_cols = {str(c).strip().lower(): c for c in sec.columns}
    rename = {}
    for out, candidates in mapping.items():
        for c in candidates:
            if c in sec.columns:
                rename[c] = out
                break
            if c.lower() in lower_cols:
                rename[lower_cols[c.lower()]] = out
                break
    sec = sec.rename(columns=rename)
    for c in ["pk_m", "pk_km", "q_m3s", "slope", "manning_n", "ancho_m", "cota_fondo_m", "profundidad_m", "d50_mm", "d84_mm", "d90_mm", "dm_mm"]:
        if c in sec.columns:
            sec[c] = pd.to_numeric(sec[c], errors="coerce")
    if "id_seccion" in sec.columns:
        sec["id_seccion"] = sec["id_seccion"].astype(str)
    if "nombre_seccion" not in sec.columns and "id_seccion" in sec.columns:
        sec["nombre_seccion"] = sec["id_seccion"].astype(str)
    return sec


def section_properties_from_wse(points: pd.DataFrame, wse_m: float) -> dict[str, float]:
    """Propiedades hidráulicas de una sección irregular para una cota de agua."""
    g = pd.DataFrame(points).dropna(subset=["distancia_transversal_m", "z_m"]).copy()
    if len(g) < 2:
        return {"A_m2": 0.0, "P_m": 0.0, "T_m": 0.0, "R_m": 0.0, "Dh_m": 0.0, "y_max_m": 0.0}
    g = g.sort_values("distancia_transversal_m")
    x = g["distancia_transversal_m"].to_numpy(dtype=float)
    z = g["z_m"].to_numpy(dtype=float)
    area = 0.0
    wetted = 0.0
    top_width = 0.0
    for i in range(len(x) - 1):
        x1, x2 = x[i], x[i + 1]
        z1, z2 = z[i], z[i + 1]
        if not np.isfinite([x1, x2, z1, z2]).all() or x2 <= x1:
            continue
        h1 = wse_m - z1
        h2 = wse_m - z2
        dx = x2 - x1
        if h1 <= 0 and h2 <= 0:
            continue
        if h1 > 0 and h2 > 0:
            area += 0.5 * (h1 + h2) * dx
            wetted += math.hypot(dx, z2 - z1)
            top_width += dx
        else:
            # Intersección del segmento terreno con la lámina de agua.
            if abs(z2 - z1) < EPS:
                continue
            t = (wse_m - z1) / (z2 - z1)
            t = max(0.0, min(1.0, t))
            xi = x1 + t * dx
            if h1 > 0:
                dxs = max(0.0, xi - x1)
                area += 0.5 * h1 * dxs
                wetted += math.hypot(dxs, wse_m - z1)
                top_width += dxs
            elif h2 > 0:
                dxs = max(0.0, x2 - xi)
                area += 0.5 * h2 * dxs
                wetted += math.hypot(dxs, z2 - wse_m)
                top_width += dxs
    R = area / max(wetted, EPS)
    Dh = area / max(top_width, EPS)
    return {
        "A_m2": float(area),
        "P_m": float(wetted),
        "T_m": float(top_width),
        "R_m": float(R),
        "Dh_m": float(Dh),
        "y_max_m": float(max(0.0, wse_m - np.nanmin(z))),
    }


def manning_capacity(points: pd.DataFrame, wse_m: float, n: float, slope: float) -> float:
    props = section_properties_from_wse(points, wse_m)
    if props["A_m2"] <= 0 or props["R_m"] <= 0:
        return 0.0
    return (1.0 / max(n, EPS)) * props["A_m2"] * (props["R_m"] ** (2.0 / 3.0)) * math.sqrt(max(slope, EPS))


def solve_normal_wse(points: pd.DataFrame, q_m3s: float, n: float, slope: float) -> tuple[float, dict[str, float], bool]:
    g = pd.DataFrame(points).dropna(subset=["distancia_transversal_m", "z_m"]).sort_values("distancia_transversal_m")
    if len(g) < 2:
        raise ValueError("La sección necesita al menos 2 puntos para hidráulica irregular.")
    q = max(_sf(q_m3s, 0.0), EPS)
    zmin = float(g["z_m"].min())
    zmax = float(g["z_m"].max())
    width = max(float(g["distancia_transversal_m"].max() - g["distancia_transversal_m"].min()), 1.0)
    lo = zmin + 1e-5
    hi = max(zmax + 0.5, zmin + 0.5)
    # expandir cota superior hasta encerrar el caudal
    for _ in range(80):
        if manning_capacity(g, hi, n, slope) >= q:
            break
        hi += max(0.5, 0.05 * width, (hi - lo) * 0.5)
    bracketed = manning_capacity(g, hi, n, slope) >= q
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        qm = manning_capacity(g, mid, n, slope)
        if qm < q:
            lo = mid
        else:
            hi = mid
    wse = 0.5 * (lo + hi)
    return wse, section_properties_from_wse(g, wse), bool(bracketed)


def _critical_function(points: pd.DataFrame, wse_m: float, q_m3s: float) -> float:
    props = section_properties_from_wse(points, wse_m)
    A = max(props["A_m2"], EPS)
    T = max(props["T_m"], EPS)
    return (q_m3s ** 2) * T / (G * A ** 3) - 1.0


def solve_critical_wse(points: pd.DataFrame, q_m3s: float) -> tuple[float, dict[str, float]]:
    g = pd.DataFrame(points).dropna(subset=["distancia_transversal_m", "z_m"]).sort_values("distancia_transversal_m")
    zmin = float(g["z_m"].min())
    zmax = float(g["z_m"].max())
    width = max(float(g["distancia_transversal_m"].max() - g["distancia_transversal_m"].min()), 1.0)
    lo = zmin + 1e-4
    hi = max(zmax + 1.0, zmin + 1.0)
    # ampliar hasta que Fr^2-1 sea negativo (subcrítico) arriba
    for _ in range(80):
        if _critical_function(g, hi, q_m3s) < 0:
            break
        hi += max(0.5, 0.05 * width, (hi - lo) * 0.5)
    # bisección si hay cruce, si no, búsqueda de mínimo de |f|
    flo = _critical_function(g, lo, q_m3s)
    fhi = _critical_function(g, hi, q_m3s)
    if flo * fhi <= 0:
        for _ in range(90):
            mid = 0.5 * (lo + hi)
            fm = _critical_function(g, mid, q_m3s)
            if flo * fm <= 0:
                hi = mid
                fhi = fm
            else:
                lo = mid
                flo = fm
        wse = 0.5 * (lo + hi)
    else:
        grid = np.linspace(lo, hi, 200)
        vals = np.array([abs(_critical_function(g, z, q_m3s)) for z in grid])
        wse = float(grid[int(np.nanargmin(vals))])
    return wse, section_properties_from_wse(g, wse)


@dataclass
class AdvancedModelParams:
    rho_w: float = 1000.0
    rho_s: float = 2650.0
    theta_c: float = 0.047
    porosity: float = 0.35
    temp_c: float = 20.0
    alpha_velocity: float = 1.0
    sediment_supply_factor: float = 1.0
    scour_safety_factor: float = 1.0
    use_roughness_correction: bool = True
    contraction_coeff: float = 1.0
    expansion_coeff: float = 1.0
    boundary_condition: str = "Normal depth"


def _sediment_predictors(q: float, props: dict[str, float], slope: float, n: float, d50_mm: float, d90_mm: float, params: AdvancedModelParams) -> dict[str, float]:
    A = max(props["A_m2"], EPS)
    R = max(props["R_m"], EPS)
    B = max(props["T_m"], EPS)
    V = q / A
    d50 = max(_sf(d50_mm, 32.0) / 1000.0, EPS)
    d90 = max(_sf(d90_mm, d50_mm) / 1000.0, EPS)
    rho_w = max(params.rho_w, EPS)
    rho_s = max(params.rho_s, rho_w + EPS)
    s = rho_s / rho_w
    tau = rho_w * G * R * max(slope, EPS)
    theta = tau / max((rho_s - rho_w) * G * d50, EPS)
    tau_c = max(params.theta_c, EPS) * (rho_s - rho_w) * G * d50
    theta_eff = theta
    if params.use_roughness_correction:
        ks_total = 1.0 / max(n, EPS)
        kr_grain = 26.0 / max(d90 ** (1.0 / 6.0), EPS)
        theta_eff = theta * (ks_total / max(kr_grain, EPS)) ** 1.5
    excess = max(0.0, theta_eff - max(params.theta_c, EPS))
    qb_mpm_m2s = 8.0 * (excess ** 1.5) * math.sqrt(max((s - 1.0) * G * d50 ** 3, 0.0))
    qb_mpm_m3s = qb_mpm_m2s * B * max(params.sediment_supply_factor, 0.0)
    # Engelund-Hansen carga total específica, sensible a V^5. C = Chezy = R^(1/6)/n.
    C_chezy = (R ** (1.0 / 6.0)) / max(n, EPS)
    qt_eh_m2s = 0.05 * (V ** 5) / max(((s - 1.0) ** 2) * math.sqrt(G) * d50 * (C_chezy ** 3), EPS)
    qt_eh_m3s = qt_eh_m2s * B * max(params.sediment_supply_factor, 0.0)
    return {
        "V_m_s": V,
        "tau_Pa": tau,
        "tau_c_Pa": tau_c,
        "theta": theta,
        "theta_eff": theta_eff,
        "theta_c": params.theta_c,
        "Indice_movilidad": theta_eff / max(params.theta_c, EPS),
        "qb_MPM_m2s": qb_mpm_m2s,
        "Gs_MPM_m3s": qb_mpm_m3s,
        "Gs_MPM_tonh": qb_mpm_m3s * 3600.0 * rho_s / 1000.0,
        "qt_EngelundHansen_m2s": qt_eh_m2s,
        "Gs_EngelundHansen_m3s": qt_eh_m3s,
        "Gs_EngelundHansen_tonh": qt_eh_m3s * 3600.0 * rho_s / 1000.0,
        "C_Chezy": C_chezy,
        "D50_m": d50,
        "D90_m": d90,
    }


def _simple_scour(y: float, q: float, props: dict[str, float], d50_mm: float, tr_years: float, params: AdvancedModelParams) -> dict[str, float]:
    # Estimador conservador preliminar combinado: exceso de movilidad y factor de avenida.
    B = max(props.get("T_m", 0.0), EPS)
    q_unit = q / B
    d50 = max(_sf(d50_mm, 32.0) / 1000.0, EPS)
    Fr = q / max(props["A_m2"], EPS) / math.sqrt(G * max(props["Dh_m"], EPS))
    tr_factor = 1.0 + 0.08 * max(0.0, math.log10(max(tr_years, 1.01)))
    grain_factor = max(0.45, min(2.2, (0.03 / d50) ** 0.18))
    froude_factor = 1.0 + max(0.0, Fr - 0.8) * 0.55
    # q_unit sirve como indicador: evita entregar cero y escala suavemente.
    q_factor = max(0.30, min(2.50, (q_unit / 5.0) ** 0.20))
    scour = max(0.0, y * 0.18 * tr_factor * grain_factor * froude_factor * q_factor)
    return {
        "Socavacion_general_m": scour * max(params.contraction_coeff, 0.1) * max(params.scour_safety_factor, 0.0),
        "Socavacion_local_m": scour * 1.35 * max(params.scour_safety_factor, 0.0),
    }


def quality_audit(sec: pd.Series, pts: pd.DataFrame, props: dict[str, float], wse: float, q: float, n: float, slope: float, fr: float, bracketed: bool) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    if len(pts) < 5:
        errors.append("Sección con menos de 5 puntos; no modelar sin completar geometría.")
    if pts["distancia_transversal_m"].duplicated().any():
        warnings.append("Distancias transversales duplicadas; revisar puntos repetidos.")
    if not pts["distancia_transversal_m"].is_monotonic_increasing:
        warnings.append("Distancia transversal no creciente; se ordenó para cálculo.")
    if pts["z_m"].isna().any():
        errors.append("Cotas nulas en puntos de sección.")
    width = float(pts["distancia_transversal_m"].max() - pts["distancia_transversal_m"].min()) if len(pts) else 0.0
    if width <= 0:
        errors.append("Ancho geométrico no positivo.")
    if props.get("A_m2", 0.0) <= 0:
        errors.append("Área mojada nula para el caudal solicitado.")
    if slope <= 0:
        errors.append("Pendiente hidráulica no positiva.")
    if slope > 0.10:
        warnings.append("Pendiente >10%; la hipótesis 1D gradualmente variada puede ser débil.")
    if n < 0.012 or n > 0.20:
        warnings.append("Manning fuera de rango usual; exige justificación o calibración.")
    if fr > 0.85 and fr < 1.15:
        warnings.append("Régimen cercano a crítico; alta sensibilidad numérica.")
    elif fr >= 1.15:
        warnings.append("Régimen supercrítico; revisar condición de borde y posible flujo rápidamente variado.")
    if not bracketed:
        warnings.append("El caudal supera la capacidad dentro de la geometría base; la lámina se extrapoló sobre las riberas.")
    if len(pts) > 2:
        dxmax = float(pts["distancia_transversal_m"].diff().abs().max())
        if dxmax > max(2.0, width / 4.0):
            warnings.append("Resolución transversal gruesa; hay separación grande entre puntos de sección.")
    if q <= 0:
        errors.append("Caudal no positivo.")
    return errors, warnings


def compute_irregular_hydraulics(
    df_secciones: pd.DataFrame,
    df_puntos: pd.DataFrame,
    params: AdvancedModelParams | dict | None = None,
    return_period_years: float = 100.0,
) -> pd.DataFrame:
    """Calcula hidráulica y sedimentos por secciones irregulares.

    Requiere df_secciones compatible con 03_Secciones y df_puntos compatible con
    04_Puntos_Seccion. Devuelve una tabla compatible con el visor 3D.
    """
    if params is None:
        p = AdvancedModelParams()
    elif isinstance(params, dict):
        clean = {k: v for k, v in params.items() if k in AdvancedModelParams.__dataclass_fields__}
        p = AdvancedModelParams(**clean)
    else:
        p = params
    p.rho_w = water_density_kgm3(p.temp_c) if not p.rho_w or p.rho_w <= 0 else p.rho_w

    sec = _normalize_sections(df_secciones)
    pts = _normalize_points(df_puntos)
    if sec.empty or pts.empty:
        raise ValueError("Se requieren 03_Secciones y 04_Puntos_Seccion para modelación irregular.")
    if "id_seccion" not in sec.columns or "id_seccion" not in pts.columns:
        raise ValueError("No se reconoce ID de sección en las tablas de geometría.")

    rows: list[dict[str, Any]] = []
    for _, srow in sec.sort_values("pk_m" if "pk_m" in sec.columns else "id_seccion").iterrows():
        sid = str(srow.get("id_seccion"))
        g = pts.loc[pts["id_seccion"].astype(str) == sid].copy()
        if g.empty and "nombre_seccion" in sec.columns:
            g = pts.loc[pts["id_seccion"].astype(str) == str(srow.get("nombre_seccion"))].copy()
        if g.empty:
            continue
        g = g.dropna(subset=["distancia_transversal_m", "z_m"]).sort_values("distancia_transversal_m")
        q = max(_sf(srow.get("q_m3s"), 50.0), EPS)
        slope = max(_sf(srow.get("slope"), 0.001), EPS)
        n = max(_sf(srow.get("manning_n"), 0.035), 0.005)
        d50 = _sf(srow.get("d50_mm"), 32.0)
        d84 = _sf(srow.get("d84_mm"), d50 * 2.0)
        d90 = _sf(srow.get("d90_mm"), d84 * 1.35)
        wse, props, bracketed = solve_normal_wse(g, q, n, slope)
        wc, props_c = solve_critical_wse(g, q)
        A = max(props["A_m2"], EPS)
        V = q / A
        Fr = V / math.sqrt(G * max(props["Dh_m"], EPS))
        E = props["y_max_m"] + p.alpha_velocity * V ** 2 / (2.0 * G)
        sed = _sediment_predictors(q, props, slope, n, d50, d90, p)
        scour = _simple_scour(props["y_max_m"], q, props, d50, return_period_years, p)
        cota_fondo = float(g["z_m"].min())
        errors, warnings = quality_audit(srow, g, props, wse, q, n, slope, Fr, bracketed)
        n_est = roughness_estimators(d50, d84, d90)
        rows.append({
            "ID": sid,
            "ID_Seccion": sid,
            "Nombre_Seccion": str(srow.get("nombre_seccion", sid)),
            "Distancia_m": _sf(srow.get("pk_m"), _sf(g["pk_m"].median() if "pk_m" in g.columns else 0.0)),
            "PK_m": _sf(srow.get("pk_m"), _sf(g["pk_m"].median() if "pk_m" in g.columns else 0.0)),
            "Q_m3s": q,
            "S_m_m": slope,
            "n": n,
            "n_grano_recomendado": n_est["n_recomendado_grano_mediana"],
            "Cota_fondo_m": cota_fondo,
            "Cota_lamina_agua_m": wse,
            "Cota_critica_m": wc,
            "y_m": props["y_max_m"],
            "y_critico_m": props_c["y_max_m"],
            "A_m2": props["A_m2"],
            "P_m": props["P_m"],
            "Be_m": props["T_m"],
            "R_m": props["R_m"],
            "Dh_m": props["Dh_m"],
            "V_Q_m_s": V,
            "Fr": Fr,
            "Regimen": "supercrítico" if Fr > 1.0 else "subcrítico" if Fr < 0.9 else "cercano a crítico",
            "Energia_especifica_m": E,
            "Q_Manning_m3s": q,
            "D50_mm": d50,
            "D84_mm": d84,
            "D90_mm": d90,
            "theta_D50": sed["theta"],
            "theta_eff": sed["theta_eff"],
            "theta_c": sed["theta_c"],
            "Indice_movilidad": sed["Indice_movilidad"],
            "Condicion": "Móvil" if sed["Indice_movilidad"] > 1.0 else "Estable/incidente",
            "tau_Pa": sed["tau_Pa"],
            "tau_c_Pa": sed["tau_c_Pa"],
            "tau_kg_m2": sed["tau_Pa"] / G,
            "qb_m2s": sed["qb_MPM_m2s"],
            "Gs_m3s": sed["Gs_MPM_m3s"],
            "Gs_m3h": sed["Gs_MPM_m3s"] * 3600.0,
            "Gs_tonh": sed["Gs_MPM_tonh"],
            "Gs_MPM_m3s": sed["Gs_MPM_m3s"],
            "Gs_EngelundHansen_m3s": sed["Gs_EngelundHansen_m3s"],
            "Gs_EngelundHansen_tonh": sed["Gs_EngelundHansen_tonh"],
            "Socavacion_base_m": scour["Socavacion_general_m"],
            "Socavacion_ajustada_m": scour["Socavacion_general_m"],
            "Socavacion_local_m": scour["Socavacion_local_m"],
            "Cota_fondo_socavado_m": cota_fondo - scour["Socavacion_general_m"],
            "Errores_QA": "; ".join(errors),
            "Advertencias_QA": "; ".join(warnings),
            "Estado_QA": "CRITICO" if errors else "REVISAR" if warnings else "OK",
            "Metodo_hidraulico": "Sección irregular + Manning normal depth + calado crítico",
        })
    if not rows:
        raise ValueError("No se pudo calcular ninguna sección irregular.")
    out = pd.DataFrame(rows).sort_values("Distancia_m")
    return out


def sensitivity_manning_irregular(
    df_secciones: pd.DataFrame,
    df_puntos: pd.DataFrame,
    params: AdvancedModelParams | dict | None = None,
    n_factors: tuple[float, ...] = (0.8, 1.0, 1.2),
    return_period_years: float = 100.0,
) -> pd.DataFrame:
    sec0 = _normalize_sections(df_secciones)
    rows = []
    for factor in n_factors:
        sec = sec0.copy()
        if "manning_n" in sec.columns:
            sec["manning_n"] = sec["manning_n"].astype(float) * factor
        res = compute_irregular_hydraulics(sec, df_puntos, params=params, return_period_years=return_period_years)
        rows.append(pd.DataFrame({
            "ID": res["ID"],
            "factor_n": factor,
            "n": res["n"],
            "Cota_lamina_agua_m": res["Cota_lamina_agua_m"],
            "y_m": res["y_m"],
            "V_Q_m_s": res["V_Q_m_s"],
            "Fr": res["Fr"],
            "Socavacion_ajustada_m": res["Socavacion_ajustada_m"],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_audit_summary(results: pd.DataFrame, sensitivity: pd.DataFrame | None = None) -> pd.DataFrame:
    df = pd.DataFrame(results)
    rows = []
    if df.empty:
        return pd.DataFrame()
    rows.append({"criterio": "Secciones calculadas", "valor": int(len(df)), "estado": "OK"})
    ncrit = int((df.get("Estado_QA", pd.Series(dtype=str)).astype(str) == "CRITICO").sum())
    nrev = int((df.get("Estado_QA", pd.Series(dtype=str)).astype(str) == "REVISAR").sum())
    rows.append({"criterio": "Errores críticos QA", "valor": ncrit, "estado": "OK" if ncrit == 0 else "CRITICO"})
    rows.append({"criterio": "Advertencias QA", "valor": nrev, "estado": "OK" if nrev == 0 else "REVISAR"})
    rows.append({"criterio": "Fr máximo", "valor": round(float(df["Fr"].max()), 3), "estado": "REVISAR" if float(df["Fr"].max()) > 0.85 else "OK"})
    rows.append({"criterio": "Índice movilidad máximo", "valor": round(float(df["Indice_movilidad"].max()), 3), "estado": "REVISAR" if float(df["Indice_movilidad"].max()) > 1.0 else "OK"})
    rows.append({"criterio": "Socavación máxima [m]", "valor": round(float(df["Socavacion_ajustada_m"].max()), 3), "estado": "REVISAR"})
    if sensitivity is not None and not pd.DataFrame(sensitivity).empty:
        s = pd.DataFrame(sensitivity)
        spread = s.groupby("ID")["Cota_lamina_agua_m"].agg(lambda x: float(max(x) - min(x))).max()
        rows.append({"criterio": "Sensibilidad máx. a Manning ±20% [m]", "valor": round(float(spread), 3), "estado": "REVISAR" if spread > 0.3 else "OK"})
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# v6.0 — Perfil conectado, calibración, incertidumbre y puntaje de confianza
# -----------------------------------------------------------------------------

@dataclass
class ConnectedProfileParams(AdvancedModelParams):
    """Parámetros del motor conectado v6.

    El perfil se calcula con el método paso a paso estándar usando balance de
    energía entre secciones consecutivas. La condición de borde por defecto es
    tirante normal en la sección aguas abajo.
    """
    downstream_boundary: str = "normal_depth"  # normal_depth | known_wse
    downstream_wse_m: float | None = None
    max_step_iterations: int = 80
    energy_tolerance_m: float = 1e-4
    min_positive_slope: float = 1e-6
    max_reasonable_dx_m: float = 500.0
    compute_independent_normal: bool = True


def _velocity_head(q: float, props: dict[str, float], alpha: float = 1.0) -> float:
    A = max(props.get("A_m2", 0.0), EPS)
    V = q / A
    return alpha * V * V / (2.0 * G)


def _friction_slope(q: float, props: dict[str, float], n: float) -> float:
    A = max(props.get("A_m2", 0.0), EPS)
    R = max(props.get("R_m", 0.0), EPS)
    return (q * max(n, EPS) / max(A * (R ** (2.0 / 3.0)), EPS)) ** 2


def _energy_at(points: pd.DataFrame, wse: float, q: float, alpha: float) -> tuple[float, dict[str, float]]:
    props = section_properties_from_wse(points, wse)
    return wse + _velocity_head(q, props, alpha), props


def _solve_standard_step_upstream(
    pts_up: pd.DataFrame,
    pts_down: pd.DataFrame,
    wse_down: float,
    q: float,
    n_up: float,
    n_down: float,
    dx: float,
    alpha: float,
    contraction_coeff: float,
    expansion_coeff: float,
    tol: float = 1e-4,
    max_iter: int = 80,
) -> tuple[float, dict[str, float], dict[str, float], bool, str]:
    """Resuelve cota de agua aguas arriba por balance de energía.

    E_up = E_down + hf + he
    hf = (Sf_up + Sf_down)/2 * dx
    he = C * |hv_up - hv_down|; usa C contracción/expansión según cambio de
    carga de velocidad.
    """
    gd = pd.DataFrame(pts_down).dropna(subset=["distancia_transversal_m", "z_m"]).sort_values("distancia_transversal_m")
    gu = pd.DataFrame(pts_up).dropna(subset=["distancia_transversal_m", "z_m"]).sort_values("distancia_transversal_m")
    _, props_d = _energy_at(gd, wse_down, q, alpha)
    hv_d = _velocity_head(q, props_d, alpha)
    E_d = wse_down + hv_d
    sf_d = _friction_slope(q, props_d, n_down)

    zmin_u = float(gu["z_m"].min())
    zmax_u = float(gu["z_m"].max())
    # Cota normal local sirve de primera referencia y evita búsqueda excesiva.
    try:
        wse_norm, _, _ = solve_normal_wse(gu, q, n_up, max(sf_d, EPS))
    except Exception:
        wse_norm = zmin_u + 1.0
    lo = max(zmin_u + 1e-5, min(wse_norm, wse_down) - max(2.0, 0.15 * abs(dx)))
    hi = max(zmax_u + 0.5, max(wse_norm, wse_down) + max(3.0, 0.25 * abs(dx)))

    def residual(wse_u: float) -> tuple[float, dict[str, float]]:
        _, props_u = _energy_at(gu, wse_u, q, alpha)
        hv_u = _velocity_head(q, props_u, alpha)
        sf_u = _friction_slope(q, props_u, n_up)
        coeff = contraction_coeff if hv_u > hv_d else expansion_coeff
        he = max(coeff, 0.0) * abs(hv_u - hv_d)
        hf = 0.5 * (sf_u + sf_d) * max(abs(dx), EPS)
        E_u = wse_u + hv_u
        return E_u - (E_d + hf + he), props_u

    f_lo, _ = residual(lo)
    f_hi, _ = residual(hi)
    expand_count = 0
    while f_lo * f_hi > 0 and expand_count < 40:
        # Si no encierra, expandir hacia arriba; en ríos subcríticos suele bastar.
        hi += max(1.0, 0.10 * max(abs(dx), 1.0))
        f_hi, _ = residual(hi)
        expand_count += 1
    if f_lo * f_hi > 0:
        # Último recurso: buscar mínimo del residuo absoluto en grilla fina.
        grid = np.linspace(lo, hi, 260)
        vals = []
        props_list = []
        for z in grid:
            f, pr = residual(float(z))
            vals.append(abs(f))
            props_list.append(pr)
        idx = int(np.nanargmin(vals))
        ok = bool(vals[idx] < max(0.05, 10 * tol))
        return float(grid[idx]), props_list[idx], props_d, ok, "min_abs_residual"

    props_mid = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid, props_mid = residual(mid)
        if abs(f_mid) <= tol:
            return float(mid), props_mid, props_d, True, "bisection"
        if f_lo * f_mid <= 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    mid = 0.5 * (lo + hi)
    _, props_mid = residual(mid)
    return float(mid), props_mid, props_d, False, "max_iter"


def compute_connected_profile_v6(
    df_secciones: pd.DataFrame,
    df_puntos: pd.DataFrame,
    params: ConnectedProfileParams | AdvancedModelParams | dict | None = None,
    return_period_years: float = 100.0,
) -> pd.DataFrame:
    """Calcula perfil 1D conectado con balance de energía entre secciones.

    Este motor no calcula cada sección aisladamente: parte desde la condición de
    borde aguas abajo y avanza aguas arriba resolviendo la ecuación de energía.
    Es más consistente para perfiles gradualmente variados y permite auditar
    pérdidas por fricción, contracción/expansión, control crítico y geometría.
    """
    if params is None:
        p = ConnectedProfileParams()
    elif isinstance(params, dict):
        base = {k: v for k, v in params.items() if k in ConnectedProfileParams.__dataclass_fields__}
        p = ConnectedProfileParams(**base)
    elif isinstance(params, ConnectedProfileParams):
        p = params
    else:
        # Promover AdvancedModelParams a ConnectedProfileParams conservando campos comunes.
        base = {k: getattr(params, k) for k in AdvancedModelParams.__dataclass_fields__ if hasattr(params, k)}
        p = ConnectedProfileParams(**base)
    p.rho_w = water_density_kgm3(p.temp_c) if not p.rho_w or p.rho_w <= 0 else p.rho_w

    sec = _normalize_sections(df_secciones)
    pts = _normalize_points(df_puntos)
    if sec.empty or pts.empty:
        raise ValueError("Se requieren 03_Secciones y 04_Puntos_Seccion.")
    if "pk_m" not in sec.columns:
        sec["pk_m"] = 0.0
    sec = sec.sort_values("pk_m").reset_index(drop=True)

    # Cálculo independiente para referencia, control crítico y respaldo local.
    indep = compute_irregular_hydraulics(sec, pts, params=p, return_period_years=return_period_years) if p.compute_independent_normal else pd.DataFrame()
    indep_by_id = {str(r["ID"]): r for _, r in indep.iterrows()} if not indep.empty else {}

    # Avance subcrítico: condición de borde aguas abajo y marcha hacia aguas arriba.
    computed: dict[str, dict[str, Any]] = {}
    order_down_to_up = list(sec.index[::-1])
    prev_sid = None
    prev_wse = None
    prev_props = None
    prev_n = None
    prev_pk = None
    for idx in order_down_to_up:
        srow = sec.loc[idx]
        sid = str(srow.get("id_seccion"))
        g = pts.loc[pts["id_seccion"].astype(str) == sid].copy()
        if g.empty and "nombre_seccion" in sec.columns:
            g = pts.loc[pts["id_seccion"].astype(str) == str(srow.get("nombre_seccion"))].copy()
        g = g.dropna(subset=["distancia_transversal_m", "z_m"]).sort_values("distancia_transversal_m")
        if g.empty:
            continue
        q = max(_sf(srow.get("q_m3s"), 50.0), EPS)
        slope = max(_sf(srow.get("slope"), 0.001), p.min_positive_slope)
        n = max(_sf(srow.get("manning_n"), 0.035), 0.005)
        pk = _sf(srow.get("pk_m"), _sf(g["pk_m"].median() if "pk_m" in g.columns else 0.0))
        d50 = _sf(srow.get("d50_mm"), 32.0)
        d84 = _sf(srow.get("d84_mm"), d50 * 2.0)
        d90 = _sf(srow.get("d90_mm"), d84 * 1.35)
        method = ""
        converged = True
        if prev_sid is None:
            if p.downstream_boundary == "known_wse" and p.downstream_wse_m is not None:
                wse = float(p.downstream_wse_m)
                props = section_properties_from_wse(g, wse)
                if props["A_m2"] <= 0:
                    raise ValueError("La condición aguas abajo conocida queda bajo el fondo de la sección.")
                method = "known_downstream_wse"
            else:
                wse, props, bracketed = solve_normal_wse(g, q, n, slope)
                converged = bool(bracketed)
                method = "downstream_normal_depth"
        else:
            # El tramo entre sección actual (aguas arriba) y la sección previa (aguas abajo).
            g_down = pts.loc[pts["id_seccion"].astype(str) == str(prev_sid)].copy()
            dx = abs(float(prev_pk) - float(pk)) if prev_pk is not None else max(1.0, abs(pk))
            wse, props, props_d, converged, method = _solve_standard_step_upstream(
                g,
                g_down,
                float(prev_wse),
                q,
                n,
                float(prev_n),
                dx,
                float(p.alpha_velocity),
                float(p.contraction_coeff),
                float(p.expansion_coeff),
                tol=float(p.energy_tolerance_m),
                max_iter=int(p.max_step_iterations),
            )
        wc, props_c = solve_critical_wse(g, q)
        A = max(props["A_m2"], EPS)
        V = q / A
        Fr = V / math.sqrt(G * max(props["Dh_m"], EPS))
        E = props["y_max_m"] + p.alpha_velocity * V ** 2 / (2.0 * G)
        sed = _sediment_predictors(q, props, slope, n, d50, d90, p)
        scour = _simple_scour(props["y_max_m"], q, props, d50, return_period_years, p)
        cota_fondo = float(g["z_m"].min())
        bracketed = converged
        errors, warnings = quality_audit(srow, g, props, wse, q, n, slope, Fr, bracketed)
        if not converged:
            warnings.append(f"Perfil conectado no convergió plenamente en sección {sid}; revisar condición de borde/espaciamiento.")
        if prev_pk is not None:
            dx = abs(float(prev_pk) - float(pk))
            if dx > p.max_reasonable_dx_m:
                warnings.append(f"Espaciamiento longitudinal alto ({dx:.1f} m); agregar/interpolar sección real.")
        # Diferencia con normal local: alerta de posible control aguas abajo o BC sensible.
        normal_ref = indep_by_id.get(sid)
        has_normal_ref = normal_ref is not None and hasattr(normal_ref, "get")
        dy_norm = float(wse - _sf(normal_ref.get("Cota_lamina_agua_m"), wse)) if has_normal_ref else 0.0
        if abs(dy_norm) > max(0.50, 0.25 * max(props["y_max_m"], 0.01)):
            warnings.append("Diferencia importante entre perfil conectado y tirante normal local; revisar control de borde o pendiente.")
        n_est = roughness_estimators(d50, d84, d90)
        computed[sid] = {
            "ID": sid,
            "ID_Seccion": sid,
            "Nombre_Seccion": str(srow.get("nombre_seccion", sid)),
            "Distancia_m": pk,
            "PK_m": pk,
            "Q_m3s": q,
            "S_m_m": slope,
            "n": n,
            "n_grano_recomendado": n_est["n_recomendado_grano_mediana"],
            "Cota_fondo_m": cota_fondo,
            "Cota_lamina_agua_m": wse,
            "Cota_lamina_normal_local_m": _sf(normal_ref.get("Cota_lamina_agua_m"), np.nan) if has_normal_ref else np.nan,
            "Delta_conectado_vs_normal_m": dy_norm,
            "Cota_critica_m": wc,
            "y_m": props["y_max_m"],
            "y_critico_m": props_c["y_max_m"],
            "A_m2": props["A_m2"],
            "P_m": props["P_m"],
            "Be_m": props["T_m"],
            "R_m": props["R_m"],
            "Dh_m": props["Dh_m"],
            "V_Q_m_s": V,
            "Fr": Fr,
            "Regimen": "supercrítico" if Fr > 1.0 else "subcrítico" if Fr < 0.9 else "cercano a crítico",
            "Energia_especifica_m": E,
            "Q_Manning_m3s": manning_capacity(g, wse, n, slope),
            "D50_mm": d50,
            "D84_mm": d84,
            "D90_mm": d90,
            "theta_D50": sed["theta"],
            "theta_eff": sed["theta_eff"],
            "theta_c": sed["theta_c"],
            "Indice_movilidad": sed["Indice_movilidad"],
            "Condicion": "Móvil" if sed["Indice_movilidad"] > 1.0 else "Estable/incidente",
            "tau_Pa": sed["tau_Pa"],
            "tau_c_Pa": sed["tau_c_Pa"],
            "tau_kg_m2": sed["tau_Pa"] / G,
            "qb_m2s": sed["qb_MPM_m2s"],
            "Gs_m3s": sed["Gs_MPM_m3s"],
            "Gs_m3h": sed["Gs_MPM_m3s"] * 3600.0,
            "Gs_tonh": sed["Gs_MPM_tonh"],
            "Gs_MPM_m3s": sed["Gs_MPM_m3s"],
            "Gs_EngelundHansen_m3s": sed["Gs_EngelundHansen_m3s"],
            "Gs_EngelundHansen_tonh": sed["Gs_EngelundHansen_tonh"],
            "Socavacion_base_m": scour["Socavacion_general_m"],
            "Socavacion_ajustada_m": scour["Socavacion_general_m"],
            "Socavacion_local_m": scour["Socavacion_local_m"],
            "Cota_fondo_socavado_m": cota_fondo - scour["Socavacion_general_m"],
            "Metodo_perfil": method,
            "Perfil_conectado_converge": bool(converged),
            "Errores_QA": "; ".join(errors),
            "Advertencias_QA": "; ".join(warnings),
            "Estado_QA": "CRITICO" if errors else "REVISAR" if warnings else "OK",
            "Metodo_hidraulico": "Perfil conectado v6: Standard Step + sección irregular + sedimentos",
        }
        prev_sid = sid
        prev_wse = wse
        prev_props = props
        prev_n = n
        prev_pk = pk
    if not computed:
        raise ValueError("No se pudo calcular ninguna sección conectada.")
    out = pd.DataFrame(list(computed.values())).sort_values("Distancia_m").reset_index(drop=True)
    return out


def calibrate_manning_multiplier(
    df_secciones: pd.DataFrame,
    df_puntos: pd.DataFrame,
    observed_df: pd.DataFrame,
    params: ConnectedProfileParams | dict | None = None,
    return_period_years: float = 100.0,
    multipliers: Iterable[float] = tuple(np.round(np.linspace(0.65, 1.45, 33), 3)),
) -> tuple[float, pd.DataFrame]:
    """Calibra un multiplicador global de Manning usando cotas observadas.

    observed_df debe contener ID_Seccion o PK_m y Cota_lamina_observada_m.
    Devuelve el mejor factor y tabla de evaluación por factor.
    """
    obs = pd.DataFrame(observed_df).copy()
    if obs.empty:
        raise ValueError("No hay datos observados para calibración.")
    col_wse = _first_existing(obs, ["Cota_lamina_observada_m", "wse_obs_m", "WSE_obs", "cota_observada_m"])
    if col_wse is None:
        raise ValueError("La tabla observada debe incluir Cota_lamina_observada_m.")
    col_id = _first_existing(obs, ["ID_Seccion", "id_seccion", "ID"])
    col_pk = _first_existing(obs, ["PK_m", "pk_m", "Distancia_m"])
    obs[col_wse] = pd.to_numeric(obs[col_wse], errors="coerce")
    sec0 = _normalize_sections(df_secciones)
    rows = []
    best_factor = 1.0
    best_rmse = float("inf")
    for factor in multipliers:
        sec = sec0.copy()
        if "manning_n" in sec.columns:
            sec["manning_n"] = sec["manning_n"].astype(float) * float(factor)
        res = compute_connected_profile_v6(sec, df_puntos, params=params, return_period_years=return_period_years)
        pairs = []
        for _, o in obs.dropna(subset=[col_wse]).iterrows():
            if col_id is not None and pd.notna(o.get(col_id)):
                rr = res.loc[res["ID_Seccion"].astype(str) == str(o.get(col_id))]
            elif col_pk is not None and pd.notna(o.get(col_pk)):
                pk = _sf(o.get(col_pk), np.nan)
                rr = res.iloc[[int((res["PK_m"] - pk).abs().idxmin())]] if np.isfinite(pk) else pd.DataFrame()
            else:
                rr = pd.DataFrame()
            if not rr.empty:
                pairs.append(float(rr.iloc[0]["Cota_lamina_agua_m"]) - float(o[col_wse]))
        if pairs:
            rmse = float(math.sqrt(np.mean(np.square(pairs))))
            mae = float(np.mean(np.abs(pairs)))
            bias = float(np.mean(pairs))
        else:
            rmse = mae = bias = float("nan")
        rows.append({"factor_n": float(factor), "RMSE_m": rmse, "MAE_m": mae, "Sesgo_m": bias, "n_observaciones": len(pairs)})
        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_factor = float(factor)
    return best_factor, pd.DataFrame(rows)


def monte_carlo_uncertainty_v6(
    df_secciones: pd.DataFrame,
    df_puntos: pd.DataFrame,
    params: ConnectedProfileParams | dict | None = None,
    return_period_years: float = 100.0,
    n_runs: int = 80,
    q_cv: float = 0.10,
    n_cv: float = 0.15,
    d50_cv: float = 0.25,
    slope_cv: float = 0.15,
    seed: int = 20260620,
) -> pd.DataFrame:
    """Propagación de incertidumbre simple para Q, n, D50 y pendiente.

    Entrega P10/P50/P90 de cota de agua, velocidad, transporte y socavación.
    """
    rng = np.random.default_rng(seed)
    sec0 = _normalize_sections(df_secciones)
    if sec0.empty:
        return pd.DataFrame()
    q_cv = max(float(q_cv), 0.0)
    n_cv = max(float(n_cv), 0.0)
    d50_cv = max(float(d50_cv), 0.0)
    slope_cv = max(float(slope_cv), 0.0)
    samples = []
    runs = max(5, int(n_runs))
    for run in range(runs):
        sec = sec0.copy()
        # Perturbaciones lognormales con media aproximadamente 1.
        for col, cv in [("q_m3s", q_cv), ("manning_n", n_cv), ("d50_mm", d50_cv), ("d84_mm", d50_cv), ("d90_mm", d50_cv), ("slope", slope_cv)]:
            if col in sec.columns and cv > 0:
                sigma = math.sqrt(math.log(1.0 + cv * cv))
                mu = -0.5 * sigma * sigma
                factors = rng.lognormal(mean=mu, sigma=sigma, size=len(sec))
                sec[col] = pd.to_numeric(sec[col], errors="coerce") * factors
        try:
            res = compute_connected_profile_v6(sec, df_puntos, params=params, return_period_years=return_period_years)
            keep = res[["ID_Seccion", "PK_m", "Cota_lamina_agua_m", "V_Q_m_s", "Fr", "Gs_MPM_m3s", "Gs_EngelundHansen_m3s", "Socavacion_ajustada_m"]].copy()
            keep["run"] = run
            samples.append(keep)
        except Exception:
            continue
    if not samples:
        return pd.DataFrame()
    all_s = pd.concat(samples, ignore_index=True)
    out_rows = []
    for sid, g in all_s.groupby("ID_Seccion"):
        row = {"ID_Seccion": sid, "n_corridas_validas": int(g["run"].nunique()), "PK_m": float(g["PK_m"].median())}
        for col in ["Cota_lamina_agua_m", "V_Q_m_s", "Fr", "Gs_MPM_m3s", "Gs_EngelundHansen_m3s", "Socavacion_ajustada_m"]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            if vals.empty:
                continue
            row[f"{col}_P10"] = float(np.percentile(vals, 10))
            row[f"{col}_P50"] = float(np.percentile(vals, 50))
            row[f"{col}_P90"] = float(np.percentile(vals, 90))
            row[f"{col}_amplitud_P90_P10"] = row[f"{col}_P90"] - row[f"{col}_P10"]
        out_rows.append(row)
    return pd.DataFrame(out_rows).sort_values("PK_m").reset_index(drop=True)


def confidence_score_v6(
    results: pd.DataFrame,
    sensitivity: pd.DataFrame | None = None,
    uncertainty: pd.DataFrame | None = None,
    calibration: pd.DataFrame | None = None,
) -> tuple[float, pd.DataFrame]:
    """Calcula puntaje 1-10 de confianza técnica del modelo.

    El 9/10 se alcanza solo si no hay errores críticos, el perfil conectado
    converge, la sensibilidad/uncertidumbre es acotada y existe calibración o
    contraste hidráulico fuerte. Sin observaciones, el techo recomendado es 8.6.
    """
    df = pd.DataFrame(results)
    score = 10.0
    rows = []
    if df.empty:
        return 1.0, pd.DataFrame([{"criterio": "Sin resultados", "penalizacion": 9.0, "estado": "CRITICO"}])

    n = len(df)
    ncrit = int((df.get("Estado_QA", pd.Series(dtype=str)).astype(str) == "CRITICO").sum())
    nrev = int((df.get("Estado_QA", pd.Series(dtype=str)).astype(str) == "REVISAR").sum())
    if ncrit:
        pen = min(4.0, 1.3 * ncrit)
        score -= pen
        rows.append({"criterio": "Errores críticos QA", "valor": ncrit, "penalizacion": pen, "estado": "CRITICO"})
    if nrev:
        pen = min(1.6, 0.18 * nrev)
        score -= pen
        rows.append({"criterio": "Advertencias QA", "valor": nrev, "penalizacion": pen, "estado": "REVISAR"})

    if "Perfil_conectado_converge" in df.columns:
        nfail = int((~df["Perfil_conectado_converge"].astype(bool)).sum())
        if nfail:
            pen = min(2.5, 0.7 * nfail)
            score -= pen
            rows.append({"criterio": "Convergencia perfil conectado", "valor": f"{nfail}/{n}", "penalizacion": pen, "estado": "REVISAR"})
    else:
        score -= 1.2
        rows.append({"criterio": "Perfil no conectado", "valor": "no disponible", "penalizacion": 1.2, "estado": "REVISAR"})

    fr_max = float(pd.to_numeric(df.get("Fr", pd.Series([0])), errors="coerce").max())
    if fr_max > 1.1:
        score -= 1.0
        rows.append({"criterio": "Régimen supercrítico/mixed", "valor": round(fr_max, 3), "penalizacion": 1.0, "estado": "REVISAR"})
    elif fr_max > 0.85:
        score -= 0.35
        rows.append({"criterio": "Régimen cercano a crítico", "valor": round(fr_max, 3), "penalizacion": 0.35, "estado": "REVISAR"})

    if sensitivity is not None and not pd.DataFrame(sensitivity).empty:
        s = pd.DataFrame(sensitivity)
        try:
            spread = float(s.groupby("ID")["Cota_lamina_agua_m"].agg(lambda x: max(x) - min(x)).max())
            if spread > 0.75:
                pen = 0.9
            elif spread > 0.30:
                pen = 0.45
            else:
                pen = 0.0
            if pen:
                score -= pen
            rows.append({"criterio": "Sensibilidad Manning ±20% [m]", "valor": round(spread, 3), "penalizacion": pen, "estado": "OK" if pen == 0 else "REVISAR"})
        except Exception:
            pass
    else:
        score -= 0.4
        rows.append({"criterio": "Sensibilidad Manning", "valor": "no ejecutada", "penalizacion": 0.4, "estado": "REVISAR"})

    if uncertainty is not None and not pd.DataFrame(uncertainty).empty:
        u = pd.DataFrame(uncertainty)
        col = "Cota_lamina_agua_m_amplitud_P90_P10"
        if col in u.columns:
            amp = float(pd.to_numeric(u[col], errors="coerce").max())
            if amp > 1.0:
                pen = 0.9
            elif amp > 0.5:
                pen = 0.45
            else:
                pen = 0.0
            score -= pen
            rows.append({"criterio": "Incertidumbre P90-P10 lámina [m]", "valor": round(amp, 3), "penalizacion": pen, "estado": "OK" if pen == 0 else "REVISAR"})
    else:
        score -= 0.4
        rows.append({"criterio": "Monte Carlo incertidumbre", "valor": "no ejecutado", "penalizacion": 0.4, "estado": "REVISAR"})

    calib_ok = False
    if calibration is not None and not pd.DataFrame(calibration).empty:
        c = pd.DataFrame(calibration)
        if "RMSE_m" in c.columns:
            rmse = float(pd.to_numeric(c["RMSE_m"], errors="coerce").min())
            calib_ok = np.isfinite(rmse)
            if calib_ok:
                if rmse > 0.75:
                    pen = 0.8
                elif rmse > 0.35:
                    pen = 0.35
                else:
                    pen = 0.0
                score -= pen
                rows.append({"criterio": "Calibración/contraste WSE RMSE [m]", "valor": round(rmse, 3), "penalizacion": pen, "estado": "OK" if pen == 0 else "REVISAR"})
    if not calib_ok:
        # Sin datos observados, por ética técnica no se certifica 9/10 aunque el motor sea robusto.
        score = min(score, 8.6)
        rows.append({"criterio": "Calibración observada", "valor": "no disponible", "penalizacion": "techo 8.6", "estado": "REVISAR"})

    score = max(1.0, min(10.0, float(score)))
    rows.insert(0, {"criterio": "Nivel de confianza global", "valor": round(score, 1), "penalizacion": 0, "estado": "OK" if score >= 9 else "REVISAR"})
    return round(score, 1), pd.DataFrame(rows)
