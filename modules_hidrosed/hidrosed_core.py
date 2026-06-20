"""Núcleo HidroSed Cauces: hidráulica fluvial, transporte, socavación y lecho móvil.

Funciones sin Streamlit para mantener trazabilidad y permitir pruebas automáticas.
La formulación reproduce la lógica descrita en el Manual Maestro HidroSed Maestra:
- Sección trapecial/manual o sección simplificada.
- Manning, Froude, tensión de fondo.
- Shields y Meyer-Peter-Müller para transporte potencial.
- Lischtvan-Levediev simplificado para socavación generalizada.
- Indicador conceptual de lecho móvil tipo Exner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

G = 9.81


@dataclass
class GlobalParams:
    project_name: str = "Proyecto HidroSed Cauces"
    river_name: str = "Cauce"
    location_name: str = "Región de Coquimbo"
    condition: str = "Sin proyecto"
    return_period_years: float = 100.0
    rho_w: float = 1000.0
    rho_s: float = 2650.0
    porosity: float = 0.35
    theta_c: float = 0.047
    sediment_supply_factor: float = 1.0
    scour_safety_factor: float = 1.0
    mu_ll: float = 1.0
    gamma_mix: float = 1.0
    use_roughness_correction: bool = True


@dataclass
class SectionInput:
    section_id: str
    distance_m: float
    dx_m: float
    q_m3s: float
    slope: float
    manning_n: float
    bottom_width_m: float
    depth_m: float
    side_slope_z: float
    bed_elevation_m: float
    is_curve: bool = False
    curve_side: str = "Eje"
    curve_factor: float = 1.0
    d50_mm: float = 32.0
    d84_mm: float = 64.0
    d90_mm: float = 90.0
    dm_mm: float = 45.0


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace(",", ".").strip()
            if not value:
                return default
        result = float(value)
        if not math.isfinite(result):
            return default
        return result
    except Exception:
        return default


def default_sections(
    n_sections: int = 5,
    q_m3s: float = 50.0,
    slope: float = 0.01,
    manning_n: float = 0.040,
    dx_m: float = 100.0,
    bottom_width_m: float = 12.0,
    depth_m: float = 2.0,
    side_slope_z: float = 1.5,
    bed_elevation_m: float = 100.0,
    d50_mm: float = 32.0,
    d84_mm: float = 64.0,
    d90_mm: float = 90.0,
    dm_mm: float = 45.0,
) -> list[SectionInput]:
    n = max(1, int(n_sections))
    out: list[SectionInput] = []
    for i in range(n):
        out.append(
            SectionInput(
                section_id=f"S{i+1}",
                distance_m=i * dx_m,
                dx_m=dx_m,
                q_m3s=q_m3s,
                slope=slope,
                manning_n=manning_n,
                bottom_width_m=bottom_width_m,
                depth_m=depth_m,
                side_slope_z=side_slope_z,
                bed_elevation_m=bed_elevation_m - i * dx_m * slope,
                d50_mm=d50_mm,
                d84_mm=d84_mm,
                d90_mm=d90_mm,
                dm_mm=dm_mm,
            )
        )
    return out


def sections_to_dataframe(sections: Sequence[SectionInput]) -> pd.DataFrame:
    rows = [asdict(s) for s in sections]
    return pd.DataFrame(rows)


def dataframe_to_sections(df: pd.DataFrame) -> list[SectionInput]:
    sections: list[SectionInput] = []
    defaults = SectionInput(section_id="S1", distance_m=0, dx_m=100, q_m3s=50, slope=0.01, manning_n=0.04,
                            bottom_width_m=12, depth_m=2, side_slope_z=1.5, bed_elevation_m=100)
    default_dict = asdict(defaults)
    for idx, row in df.iterrows():
        data = default_dict.copy()
        for key in data.keys():
            if key in row:
                data[key] = row[key]
        data["section_id"] = str(data.get("section_id") or f"S{idx+1}")
        data["curve_side"] = str(data.get("curve_side") or "Eje")
        data["is_curve"] = bool(data.get("is_curve"))
        for key in [k for k in data.keys() if k not in {"section_id", "curve_side", "is_curve"}]:
            data[key] = safe_float(data[key], safe_float(default_dict[key], 0.0))
        sections.append(SectionInput(**data))
    return sections


def parse_grain_curve(text: str) -> pd.DataFrame:
    """Parsea curva granulométrica: diametro_mm; porcentaje_pasa."""
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # admite coma decimal, ;, tab o coma como separador de columnas
        line = line.replace("\t", ";")
        if ";" in line:
            parts = line.split(";")
        else:
            parts = line.split(",")
        if len(parts) < 2:
            continue
        d = safe_float(parts[0], None)
        p = safe_float(parts[1], None)
        if d is None or p is None or d <= 0:
            continue
        rows.append({"diametro_mm": d, "porcentaje_pasa": max(0.0, min(100.0, p))})
    if not rows:
        raise ValueError("No se pudo leer una curva granulométrica válida.")
    df = pd.DataFrame(rows).drop_duplicates(subset=["porcentaje_pasa"]).sort_values("porcentaje_pasa")
    return df


def interpolate_dp(curve_df: pd.DataFrame, p: float) -> float:
    df = curve_df.sort_values("porcentaje_pasa")
    perc = df["porcentaje_pasa"].to_numpy(dtype=float)
    diam = df["diametro_mm"].to_numpy(dtype=float)
    if p <= perc.min():
        return float(diam[0])
    if p >= perc.max():
        return float(diam[-1])
    idx = np.searchsorted(perc, p)
    p1, p2 = perc[idx - 1], perc[idx]
    d1, d2 = diam[idx - 1], diam[idx]
    t = (p - p1) / max(p2 - p1, 1e-9)
    return float(10 ** (math.log10(d1) + t * (math.log10(d2) - math.log10(d1))))


def grain_stats(curve_df: pd.DataFrame) -> dict[str, float]:
    d10 = interpolate_dp(curve_df, 10)
    d16 = interpolate_dp(curve_df, 16)
    d30 = interpolate_dp(curve_df, 30)
    d50 = interpolate_dp(curve_df, 50)
    d60 = interpolate_dp(curve_df, 60)
    d84 = interpolate_dp(curve_df, 84)
    d90 = interpolate_dp(curve_df, 90)
    # diámetro medio geométrico ponderado por incremento de % pasa
    df = curve_df.sort_values("porcentaje_pasa")
    diam = df["diametro_mm"].to_numpy(dtype=float)
    perc = df["porcentaje_pasa"].to_numpy(dtype=float)
    weights = np.diff(perc)
    if len(weights) > 0 and weights.sum() > 0:
        mids = np.sqrt(diam[1:] * diam[:-1])
        dm = float(np.sum(mids * weights) / np.sum(weights))
    else:
        dm = d50
    return {
        "D10_mm": d10,
        "D16_mm": d16,
        "D30_mm": d30,
        "D50_mm": d50,
        "D60_mm": d60,
        "D84_mm": d84,
        "D90_mm": d90,
        "Dm_mm": dm,
        "Cu": d60 / d10 if d10 > 0 else math.nan,
        "Cc": (d30 ** 2) / (d10 * d60) if d10 > 0 and d60 > 0 else math.nan,
    }


def apply_grain_stats_to_sections(sections: Sequence[SectionInput], stats: dict[str, float]) -> list[SectionInput]:
    out = []
    for s in sections:
        d = asdict(s)
        d["d50_mm"] = float(stats.get("D50_mm", s.d50_mm))
        d["d84_mm"] = float(stats.get("D84_mm", s.d84_mm))
        d["d90_mm"] = float(stats.get("D90_mm", s.d90_mm))
        d["dm_mm"] = float(stats.get("Dm_mm", s.dm_mm))
        out.append(SectionInput(**d))
    return out


def trapezoid_geometry(bottom_width_m: float, depth_m: float, side_slope_z: float) -> dict[str, float]:
    b = max(float(bottom_width_m), 1e-6)
    y = max(float(depth_m), 1e-6)
    z = max(float(side_slope_z), 0.0)
    area = y * (b + z * y)
    wetted_perimeter = b + 2.0 * y * math.sqrt(1.0 + z * z)
    top_width = b + 2.0 * z * y
    hydraulic_radius = area / max(wetted_perimeter, 1e-9)
    hydraulic_depth = area / max(top_width, 1e-9)
    return {
        "A_m2": area,
        "P_m": wetted_perimeter,
        "Be_m": top_width,
        "R_m": hydraulic_radius,
        "Dh_m": hydraulic_depth,
    }


def compute_section(section: SectionInput, params: GlobalParams) -> dict[str, float | str | bool]:
    q = max(section.q_m3s, 1e-9)
    slope = max(section.slope, 1e-9)
    n = max(section.manning_n, 1e-6)
    geom = trapezoid_geometry(section.bottom_width_m, section.depth_m, section.side_slope_z)
    area = geom["A_m2"]
    r = geom["R_m"]
    be = geom["Be_m"]
    dh = geom["Dh_m"]

    v_from_q = q / max(area, 1e-9)
    v_manning = (1.0 / n) * (r ** (2.0 / 3.0)) * math.sqrt(slope)
    q_manning = v_manning * area
    froude = v_from_q / math.sqrt(G * max(dh, 1e-9))
    tau_pa = params.rho_w * G * r * slope
    tau_kg_m2 = tau_pa / G

    d50_m = max(section.d50_mm / 1000.0, 1e-9)
    d90_m = max(section.d90_mm / 1000.0, 1e-9)
    dm_m = max(section.dm_mm / 1000.0, 1e-9)
    theta_d50 = tau_pa / max((params.rho_s - params.rho_w) * G * d50_m, 1e-9)
    theta_dm = tau_pa / max((params.rho_s - params.rho_w) * G * dm_m, 1e-9)

    ks = 1.0 / n
    kr = 26.0 / (d90_m ** (1.0 / 6.0))
    theta_eff = theta_dm * ((ks / kr) ** 1.5) if params.use_roughness_correction and kr > 0 else theta_dm
    theta_c = max(params.theta_c, 1e-9)
    mobility_index = theta_eff / theta_c
    excess = max(0.0, theta_eff - theta_c)
    qb_m2s = 8.0 * (excess ** 1.5) * math.sqrt(max((params.rho_s / params.rho_w - 1.0) * G * (dm_m ** 3), 0.0))
    gs_m3s = qb_m2s * be * max(params.sediment_supply_factor, 0.0)
    gs_m3h = gs_m3s * 3600.0
    gs_tonh = gs_m3h * params.rho_s / 1000.0

    tr = max(params.return_period_years, 1.01)
    alpha = q / max(be * (section.depth_m ** (5.0 / 3.0)), 1e-9)
    beta = 0.7929 + 0.0973 * math.log10(tr)
    logdm = math.log10(max(section.dm_mm, 1e-9))
    z_ll = 0.394557 - 0.04136 * logdm - 0.00891 * (logdm ** 2)
    gamma_mix = max(params.gamma_mix, 0.01)
    phi = 1.0 if gamma_mix <= 1.0 else max(0.1, -0.54 + 1.5143 * gamma_mix)
    denominator = max(0.68 * beta * max(params.mu_ll, 1e-9) * phi * (section.dm_mm ** 0.28), 1e-9)
    hs = ((alpha * (section.depth_m ** (5.0 / 3.0))) / denominator) ** (1.0 / max(1.0 + z_ll, 0.01))
    scour_base = max(0.0, hs - section.depth_m)
    curve_multiplier = section.curve_factor if section.is_curve and section.curve_side.lower().startswith("exterior") else 1.0
    scour_adjusted = scour_base * max(curve_multiplier, 0.0) * max(params.scour_safety_factor, 0.0)
    scoured_bed = section.bed_elevation_m - scour_adjusted

    alerts: list[str] = []
    if froude > 1.0:
        alerts.append("Fr>1: revisar régimen supercrítico")
    if mobility_index > 1.0:
        alerts.append("Lecho móvil potencial")
    if section.is_curve and section.curve_side.lower().startswith("exterior"):
        alerts.append("Curva exterior: revisar erosión lateral")
    if section.d50_mm < 2.0:
        alerts.append("D50 fino: verificar cohesión/limo")
    if abs(q_manning - q) / max(q, 1e-9) > 0.50:
        alerts.append("Q Manning difiere del Q ingresado >50%")

    return {
        "ID": section.section_id,
        "Distancia_m": section.distance_m,
        "dx_m": section.dx_m,
        "Q_m3s": q,
        "S_m_m": slope,
        "n": n,
        "B_m": section.bottom_width_m,
        "y_m": section.depth_m,
        "z_HV": section.side_slope_z,
        "Cota_fondo_m": section.bed_elevation_m,
        "Curva": section.is_curve,
        "Lado_curva": section.curve_side,
        **geom,
        "V_Q_m_s": v_from_q,
        "V_Manning_m_s": v_manning,
        "Q_Manning_m3s": q_manning,
        "Fr": froude,
        "tau_Pa": tau_pa,
        "tau_kg_m2": tau_kg_m2,
        "D50_mm": section.d50_mm,
        "D84_mm": section.d84_mm,
        "D90_mm": section.d90_mm,
        "Dm_mm": section.dm_mm,
        "theta_D50": theta_d50,
        "theta_Dm": theta_dm,
        "theta_eff": theta_eff,
        "theta_c": theta_c,
        "Indice_movilidad": mobility_index,
        "Condicion": "Móvil" if mobility_index > 1.0 else "Estable/incidente",
        "qb_m2s": qb_m2s,
        "Gs_m3s": gs_m3s,
        "Gs_m3h": gs_m3h,
        "Gs_tonh": gs_tonh,
        "alpha_LL": alpha,
        "beta_LL": beta,
        "z_LL": z_ll,
        "Hs_LL_m": hs,
        "Socavacion_base_m": scour_base,
        "Socavacion_ajustada_m": scour_adjusted,
        "Cota_fondo_socavado_m": scoured_bed,
        "Alertas": "; ".join(alerts),
    }


def compute_all(sections: Sequence[SectionInput], params: GlobalParams) -> pd.DataFrame:
    if not sections:
        raise ValueError("No hay secciones para calcular.")
    rows = [compute_section(s, params) for s in sections]
    return pd.DataFrame(rows)


def exner_simulation(results_df: pd.DataFrame, params: GlobalParams, duration_h: float, dt_h: float, upstream_qs_m3h: float) -> pd.DataFrame:
    if results_df.empty:
        raise ValueError("Primero calcula resultados HidroSed.")
    duration_h = max(float(duration_h), 0.0)
    dt_h = max(float(dt_h), 0.01)
    n_steps = max(1, int(math.ceil(duration_h / dt_h)))
    rows = []
    prev_qs = max(float(upstream_qs_m3h), 0.0)
    for _, row in results_df.iterrows():
        be = max(float(row.get("Be_m", 0.0)), 1e-6)
        dx = max(float(row.get("dx_m", 0.0)), 1e-6)
        qs_out = max(float(row.get("Gs_m3h", 0.0)), 0.0)
        dz_step = -((qs_out - prev_qs) * dt_h) / max((1.0 - params.porosity) * be * dx, 1e-9)
        # limitar cambio por paso para evitar explosión en predimensionamiento
        dz_step_limited = max(-0.25, min(0.25, dz_step))
        dz_total = dz_step_limited * n_steps
        tendencia = "Erosión" if dz_total < -0.01 else "Depósito" if dz_total > 0.01 else "Casi estable"
        rows.append({
            "ID": row.get("ID"),
            "Distancia_m": row.get("Distancia_m"),
            "Qs_in_m3h": prev_qs,
            "Qs_out_m3h": qs_out,
            "Delta_z_por_paso_m": dz_step_limited,
            "Delta_z_total_m": dz_total,
            "Tendencia": tendencia,
            "Cota_fondo_inicial_m": row.get("Cota_fondo_m"),
            "Cota_fondo_final_m": safe_float(row.get("Cota_fondo_m"), 0.0) + dz_total,
        })
        prev_qs = qs_out
    return pd.DataFrame(rows)


def technical_summary(results_df: pd.DataFrame) -> dict[str, object]:
    if results_df.empty:
        return {}
    idx_scour = results_df["Socavacion_ajustada_m"].astype(float).idxmax()
    idx_v = results_df["V_Q_m_s"].astype(float).idxmax()
    idx_mob = results_df["Indice_movilidad"].astype(float).idxmax()
    return {
        "n_secciones": int(len(results_df)),
        "velocidad_max_m_s": float(results_df.loc[idx_v, "V_Q_m_s"]),
        "seccion_velocidad_max": str(results_df.loc[idx_v, "ID"]),
        "socavacion_max_m": float(results_df.loc[idx_scour, "Socavacion_ajustada_m"]),
        "seccion_socavacion_max": str(results_df.loc[idx_scour, "ID"]),
        "indice_movilidad_max": float(results_df.loc[idx_mob, "Indice_movilidad"]),
        "seccion_movilidad_max": str(results_df.loc[idx_mob, "ID"]),
        "secciones_moviles": int((results_df["Indice_movilidad"].astype(float) > 1.0).sum()),
        "fr_supercritico": int((results_df["Fr"].astype(float) > 1.0).sum()),
    }


def df_to_html_report(metadata: dict, params: GlobalParams, results: pd.DataFrame, exner_df: pd.DataFrame | None = None) -> str:
    summary = technical_summary(results)
    css = """
    <style>
    body{font-family:Arial, sans-serif; margin:28px; color:#1f2937} h1,h2{color:#0f172a} 
    table{border-collapse:collapse; width:100%; font-size:12px; margin:12px 0} th,td{border:1px solid #d1d5db; padding:5px; text-align:right} th{background:#e5e7eb} td:first-child,th:first-child{text-align:left}
    .card{border:1px solid #cbd5e1; border-radius:10px; padding:12px; margin:12px 0; background:#f8fafc}
    .warn{background:#fff7ed;border-color:#fdba74}.small{font-size:12px;color:#475569}
    </style>
    """
    meta_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in metadata.items() if v is not None)
    summary_rows = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in summary.items())
    exner_html = ""
    if exner_df is not None and not exner_df.empty:
        exner_html = "<h2>Lecho móvil conceptual tipo Exner</h2>" + exner_df.to_html(index=False, float_format=lambda x: f"{x:.4g}")
    return f"""<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Reporte HidroSed Cauces</title>{css}</head>
    <body><h1>Reporte HidroSed Cauces</h1>
    <div class='card'><h2>Identificación y trazabilidad</h2><table>{meta_rows}</table></div>
    <div class='card'><h2>Parámetros globales</h2><table>{''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in asdict(params).items())}</table></div>
    <div class='card'><h2>Resumen crítico</h2><table>{summary_rows}</table></div>
    <h2>Resultados por sección</h2>{results.to_html(index=False, float_format=lambda x: f'{x:.4g}')}
    {exner_html}
    <div class='card warn'><h2>Limitaciones técnicas</h2><p>Herramienta de apoyo para predimensionamiento y revisión. No reemplaza levantamientos topográficos, modelación hidráulica 1D/2D calibrada, granulometría representativa ni revisión profesional competente.</p></div>
    <p class='small'>Generado por HidroSed Cauces · DEM → Curvas → Hidráulica fluvial.</p></body></html>"""
