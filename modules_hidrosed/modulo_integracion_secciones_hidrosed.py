"""Integración secciones v13 -> HidroSed Maestra.

Este módulo recibe las salidas esperadas de la aplicación generadora de secciones
`v13_fix_km_final_utm19s_3d` y las transforma al formato interno de HidroSed
Maestra:

- 03_Secciones
- 04_Puntos_Seccion

La implementación evita depender de Streamlit, salvo en `transferir_a_session_state`,
para que pueda probarse automáticamente y reutilizarse desde otras aplicaciones.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


FUENTE_GEOMETRIA = "v13_fix_km_final_utm19s_3d"
DEFAULT_MANNING_N = 0.035
DEFAULT_Q_M3S = 50.0

REQ_SECCIONES = [
    "id_seccion",
    "pk_m",
    "pk_km",
    "x_centro",
    "y_centro",
    "cota_fondo",
    "cota_borde_izq",
    "cota_borde_der",
    "ancho_superior",
    "ancho_fondo",
    "profundidad_max",
    "area_geom_aprox",
    "pendiente_local",
    "estado",
    "observacion",
]

REQ_PUNTOS = [
    "id_seccion",
    "pk_m",
    "distancia_transversal_m",
    "x_utm",
    "y_utm",
    "z_m",
    "ribera",
    "tipo_punto",
]

HIDROSED_SECCIONES_COLS = [
    "PK_m",
    "PK_km",
    "ID_Seccion",
    "Nombre_Seccion",
    "Caudal_m3s",
    "Pendiente_m_m",
    "Manning_n",
    "Ancho_Cauce_m",
    "Cota_Fondo_m",
    "Cota_Borde_Izq_m",
    "Cota_Borde_Der_m",
    "Profundidad_Max_m",
    "Area_Geometrica_m2",
    "Estado_Modelacion",
    "Fuente_Geometria",
    "Observaciones",
]

HIDROSED_PUNTOS_COLS = [
    "ID_Seccion",
    "Nombre_Seccion",
    "PK_m",
    "Distancia_Transversal_m",
    "X_UTM",
    "Y_UTM",
    "Z_m",
    "Ribera",
    "Tipo_Punto",
    "Usar_En_Modelacion",
]


@dataclass
class ValidationMessage:
    nivel: str
    id_seccion: str | None
    campo: str | None
    mensaje: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "nivel": self.nivel,
            "id_seccion": self.id_seccion,
            "campo": self.campo,
            "mensaje": self.mensaje,
        }


def _clean_col(name: Any) -> str:
    return str(name).strip().replace(" ", "_").replace("-", "_")


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas y alias frecuentes sin modificar el origen."""
    out = df.copy()
    out.columns = [_clean_col(c) for c in out.columns]
    aliases = {
        "ID": "id_seccion",
        "Id_Seccion": "id_seccion",
        "ID_Seccion": "id_seccion",
        "Seccion": "id_seccion",
        "PK": "pk_m",
        "pk": "pk_m",
        "Progresiva_m": "pk_m",
        "PK_m": "pk_m",
        "PK_km": "pk_km",
        "X_Centro": "x_centro",
        "Y_Centro": "y_centro",
        "Cota_Fondo": "cota_fondo",
        "Cota_Fondo_m": "cota_fondo",
        "Cota_Borde_Izq_m": "cota_borde_izq",
        "Cota_Borde_Der_m": "cota_borde_der",
        "Ancho_Cauce_m": "ancho_superior",
        "Ancho_Superior": "ancho_superior",
        "Ancho_Fondo": "ancho_fondo",
        "Profundidad_Max_m": "profundidad_max",
        "Area_Geometrica_m2": "area_geom_aprox",
        "Pendiente_m_m": "pendiente_local",
        "Pendiente": "pendiente_local",
        "Estado_Modelacion": "estado",
        "Observaciones": "observacion",
        "Distancia_Transversal_m": "distancia_transversal_m",
        "X_UTM": "x_utm",
        "Y_UTM": "y_utm",
        "Z_m": "z_m",
        "Ribera": "ribera",
        "Tipo_Punto": "tipo_punto",
    }
    rename = {c: aliases[c] for c in out.columns if c in aliases}
    out = out.rename(columns=rename)
    return out


def _to_num(series: pd.Series, default: float | None = np.nan) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(default)


def _require_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> list[ValidationMessage]:
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return []
    return [
        ValidationMessage(
            nivel="ERROR",
            id_seccion=None,
            campo=", ".join(missing),
            mensaje=f"Faltan columnas obligatorias en {label}: {', '.join(missing)}.",
        )
    ]


def _is_left(value: Any) -> bool:
    txt = str(value).strip().lower()
    return any(t in txt for t in ["izq", "izquierda", "left", "ribera_i"])


def _is_right(value: Any) -> bool:
    txt = str(value).strip().lower()
    return any(t in txt for t in ["der", "derecha", "right", "ribera_d"])


def _section_name_map(df: pd.DataFrame) -> dict[str, str]:
    ids = [str(x) for x in df["id_seccion"].tolist()]
    return {sid: f"SEC_{i + 1:04d}" for i, sid in enumerate(ids)}


def validar_secciones_para_hidrosed(
    df_secciones_validas: pd.DataFrame,
    df_puntos_secciones: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], pd.DataFrame]:
    """Valida que las secciones v13 sean aptas para HidroSed Maestra.

    Retorna `(errores, advertencias, df_filtrado)`.

    - `errores`: fallas críticas de estructura o ausencia total de secciones transferibles.
    - `advertencias`: secciones puntuales no modelables o ajustes aplicados.
    - `df_filtrado`: solo secciones aptas, ordenadas por PK y sin duplicados.
    """
    errores: list[ValidationMessage] = []
    advertencias: list[ValidationMessage] = []

    df_sec = normalizar_columnas(pd.DataFrame(df_secciones_validas)).copy()
    df_pts = normalizar_columnas(pd.DataFrame(df_puntos_secciones)).copy()

    errores.extend(_require_columns(df_sec, REQ_SECCIONES, "df_secciones_validas"))
    errores.extend(_require_columns(df_pts, REQ_PUNTOS, "df_puntos_secciones"))
    if errores:
        return [e.as_dict() for e in errores], [a.as_dict() for a in advertencias], pd.DataFrame()

    for col in [
        "pk_m",
        "pk_km",
        "x_centro",
        "y_centro",
        "cota_fondo",
        "cota_borde_izq",
        "cota_borde_der",
        "ancho_superior",
        "ancho_fondo",
        "profundidad_max",
        "area_geom_aprox",
        "pendiente_local",
    ]:
        df_sec[col] = _to_num(df_sec[col])
    for col in ["pk_m", "distancia_transversal_m", "x_utm", "y_utm", "z_m"]:
        df_pts[col] = _to_num(df_pts[col])

    df_sec["id_seccion"] = df_sec["id_seccion"].astype(str).str.strip()
    df_pts["id_seccion"] = df_pts["id_seccion"].astype(str).str.strip()
    df_sec = df_sec.dropna(subset=["id_seccion", "pk_m"]).sort_values("pk_m").reset_index(drop=True)

    if df_sec.empty:
        errores.append(ValidationMessage("ERROR", None, None, "No existen secciones válidas para transferir."))
        return [e.as_dict() for e in errores], [a.as_dict() for a in advertencias], pd.DataFrame()

    if df_sec["pk_m"].duplicated().any():
        dup = df_sec.loc[df_sec["pk_m"].duplicated(), "pk_m"].tolist()
        errores.append(ValidationMessage("ERROR", None, "pk_m", f"Existen PK duplicados: {dup}."))

    valid_ids: list[str] = []
    invalid_ids: list[str] = []
    for _, row in df_sec.iterrows():
        sid = str(row["id_seccion"])
        pts = df_pts[df_pts["id_seccion"] == sid].copy().sort_values("distancia_transversal_m")
        reasons: list[str] = []

        if len(pts) < 5:
            reasons.append("menos de 5 puntos transversales")
        if not pts["ribera"].map(_is_left).any():
            reasons.append("sin ribera izquierda")
        if not pts["ribera"].map(_is_right).any():
            reasons.append("sin ribera derecha")
        if pts[["x_utm", "y_utm", "z_m", "distancia_transversal_m"]].isna().any().any():
            reasons.append("coordenadas, cotas o distancias nulas")
        if not pts["distancia_transversal_m"].is_monotonic_increasing:
            reasons.append("distancia transversal no creciente")

        z_min = float(pts["z_m"].min()) if not pts.empty else math.nan
        z_max = float(pts["z_m"].max()) if not pts.empty else math.nan
        cota_fondo = float(row["cota_fondo"])
        if not math.isfinite(cota_fondo) or not (z_min - 0.50 <= cota_fondo <= z_max + 0.50):
            reasons.append("cota de fondo fuera del perfil transversal")

        if not math.isfinite(float(row["ancho_superior"])) or float(row["ancho_superior"]) <= 0:
            reasons.append("ancho hidráulico no positivo")
        if not math.isfinite(float(row["profundidad_max"])) or float(row["profundidad_max"]) <= 0:
            reasons.append("profundidad máxima no positiva")
        if not math.isfinite(float(row["pendiente_local"])) or float(row["pendiente_local"]) <= 0:
            reasons.append("pendiente local no positiva")

        if reasons:
            invalid_ids.append(sid)
            advertencias.append(
                ValidationMessage(
                    nivel="NO_MODELABLE",
                    id_seccion=sid,
                    campo=None,
                    mensaje="Sección excluida de transferencia automática: " + "; ".join(reasons) + ".",
                )
            )
        else:
            valid_ids.append(sid)

    df_filtrado = df_sec[df_sec["id_seccion"].isin(valid_ids)].copy().sort_values("pk_m").reset_index(drop=True)

    if df_filtrado.empty:
        errores.append(ValidationMessage("ERROR", None, None, "Ninguna sección cumple las validaciones hidráulicas mínimas."))

    return [e.as_dict() for e in errores], [a.as_dict() for a in advertencias], df_filtrado


def convertir_secciones_a_hidrosed(df_secciones_validas: pd.DataFrame, caudal_default: float | None = None, manning_default: float = DEFAULT_MANNING_N) -> pd.DataFrame:
    """Convierte secciones v13 al formato `03_Secciones` de HidroSed Maestra."""
    df = normalizar_columnas(pd.DataFrame(df_secciones_validas)).copy().sort_values("pk_m").reset_index(drop=True)
    for col in ["pk_m", "pk_km", "pendiente_local", "ancho_superior", "cota_fondo", "cota_borde_izq", "cota_borde_der", "profundidad_max", "area_geom_aprox"]:
        if col in df.columns:
            df[col] = _to_num(df[col])
    name_map = _section_name_map(df)
    out = pd.DataFrame({
        "PK_m": df["pk_m"],
        "PK_km": df["pk_km"],
        "ID_Seccion": df["id_seccion"].astype(str),
        "Nombre_Seccion": df["id_seccion"].astype(str).map(name_map),
        "Caudal_m3s": np.nan if caudal_default is None else float(caudal_default),
        "Pendiente_m_m": df["pendiente_local"],
        "Manning_n": float(manning_default),
        "Ancho_Cauce_m": df["ancho_superior"],
        "Cota_Fondo_m": df["cota_fondo"],
        "Cota_Borde_Izq_m": df["cota_borde_izq"],
        "Cota_Borde_Der_m": df["cota_borde_der"],
        "Profundidad_Max_m": df["profundidad_max"],
        "Area_Geometrica_m2": df["area_geom_aprox"],
        "Estado_Modelacion": "Modelable",
        "Fuente_Geometria": FUENTE_GEOMETRIA,
        "Observaciones": df.get("observacion", pd.Series([""] * len(df))).fillna(""),
    })
    return out[HIDROSED_SECCIONES_COLS]


def convertir_puntos_a_hidrosed(df_puntos_secciones: pd.DataFrame, df_secciones_validas: pd.DataFrame) -> pd.DataFrame:
    """Convierte puntos transversales al formato `04_Puntos_Seccion`."""
    df_sec = normalizar_columnas(pd.DataFrame(df_secciones_validas)).copy().sort_values("pk_m").reset_index(drop=True)
    df_pts = normalizar_columnas(pd.DataFrame(df_puntos_secciones)).copy()
    name_map = _section_name_map(df_sec)
    ids_validos = set(df_sec["id_seccion"].astype(str))
    df_pts["id_seccion"] = df_pts["id_seccion"].astype(str).str.strip()
    df_pts = df_pts[df_pts["id_seccion"].isin(ids_validos)].copy()
    for col in ["pk_m", "distancia_transversal_m", "x_utm", "y_utm", "z_m"]:
        df_pts[col] = _to_num(df_pts[col])
    df_pts = df_pts.sort_values(["id_seccion", "distancia_transversal_m"]).reset_index(drop=True)
    out = pd.DataFrame({
        "ID_Seccion": df_pts["id_seccion"],
        "Nombre_Seccion": df_pts["id_seccion"].map(name_map),
        "PK_m": df_pts["pk_m"],
        "Distancia_Transversal_m": df_pts["distancia_transversal_m"],
        "X_UTM": df_pts["x_utm"],
        "Y_UTM": df_pts["y_utm"],
        "Z_m": df_pts["z_m"],
        "Ribera": df_pts["ribera"].astype(str),
        "Tipo_Punto": df_pts["tipo_punto"].astype(str),
        "Usar_En_Modelacion": "SI",
    })
    return out[HIDROSED_PUNTOS_COLS]


def hidrosed_03_to_sections_df(df_hidrosed_secciones: pd.DataFrame, q_default: float = DEFAULT_Q_M3S) -> pd.DataFrame:
    """Adapta `03_Secciones` al esquema actual de cálculo HidroSed Cauces.

    La hidráulica detallada todavía usa una geometría trapecial equivalente. La geometría
    transversal completa queda conservada en `04_Puntos_Seccion` para trazabilidad y futuras
    mejoras de cálculo por perfil real.
    """
    df = pd.DataFrame(df_hidrosed_secciones).copy()
    # Normaliza ausencia de caudal: aplica q_default para permitir cálculo inmediato.
    q = pd.to_numeric(df.get("Caudal_m3s", np.nan), errors="coerce").fillna(q_default)
    pk = pd.to_numeric(df["PK_m"], errors="coerce").fillna(0.0)
    dx = pk.diff().abs().replace(0, np.nan).fillna(pk.diff(-1).abs()).fillna(100.0)
    depth = pd.to_numeric(df["Profundidad_Max_m"], errors="coerce").fillna(1.0).clip(lower=0.01)
    top_width = pd.to_numeric(df["Ancho_Cauce_m"], errors="coerce").fillna(1.0).clip(lower=0.1)
    bottom_width = (top_width * 0.55).clip(lower=0.1)
    # Talud equivalente desde ancho superior, ancho de fondo y profundidad.
    side_z = ((top_width - bottom_width) / (2.0 * depth)).replace([np.inf, -np.inf], np.nan).fillna(1.5).clip(lower=0.0)

    out = pd.DataFrame({
        "section_id": df["Nombre_Seccion"].fillna(df["ID_Seccion"]).astype(str),
        "distance_m": pk,
        "dx_m": dx,
        "q_m3s": q,
        "slope": pd.to_numeric(df["Pendiente_m_m"], errors="coerce").fillna(0.001).clip(lower=1e-6),
        "manning_n": pd.to_numeric(df["Manning_n"], errors="coerce").fillna(DEFAULT_MANNING_N).clip(lower=0.005),
        "bottom_width_m": bottom_width,
        "depth_m": depth,
        "side_slope_z": side_z,
        "bed_elevation_m": pd.to_numeric(df["Cota_Fondo_m"], errors="coerce").fillna(0.0),
        "is_curve": False,
        "curve_side": "Eje",
        "curve_factor": 1.0,
        "d50_mm": 32.0,
        "d84_mm": 64.0,
        "d90_mm": 90.0,
        "dm_mm": 45.0,
    })
    return out


def transferir_a_session_state(
    df_hidrosed_secciones: pd.DataFrame,
    df_hidrosed_puntos: pd.DataFrame,
    df_descartadas: pd.DataFrame | None = None,
    advertencias: list[dict[str, Any]] | None = None,
) -> None:
    """Guarda las tablas convertidas en `st.session_state` para HidroSed Maestra."""
    import streamlit as st

    st.session_state["hidrosed_secciones"] = pd.DataFrame(df_hidrosed_secciones).to_dict("records")
    st.session_state["hidrosed_puntos_seccion"] = pd.DataFrame(df_hidrosed_puntos).to_dict("records")
    st.session_state["origen_geometria"] = FUENTE_GEOMETRIA
    st.session_state["secciones_transferidas"] = True
    st.session_state["secciones_descartadas_v13"] = pd.DataFrame(df_descartadas if df_descartadas is not None else []).to_dict("records")
    st.session_state["advertencias_integracion_secciones"] = advertencias or []
    st.session_state["sections_df"] = hidrosed_03_to_sections_df(pd.DataFrame(df_hidrosed_secciones)).to_dict("records")
    st.session_state["modulo_activo"] = "hidraulica"


def generar_excel_integracion_hidrosed(
    df_hidrosed_secciones: pd.DataFrame,
    df_hidrosed_puntos: pd.DataFrame,
    df_descartadas: pd.DataFrame | None = None,
    df_perfil_longitudinal: pd.DataFrame | None = None,
    metadatos: dict[str, Any] | None = None,
    output_path: Path | None = None,
) -> Path | BytesIO:
    """Genera Excel de respaldo con hojas 03/04, descartadas, perfil y metadatos."""
    buffer: Path | BytesIO
    if output_path is None:
        buffer = BytesIO()
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        buffer = output_path
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(df_hidrosed_secciones).to_excel(writer, index=False, sheet_name="03_Secciones")
        pd.DataFrame(df_hidrosed_puntos).to_excel(writer, index=False, sheet_name="04_Puntos_Seccion")
        pd.DataFrame(df_descartadas if df_descartadas is not None else []).to_excel(writer, index=False, sheet_name="Secciones_Descartadas")
        pd.DataFrame(df_perfil_longitudinal if df_perfil_longitudinal is not None else []).to_excel(writer, index=False, sheet_name="Perfil_Longitudinal")
        meta_rows = [{"campo": k, "valor": str(v)} for k, v in (metadatos or {}).items()]
        pd.DataFrame(meta_rows).to_excel(writer, index=False, sheet_name="Metadatos")
    if isinstance(buffer, BytesIO):
        buffer.seek(0)
    return buffer


def leer_salidas_v13_desde_excel(file: Any) -> dict[str, pd.DataFrame]:
    """Lee un Excel de salidas v13 o de integración y devuelve dataframes normalizados.

    Admite hojas con nombres esperados o aproximados:
    - df_secciones_validas / Secciones_Validas / 03_Secciones
    - df_puntos_secciones / Puntos_Secciones / 04_Puntos_Seccion
    - df_secciones_descartadas / Secciones_Descartadas
    - df_perfil_longitudinal / Perfil_Longitudinal
    """
    xls = pd.ExcelFile(file)
    names = {name.lower().strip(): name for name in xls.sheet_names}

    def pick(candidates: list[str]) -> pd.DataFrame:
        for cand in candidates:
            key = cand.lower().strip()
            if key in names:
                return pd.read_excel(xls, sheet_name=names[key])
        return pd.DataFrame()

    sec = pick(["df_secciones_validas", "Secciones_Validas", "Secciones", "03_Secciones"])
    pts = pick(["df_puntos_secciones", "Puntos_Secciones", "Puntos", "04_Puntos_Seccion"])
    desc = pick(["df_secciones_descartadas", "Secciones_Descartadas", "Descartadas"])
    perfil = pick(["df_perfil_longitudinal", "Perfil_Longitudinal", "Perfil"])

    # Si viene en formato 03/04, convertir a formato v13 aproximado para que pase por el mismo flujo.
    if not sec.empty and "PK_m" in sec.columns:
        sec = pd.DataFrame({
            "id_seccion": sec.get("ID_Seccion"),
            "pk_m": sec.get("PK_m"),
            "pk_km": sec.get("PK_km", pd.to_numeric(sec.get("PK_m"), errors="coerce") / 1000.0),
            "x_centro": np.nan,
            "y_centro": np.nan,
            "cota_fondo": sec.get("Cota_Fondo_m"),
            "cota_borde_izq": sec.get("Cota_Borde_Izq_m"),
            "cota_borde_der": sec.get("Cota_Borde_Der_m"),
            "ancho_superior": sec.get("Ancho_Cauce_m"),
            "ancho_fondo": pd.to_numeric(sec.get("Ancho_Cauce_m"), errors="coerce") * 0.55,
            "profundidad_max": sec.get("Profundidad_Max_m"),
            "area_geom_aprox": sec.get("Area_Geometrica_m2"),
            "pendiente_local": sec.get("Pendiente_m_m"),
            "estado": sec.get("Estado_Modelacion", "Modelable"),
            "observacion": sec.get("Observaciones", "Importado desde 03_Secciones"),
        })
    if not pts.empty and "ID_Seccion" in pts.columns:
        pts = pd.DataFrame({
            "id_seccion": pts.get("ID_Seccion"),
            "pk_m": pts.get("PK_m"),
            "distancia_transversal_m": pts.get("Distancia_Transversal_m"),
            "x_utm": pts.get("X_UTM"),
            "y_utm": pts.get("Y_UTM"),
            "z_m": pts.get("Z_m"),
            "ribera": pts.get("Ribera"),
            "tipo_punto": pts.get("Tipo_Punto"),
        })
    return {
        "df_secciones_validas": normalizar_columnas(sec),
        "df_puntos_secciones": normalizar_columnas(pts),
        "df_secciones_descartadas": normalizar_columnas(desc),
        "df_perfil_longitudinal": normalizar_columnas(perfil),
    }


def generar_datos_demo_v13(n_secciones: int = 5, puntos_por_seccion: int = 11) -> dict[str, pd.DataFrame]:
    """Crea un set demo para pruebas internas y demostración en Streamlit."""
    rows_sec = []
    rows_pts = []
    for i in range(n_secciones):
        pk = i * 100.0
        fondo = 100.0 - i * 0.8
        ancho = 20.0 + i * 1.5
        prof = 2.0 + 0.15 * i
        sid = f"S{i+1:03d}"
        rows_sec.append({
            "id_seccion": sid,
            "pk_m": pk,
            "pk_km": pk / 1000.0,
            "x_centro": 300000.0 + i * 30,
            "y_centro": 6680000.0 - i * 95,
            "cota_fondo": fondo,
            "cota_borde_izq": fondo + prof,
            "cota_borde_der": fondo + prof * 0.95,
            "ancho_superior": ancho,
            "ancho_fondo": ancho * 0.50,
            "profundidad_max": prof,
            "area_geom_aprox": ancho * prof * 0.65,
            "pendiente_local": 0.008,
            "estado": "Válida",
            "observacion": "Demo integración v13",
        })
        distances = np.linspace(-ancho / 2, ancho / 2, puntos_por_seccion)
        for d in distances:
            rel = abs(d) / (ancho / 2)
            z = fondo + prof * (rel ** 1.6)
            if d < -ancho * 0.35:
                rib = "Izquierda"
            elif d > ancho * 0.35:
                rib = "Derecha"
            elif abs(d) < 1e-6:
                rib = "Eje"
            else:
                rib = "Terreno"
            rows_pts.append({
                "id_seccion": sid,
                "pk_m": pk,
                "distancia_transversal_m": float(d),
                "x_utm": 300000.0 + i * 30 + d,
                "y_utm": 6680000.0 - i * 95,
                "z_m": float(z),
                "ribera": rib,
                "tipo_punto": "Fondo" if abs(d) < 1e-6 else "Terreno",
            })
    perfil = pd.DataFrame({"PK_m": [r["pk_m"] for r in rows_sec], "Cota_Fondo_m": [r["cota_fondo"] for r in rows_sec]})
    return {
        "df_secciones_validas": pd.DataFrame(rows_sec),
        "df_puntos_secciones": pd.DataFrame(rows_pts),
        "df_secciones_descartadas": pd.DataFrame(),
        "df_perfil_longitudinal": perfil,
    }
