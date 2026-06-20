"""Visualización 3D del cauce y secciones transversales para HidroSed Maestra.

El objetivo es revisar técnicamente la geometría transferida desde v13 o desde
las tablas HidroSed `03_Secciones` y `04_Puntos_Seccion` antes de ejecutar o
interpretar resultados hidráulicos. El módulo no depende de Streamlit, de modo
que puede probarse de forma independiente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules_hidrosed.modulo_integracion_secciones_hidrosed import normalizar_columnas


LEFT_WORDS = {"izq", "izquierda", "left", "ribera_i", "i"}
RIGHT_WORDS = {"der", "derecha", "right", "ribera_d", "d"}


def _num(s: pd.Series, default: float = np.nan) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce").fillna(default)


def _first_existing(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def preparar_puntos_3d(
    df_puntos: pd.DataFrame,
    df_secciones: pd.DataFrame | None = None,
    incluir_solo_modelables: bool = True,
) -> pd.DataFrame:
    """Normaliza puntos transversales para graficar en 3D.

    Acepta tanto el formato v13 (`x_utm`, `y_utm`, `z_m`) como el formato
    HidroSed (`X_UTM`, `Y_UTM`, `Z_m`), porque `normalizar_columnas` homologa
    los nombres.
    """
    pts = normalizar_columnas(pd.DataFrame(df_puntos)).copy()
    if pts.empty:
        return pd.DataFrame()

    # Alias frecuentes adicionales.
    aliases = {
        "ID_Seccion": "id_seccion",
        "Nombre_Seccion": "nombre_seccion",
        "PK_m": "pk_m",
        "Distancia_Transversal_m": "distancia_transversal_m",
        "X_UTM": "x_utm",
        "Y_UTM": "y_utm",
        "Z_m": "z_m",
        "Usar_En_Modelacion": "usar_en_modelacion",
    }
    pts = pts.rename(columns={c: aliases[c] for c in pts.columns if c in aliases})

    required = ["id_seccion", "x_utm", "y_utm", "z_m"]
    if any(c not in pts.columns for c in required):
        return pd.DataFrame()

    if "distancia_transversal_m" not in pts.columns:
        pts["distancia_transversal_m"] = 0.0
    if "ribera" not in pts.columns:
        pts["ribera"] = ""
    if "tipo_punto" not in pts.columns:
        pts["tipo_punto"] = "Terreno"
    if "usar_en_modelacion" not in pts.columns:
        pts["usar_en_modelacion"] = "SI"

    for col in ["x_utm", "y_utm", "z_m", "pk_m", "distancia_transversal_m"]:
        if col in pts.columns:
            pts[col] = _num(pts[col])

    pts["id_seccion"] = pts["id_seccion"].astype(str)
    pts = pts.dropna(subset=["x_utm", "y_utm", "z_m"])

    if incluir_solo_modelables:
        mask = pts["usar_en_modelacion"].astype(str).str.upper().isin(["SI", "SÍ", "YES", "TRUE", "1", "MODELABLE", ""])
        pts = pts.loc[mask].copy()

    sec = normalizar_columnas(pd.DataFrame(df_secciones if df_secciones is not None else [])).copy()
    if not sec.empty:
        sec = sec.rename(columns={c: aliases[c] for c in sec.columns if c in aliases})
        if "id_seccion" in sec.columns:
            keep_cols = ["id_seccion"]
            for c in ["nombre_seccion", "pk_m", "x_centro", "y_centro", "cota_fondo", "estado"]:
                if c in sec.columns and c not in keep_cols:
                    keep_cols.append(c)
            sec["id_seccion"] = sec["id_seccion"].astype(str)
            pts = pts.merge(sec[keep_cols].drop_duplicates("id_seccion"), on="id_seccion", how="left", suffixes=("", "_sec"))
            if "pk_m_sec" in pts.columns:
                pts["pk_m"] = pts.get("pk_m", np.nan)
                pts["pk_m"] = pts["pk_m"].where(pts["pk_m"].notna(), pts["pk_m_sec"])

    if "pk_m" not in pts.columns or pts["pk_m"].isna().all():
        order = {sid: i for i, sid in enumerate(sorted(pts["id_seccion"].unique()))}
        pts["pk_m"] = pts["id_seccion"].map(order).astype(float)

    if "nombre_seccion" not in pts.columns:
        pts["nombre_seccion"] = pts["id_seccion"]
    pts["nombre_seccion"] = pts["nombre_seccion"].fillna(pts["id_seccion"])

    pts = pts.sort_values(["pk_m", "id_seccion", "distancia_transversal_m"]).reset_index(drop=True)
    return pts


def _sample_section_ids(pts: pd.DataFrame, max_sections: int) -> list[str]:
    sec_order = pts.groupby("id_seccion")["pk_m"].median().sort_values()
    ids = sec_order.index.astype(str).tolist()
    if len(ids) <= max_sections:
        return ids
    idx = np.linspace(0, len(ids) - 1, max_sections).round().astype(int)
    return [ids[i] for i in sorted(set(idx))]


def _axis_from_points(pts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in pts.groupby("id_seccion", sort=False):
        g = g.sort_values("distancia_transversal_m")
        if len(g) == 0:
            continue
        # El eje se estima como el punto más cercano a distancia transversal cero.
        idx = (g["distancia_transversal_m"].abs()).idxmin()
        row = g.loc[idx].copy()
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    axis = pd.DataFrame(rows).sort_values("pk_m")
    return axis


def _z_scaled(z: np.ndarray | pd.Series, base_z: float, exageracion_vertical: float) -> np.ndarray:
    arr = np.asarray(z, dtype=float)
    return base_z + (arr - base_z) * float(exageracion_vertical)


def _resample_sections_for_mesh(pts: pd.DataFrame, max_cross_points: int = 45) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Re-muestrea secciones a un número común de puntos para formar una malla."""
    ids = pts.groupby("id_seccion")["pk_m"].median().sort_values().index.astype(str).tolist()
    if len(ids) < 2:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), []

    # Número de puntos transversal razonable, limitado por la sección más pobre.
    min_count = int(pts.groupby("id_seccion").size().min())
    n_cross = max(3, min(int(max_cross_points), min_count))
    t_common = np.linspace(0.0, 1.0, n_cross)

    xs, ys, zs = [], [], []
    valid_ids = []
    for sid in ids:
        g = pts.loc[pts["id_seccion"].astype(str) == sid].sort_values("distancia_transversal_m")
        if len(g) < 3:
            continue
        d = g["distancia_transversal_m"].to_numpy(dtype=float)
        if np.nanmax(d) - np.nanmin(d) < 1e-9:
            t = np.linspace(0, 1, len(g))
        else:
            t = (d - np.nanmin(d)) / max(np.nanmax(d) - np.nanmin(d), 1e-9)
        xs.append(np.interp(t_common, t, g["x_utm"].to_numpy(dtype=float)))
        ys.append(np.interp(t_common, t, g["y_utm"].to_numpy(dtype=float)))
        zs.append(np.interp(t_common, t, g["z_m"].to_numpy(dtype=float)))
        valid_ids.append(sid)

    if len(valid_ids) < 2:
        return np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0)), []
    return np.vstack(xs), np.vstack(ys), np.vstack(zs), valid_ids


def _add_surface_mesh(fig: go.Figure, pts: pd.DataFrame, base_z: float, exageracion_vertical: float, opacity: float = 0.33) -> None:
    xg, yg, zg, ids = _resample_sections_for_mesh(pts)
    if xg.size == 0:
        return
    zg_plot = _z_scaled(zg, base_z, exageracion_vertical)
    n_sec, n_cross = xg.shape
    vertices_x = xg.ravel()
    vertices_y = yg.ravel()
    vertices_z = zg_plot.ravel()
    i_list, j_list, k_list = [], [], []
    for i in range(n_sec - 1):
        for j in range(n_cross - 1):
            a = i * n_cross + j
            b = (i + 1) * n_cross + j
            c = i * n_cross + (j + 1)
            d = (i + 1) * n_cross + (j + 1)
            i_list.extend([a, c])
            j_list.extend([b, b])
            k_list.extend([c, d])
    fig.add_trace(
        go.Mesh3d(
            x=vertices_x,
            y=vertices_y,
            z=vertices_z,
            i=i_list,
            j=j_list,
            k=k_list,
            name="Superficie simplificada cauce",
            color="lightgray",
            opacity=opacity,
            hoverinfo="skip",
            showscale=False,
        )
    )


def _add_axis(fig: go.Figure, pts: pd.DataFrame, base_z: float, exageracion_vertical: float) -> None:
    axis = _axis_from_points(pts)
    if axis.empty:
        return
    fig.add_trace(
        go.Scatter3d(
            x=axis["x_utm"],
            y=axis["y_utm"],
            z=_z_scaled(axis["z_m"], base_z, exageracion_vertical),
            mode="lines+markers",
            name="Eje longitudinal estimado",
            line={"color": "black", "width": 7},
            marker={"size": 4, "color": "black"},
            customdata=np.stack([axis["id_seccion"].astype(str), axis["pk_m"].astype(float)], axis=-1),
            hovertemplate="Eje<br>Sección %{customdata[0]}<br>PK %{customdata[1]:.1f} m<br>X %{x:.2f}<br>Y %{y:.2f}<br>Z visual %{z:.2f}<extra></extra>",
        )
    )


def _add_sections(fig: go.Figure, pts: pd.DataFrame, base_z: float, exageracion_vertical: float) -> None:
    for sid, g in pts.groupby("id_seccion", sort=False):
        g = g.sort_values("distancia_transversal_m")
        if len(g) < 2:
            continue
        estado = str(g.get("estado", pd.Series([""])).iloc[0]) if "estado" in g.columns else ""
        color = "royalblue" if not any(t in estado.lower() for t in ["desc", "no", "revis"]) else "red"
        nombre = str(g["nombre_seccion"].iloc[0]) if "nombre_seccion" in g.columns else str(sid)
        pk = float(g["pk_m"].median()) if "pk_m" in g.columns else np.nan
        fig.add_trace(
            go.Scatter3d(
                x=g["x_utm"],
                y=g["y_utm"],
                z=_z_scaled(g["z_m"], base_z, exageracion_vertical),
                mode="lines+markers",
                name=f"{nombre} · PK {pk:.0f} m",
                line={"color": color, "width": 4},
                marker={"size": 3, "color": color},
                customdata=np.stack([
                    g["distancia_transversal_m"].astype(float),
                    g["z_m"].astype(float),
                    g["ribera"].astype(str),
                    g["tipo_punto"].astype(str),
                ], axis=-1),
                hovertemplate=(
                    f"{nombre}<br>PK {pk:.1f} m"
                    "<br>Dist. transversal %{customdata[0]:.2f} m"
                    "<br>Z real %{customdata[1]:.2f} m"
                    "<br>Ribera %{customdata[2]}"
                    "<br>Tipo %{customdata[3]}"
                    "<br>X %{x:.2f}<br>Y %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )


def _add_water_and_scour(
    fig: go.Figure,
    pts: pd.DataFrame,
    resultados: pd.DataFrame | None,
    base_z: float,
    exageracion_vertical: float,
    mostrar_agua: bool,
    mostrar_socavacion: bool,
) -> None:
    if resultados is None or pd.DataFrame(resultados).empty:
        return
    res = pd.DataFrame(resultados).copy()
    res.columns = [str(c).strip() for c in res.columns]
    id_col = _first_existing(res, ["ID", "ID_Seccion", "section_id", "id_seccion"])
    if id_col is None:
        return
    res[id_col] = res[id_col].astype(str)
    y_col = _first_existing(res, ["y_m", "Tirante_m", "Altura_Normal_m", "Profundidad_m"])
    bed_col = _first_existing(res, ["Cota_fondo_m", "Cota_Fondo_m", "bed_elevation_m"])
    scour_col = _first_existing(res, ["Cota_fondo_socavado_m", "Cota_Fondo_Socavado_m"])

    for sid, g in pts.groupby("id_seccion", sort=False):
        r = res.loc[res[id_col].astype(str) == str(sid)]
        if r.empty:
            continue
        g = g.sort_values("distancia_transversal_m")
        if mostrar_agua and y_col is not None:
            bed = float(r.iloc[0][bed_col]) if bed_col is not None and pd.notna(r.iloc[0][bed_col]) else float(g["z_m"].min())
            y = float(r.iloc[0][y_col]) if pd.notna(r.iloc[0][y_col]) else 0.0
            z_water = bed + max(y, 0.0)
            fig.add_trace(
                go.Scatter3d(
                    x=g["x_utm"],
                    y=g["y_utm"],
                    z=np.full(len(g), _z_scaled(np.array([z_water]), base_z, exageracion_vertical)[0]),
                    mode="lines",
                    name="Nivel de agua calculado",
                    line={"color": "deepskyblue", "width": 5, "dash": "dash"},
                    hovertemplate=f"Nivel de agua<br>Sección {sid}<br>Z real {z_water:.2f} m<extra></extra>",
                    showlegend=False,
                )
            )
        if mostrar_socavacion and scour_col is not None and pd.notna(r.iloc[0][scour_col]):
            z_scour = float(r.iloc[0][scour_col])
            # Línea de fondo socavado aproximada bajo el tramo central de la sección.
            d = g["distancia_transversal_m"].astype(float)
            mask = d.abs() <= max(float(d.abs().quantile(0.35)), 1.0)
            sg = g.loc[mask] if mask.any() else g
            fig.add_trace(
                go.Scatter3d(
                    x=sg["x_utm"],
                    y=sg["y_utm"],
                    z=np.full(len(sg), _z_scaled(np.array([z_scour]), base_z, exageracion_vertical)[0]),
                    mode="lines",
                    name="Cota fondo socavado",
                    line={"color": "orangered", "width": 6},
                    hovertemplate=f"Fondo socavado<br>Sección {sid}<br>Z real {z_scour:.2f} m<extra></extra>",
                    showlegend=False,
                )
            )



def aplicar_vista_3d(
    fig: go.Figure,
    vista: str | None = "Isométrica",
    projection_type: str = "perspective",
) -> go.Figure:
    """Aplica cámaras predefinidas al modelo 3D Plotly.

    Todas las vistas siguen siendo interactivas: el usuario puede rotar,
    hacer zoom y desplazar el modelo después de cargar la cámara inicial.
    """
    vista_norm = (vista or "Isométrica").strip().lower()
    projection_type = "orthographic" if str(projection_type).lower().startswith("ortho") else "perspective"

    cameras = {
        "isométrica": {"eye": {"x": 1.55, "y": -1.65, "z": 1.05}, "up": {"x": 0, "y": 0, "z": 1}},
        "isometrica": {"eye": {"x": 1.55, "y": -1.65, "z": 1.05}, "up": {"x": 0, "y": 0, "z": 1}},
        "rotación libre": {"eye": {"x": 1.55, "y": -1.65, "z": 1.05}, "up": {"x": 0, "y": 0, "z": 1}},
        "rotacion libre": {"eye": {"x": 1.55, "y": -1.65, "z": 1.05}, "up": {"x": 0, "y": 0, "z": 1}},
        "planta / superior": {"eye": {"x": 0.0, "y": 0.0, "z": 2.8}, "up": {"x": 0, "y": 1, "z": 0}},
        "vista superior": {"eye": {"x": 0.0, "y": 0.0, "z": 2.8}, "up": {"x": 0, "y": 1, "z": 0}},
        "superior": {"eye": {"x": 0.0, "y": 0.0, "z": 2.8}, "up": {"x": 0, "y": 1, "z": 0}},
        "planta": {"eye": {"x": 0.0, "y": 0.0, "z": 2.8}, "up": {"x": 0, "y": 1, "z": 0}},
        "lateral": {"eye": {"x": 2.35, "y": 0.0, "z": 0.38}, "up": {"x": 0, "y": 0, "z": 1}},
        "perfil longitudinal": {"eye": {"x": 2.35, "y": 0.0, "z": 0.38}, "up": {"x": 0, "y": 0, "z": 1}},
        "aguas abajo": {"eye": {"x": 0.0, "y": -2.35, "z": 0.38}, "up": {"x": 0, "y": 0, "z": 1}},
        "aguas arriba": {"eye": {"x": 0.0, "y": 2.35, "z": 0.38}, "up": {"x": 0, "y": 0, "z": 1}},
        "frontal": {"eye": {"x": 0.0, "y": -2.35, "z": 0.38}, "up": {"x": 0, "y": 0, "z": 1}},
    }
    camera = cameras.get(vista_norm, cameras["isométrica"]).copy()
    camera["projection"] = {"type": projection_type}
    fig.update_layout(
        scene_camera=camera,
        scene_dragmode="orbit",
        uirevision=f"modelo_3d_{vista_norm}_{projection_type}",
    )
    return fig

def generar_figura_3d_cauce_secciones(
    df_puntos: pd.DataFrame,
    df_secciones: pd.DataFrame | None = None,
    resultados: pd.DataFrame | None = None,
    max_sections: int = 80,
    max_cross_points: int = 45,
    exageracion_vertical: float = 1.0,
    mostrar_superficie: bool = True,
    mostrar_agua: bool = True,
    mostrar_socavacion: bool = True,
    vista_3d: str | None = "Isométrica",
    projection_type: str = "perspective",
) -> go.Figure:
    """Genera figura Plotly 3D con eje, secciones y malla simplificada.

    `vista_3d` permite fijar una cámara inicial: Isométrica, Planta / superior,
    Lateral, Aguas abajo, Aguas arriba o Rotación libre. La figura sigue siendo
    completamente interactiva.
    """
    pts = preparar_puntos_3d(df_puntos, df_secciones=df_secciones, incluir_solo_modelables=True)
    if pts.empty:
        fig = go.Figure()
        fig.update_layout(title="Modelo 3D no disponible: faltan puntos transversales con X_UTM, Y_UTM y Z_m")
        return fig

    ids = _sample_section_ids(pts, max_sections=max(2, int(max_sections)))
    pts = pts.loc[pts["id_seccion"].astype(str).isin(ids)].copy()
    # Limita puntos por sección si son excesivos.
    sampled = []
    for _, g in pts.groupby("id_seccion", sort=False):
        if len(g) > max_cross_points:
            idx = np.linspace(0, len(g) - 1, max_cross_points).round().astype(int)
            g = g.iloc[sorted(set(idx))]
        sampled.append(g)
    pts = pd.concat(sampled, ignore_index=True) if sampled else pts

    base_z = float(np.nanmin(pts["z_m"].to_numpy(dtype=float)))
    fig = go.Figure()
    if mostrar_superficie:
        _add_surface_mesh(fig, pts, base_z, exageracion_vertical)
    _add_sections(fig, pts, base_z, exageracion_vertical)
    _add_axis(fig, pts, base_z, exageracion_vertical)
    _add_water_and_scour(fig, pts, resultados, base_z, exageracion_vertical, mostrar_agua, mostrar_socavacion)

    xmin, xmax = float(pts["x_utm"].min()), float(pts["x_utm"].max())
    ymin, ymax = float(pts["y_utm"].min()), float(pts["y_utm"].max())
    z_plot = _z_scaled(pts["z_m"], base_z, exageracion_vertical)
    zmin, zmax = float(np.nanmin(z_plot)), float(np.nanmax(z_plot))
    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)
    dz = max(zmax - zmin, 1.0)
    max_horizontal = max(dx, dy)

    fig.update_layout(
        title="Modelo 3D del cauce y secciones transversales",
        scene={
            "xaxis_title": "X UTM [m]",
            "yaxis_title": "Y UTM [m]",
            "zaxis_title": f"Cota [m] · exageración {exageracion_vertical:g}x",
            "aspectmode": "manual",
            "aspectratio": {"x": dx / max_horizontal, "y": dy / max_horizontal, "z": max(0.08, min(0.8, dz / max_horizontal * 8))},
        },
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        legend={"orientation": "h", "y": 0.02, "x": 0.01},
        height=720,
    )
    if vista_3d:
        aplicar_vista_3d(fig, vista_3d, projection_type=projection_type)
    return fig


def exportar_modelo_3d_html(fig: go.Figure, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)
    return path
