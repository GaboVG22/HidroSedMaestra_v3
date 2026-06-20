import os
import tempfile
from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import streamlit as st
import folium
import matplotlib.pyplot as plt
from folium.plugins import Draw, Fullscreen, MeasureControl
from streamlit_folium import st_folium
from shapely.geometry import shape, mapping

from src.hidro_kmz_core import (
    read_kmz_or_kml,
    parse_kml_features,
    classify_features,
    compute_basin_metrics,
    compute_contour_metrics,
    sample_line_profile,
    estimate_outlet_from_contours,
    estimate_channel_from_contours,
    detect_methods,
    profile_summary,
    HydrologyInputs,
    STANDARD_LAND_USES,
    VERNI_KING_PRESETS,
    DGA_AC_PRESETS,
    guess_chile_region_basin,
    input_diagnostics,
    project_geom,
    compute_hydrology,
    isohyet_features_from_kml,
    nearest_isohyet_to_point,
    debris_cv_classes_dataframe,
    takahashi_cv,
    debris_discharge_table,
    sediment_volume_estimates,
    empirical_debris_peak_from_volume,
    debris_susceptibility,
)
from src.hidro_report import generate_memoria_bytes
from src.hidro_refs import (
    IDF_REFERENCE_STATIONS,
    intensity_from_p10d,
    p24_from_p10d,
    direct_idf_intensity,
    rainfall_reference_dataframe,
    METHODOLOGY_REFERENCES,
)
from src.historical_data import (
    available_historical_files,
    build_station_catalog,
    nearest_stations,
    pmax24_frequency_estimates,
    sediment_nearest_summary,
    climate_nearest_summary,
)
from src.hidro_advanced import (
    compute_mociornita_slope_pct,
    time_concentration_methods,
    design_flow_recommendation,
    rainfall_crosscheck_summary,
)
from src.hidro_design import (
    risk_table,
    return_period_for_risk,
    pluvial_area_linear,
    hydraulic_verification_table,
    objective_design_guidance,
)

# st.set_page_config se controla desde app.py maestro

st.title("Módulo hidrológico KMZ avanzado")
st.caption("Versión maestra v6.7 — integrada con DEM → curvas → HidroSed Cauces")

DEMO_KMZ_PATH = Path(__file__).parent / "data" / "Quebrada_Las_Cardas_2_1.kmz"
DEMO_ISOHYET_PATH = Path(__file__).parent / "data" / "Precipitaciones_Maxima_Diarias.kmz"
HISTORICAL_DATA_DIR = Path(__file__).parent / "data" / "historico"
DEMO_INTENSITY = {2.0: 18.0, 5.0: 23.0, 10.0: 28.0, 25.0: 34.0, 50.0: 39.0, 100.0: 45.0}
DEMO_P24 = {2.0: 25.0, 5.0: 35.0, 10.0: 45.0, 25.0: 58.0, 50.0: 70.0, 100.0: 85.0}


def parse_periods(defaults, manual_text):
    periods = set(float(x) for x in defaults)
    if manual_text.strip():
        for token in manual_text.replace(";", ",").split(","):
            token = token.strip()
            if token:
                try:
                    periods.add(float(token.replace(",", ".")))
                except Exception:
                    pass
    return sorted(periods)



def safe_map_geometry(geom, simplify_deg=0.00008):
    """Return a simplified GeoJSON mapping or None, avoiding map crashes with heavy/invalid KML."""
    if geom is None or getattr(geom, "is_empty", False):
        return None
    try:
        g = geom
        if not g.is_valid:
            g = g.buffer(0)
        # Simplify in geographic degrees only for visualization. Calculations use original geometry.
        g = g.simplify(simplify_deg, preserve_topology=True)
        return mapping(g)
    except Exception:
        return None

def create_basin_plot(basin_feature, contours, channel_geom, outlet_geom, epsg, out_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=170)
    basin_m = project_geom(basin_feature.geometry, epsg)
    xb, yb = basin_m.exterior.xy
    ax.plot(xb, yb, linewidth=2, label="Cuenca")
    for c in contours[:900]:
        try:
            gm = project_geom(c.geometry, epsg)
            geoms = [gm] if gm.geom_type == "LineString" else list(getattr(gm, "geoms", []))
            for g in geoms:
                if hasattr(g, "xy"):
                    x, y = g.xy
                    ax.plot(x, y, linewidth=0.35, alpha=0.55)
        except Exception:
            pass
    if channel_geom is not None:
        try:
            chm = project_geom(channel_geom, epsg)
            x, y = chm.xy
            ax.plot(x, y, linewidth=2.5, label="Cauce principal")
        except Exception:
            pass
    if outlet_geom is not None:
        try:
            pm = project_geom(outlet_geom, epsg)
            ax.scatter([pm.x], [pm.y], s=45, marker="o", label="Punto de descarga")
        except Exception:
            pass
    ax.set_title("Cuenca, curvas de nivel y elementos hidráulicos")
    ax.set_xlabel(f"UTM E (m) EPSG:{epsg}")
    ax.set_ylabel("UTM N (m)")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_factors_from_df(df, col="factor"):
    out = {}
    for _, r in df.iterrows():
        try:
            T = float(r["T_anios"])
            v = float(r[col])
            if np.isfinite(T) and np.isfinite(v):
                out[T] = v
        except Exception:
            pass
    return out


# -----------------------
# Entrada espacial inicial
# -----------------------
with st.sidebar:
    st.header("1. Entrada espacial")
    kmz_file = st.file_uploader("Archivo KMZ/KML de cuenca y curvas", type=["kmz", "kml"])
    channel_file = st.file_uploader("Cauce principal KMZ/KML opcional", type=["kmz", "kml"])
    isohyet_file = st.file_uploader("Mapa de isoyetas KMZ/KML opcional", type=["kmz", "kml"], help="Debe contener líneas o polígonos de isoyetas con valor en nombre, descripción, atributo o coordenada Z.")
    use_builtin_dga_isohyets = st.checkbox("Usar KMZ DGA de precipitaciones máximas diarias incluido si no subo isoyetas", value=True)
    show_isohyets_map = st.checkbox("Mostrar isoyetas en el mapa", value=False, help="Para KMZ grandes conviene dejarlo desactivado. La detección de P10D funciona igual.")
    max_isohyets_map = st.number_input("Máx. isoyetas a dibujar", min_value=50, max_value=1000, value=250, step=50)
    use_demo = st.checkbox("Usar archivo demo Las Cardas si no subo KMZ", value=True)
    generated_workspace_path = st.session_state.get("hidro_workspace_kmz_path")
    generated_basin_path = st.session_state.get("basin_auto_kmz_path")
    use_generated_workspace = False
    use_generated_basin = False
    if generated_workspace_path:
        use_generated_workspace = st.checkbox(
            "Usar KMZ combinado generado automáticamente",
            value=True,
            help="Usa el KMZ armado automáticamente con la cuenca delimitada y las curvas generadas desde DEM.",
        )
    elif generated_basin_path:
        use_generated_basin = st.checkbox(
            "Usar cuenca delimitada automáticamente",
            value=True,
            help="Usa el KMZ de cuenca generado desde DEM + punto de control, aunque aún no exista KMZ combinado con curvas.",
        )
    auto_outlet_enabled = st.checkbox("Sugerir punto de descarga automáticamente", value=True)
    auto_channel_enabled = st.checkbox("Estimar eje de cauce si no viene cargado", value=True)
    use_demo_hydrology = st.checkbox("Cargar datos hidrológicos demo para verificar", value=True)
    st.caption("Los datos demo y presets preliminares permiten probar la app; deben reemplazarse por datos oficiales/adoptados antes de diseño.")

if kmz_file is None:
    if 'use_generated_workspace' in locals() and use_generated_workspace and generated_workspace_path and Path(str(generated_workspace_path)).exists():
        st.info("Se cargó automáticamente el KMZ combinado generado desde DEM + cuenca + curvas. Puedes continuar con morfometría e hidrología sin subir nuevamente el archivo.")
        kmz_source = str(generated_workspace_path)
        demo_active = False
    elif 'use_generated_basin' in locals() and use_generated_basin and generated_basin_path and Path(str(generated_basin_path)).exists():
        st.info("Se cargó automáticamente la cuenca delimitada desde DEM. Para mejorar morfometría, genera también curvas y el KMZ combinado.")
        kmz_source = str(generated_basin_path)
        demo_active = False
    elif use_demo and DEMO_KMZ_PATH.exists():
        st.info("No se subió KMZ. Se cargó automáticamente el archivo demo de Quebrada Las Cardas para verificar funcionamiento.")
        kmz_source = str(DEMO_KMZ_PATH)
        demo_active = True
    else:
        st.info("Suba un archivo KMZ/KML. Puede contener polígono de cuenca, curvas de nivel, cauce principal y punto de descarga.")
        st.markdown(
            """
            ### Flujo de trabajo
            1. Cargar KMZ/KML de cuenca y curvas de nivel, o generar un KMZ combinado en la etapa DEM/curvas.  
            2. Confirmar o dibujar eje del cauce principal.  
            3. Señalar punto de descarga o punto de control.  
            4. Seleccionar objetivo y periodos de retorno.  
            5. Ingresar lluvia, uso de suelo y parámetros regionales.  
            6. Revisar métodos recomendados, resultados y memoria técnica.
            """
        )
        st.stop()
else:
    kmz_source = kmz_file
    demo_active = False

try:
    kml_text = read_kmz_or_kml(kmz_source)
    features = parse_kml_features(kml_text)
    classes = classify_features(features)
except Exception as e:
    st.error(f"No fue posible leer el KMZ/KML: {e}")
    st.stop()

basin_f = classes["basin"]
if basin_f is None:
    st.error("No se detectó polígono de cuenca en el archivo.")
    st.stop()

contours = classes["contours"]
channels = classes["channels"]
outlets = classes["outlets"]

if channel_file is not None:
    try:
        ch_kml = read_kmz_or_kml(channel_file)
        ch_features = parse_kml_features(ch_kml)
        ch_classes = classify_features(ch_features)
        channels += ch_classes["channels"] + ch_classes["other_lines"]
        outlets += ch_classes["outlets"]
    except Exception as e:
        st.warning(f"No se pudo leer el archivo de cauce: {e}")

isohyets = []
isohyet_source_name = ""
try:
    iso_source = None
    if isohyet_file is not None:
        iso_source = isohyet_file
        isohyet_source_name = getattr(isohyet_file, "name", "KMZ/KML cargado")
    elif use_builtin_dga_isohyets and DEMO_ISOHYET_PATH.exists():
        iso_source = str(DEMO_ISOHYET_PATH)
        isohyet_source_name = "Base incluida: Precipitaciones_Maxima_Diarias.kmz"
    if iso_source is not None:
        iso_kml = read_kmz_or_kml(iso_source)
        isohyets = isohyet_features_from_kml(iso_kml)
        if not isohyets:
            st.warning("El archivo de isoyetas fue leído, pero no contiene líneas/polígonos reconocibles.")
except Exception as e:
    st.warning(f"No se pudo leer el archivo KMZ/KML de isoyetas: {e}")

metrics = compute_basin_metrics(basin_f)
contour_metrics = compute_contour_metrics(contours)
region_guess = guess_chile_region_basin(metrics.centroid_lon, metrics.centroid_lat)

estimated_outlet = estimate_outlet_from_contours(basin_f, contours) if auto_outlet_enabled else None
pre_channel_geom = max([ch.geometry for ch in channels], key=lambda g: g.length) if channels else None
pre_outlet_geom = outlets[0].geometry if outlets else estimated_outlet
channel_auto_info = {}
if pre_channel_geom is None and auto_channel_enabled and pre_outlet_geom is not None:
    pre_channel_geom, channel_auto_info = estimate_channel_from_contours(basin_f, contours, pre_outlet_geom)

# -----------------------
# Configuración hidrológica dependiente de la cuenca
# -----------------------
with st.sidebar:
    st.header("2. Objetivo")
    objective = st.selectbox(
        "Objetivo del cálculo",
        [
            "Quebrada natural / descarga aluvial",
            "Atravieso / alcantarilla / baden",
            "Canal o conduccion",
            "Urbanizacion / aguas lluvias",
            "Embalse pequeno / regulacion",
            "Bocatoma",
            "Estudio comparativo preliminar",
        ],
        index=0,
    )
    project_name = st.text_input("Nombre del proyecto/cuenca", "Quebrada Las Cardas" if demo_active else "Cuenca en análisis")

    st.header("3. Periodos de retorno")
    periods = [2, 5, 10, 25, 50, 100, 200]
    st.info("Periodos de retorno fijos del modelo: 2, 5, 10, 25, 50, 100 y 200 años.")
    st.caption("El periodo 200 años queda incluido para verificación y sensibilidad. En métodos DGA/Verni-King, los factores precargados para 200 años son referenciales/editables y deben confirmarse con fuente oficial o criterio del especialista.")

    st.header("4. Uso de suelo")
    land_use = st.selectbox("Uso de suelo estándar", list(STANDARD_LAND_USES.keys()))
    default_C = STANDARD_LAND_USES[land_use]["C"]
    default_CN = STANDARD_LAND_USES[land_use]["CN"]
    C = st.number_input("Coeficiente de escorrentía C", min_value=0.01, max_value=1.0, value=float(default_C), step=0.01)
    CN = st.number_input("Número de curva CN", min_value=30.0, max_value=99.0, value=float(default_CN), step=1.0)
    storm_duration_h = st.number_input("Duración tormenta para SCS/HUS (h)", min_value=0.25, max_value=72.0, value=24.0, step=0.25)

    st.header("5. Preset regional editable")
    st.caption(f"Sugerencia automática: {region_guess['region']} / {region_guess['cuenca_sugerida']} ({region_guess['confianza']}).")
    regional_mode = st.radio(
        "Coeficientes DGA/Verni-King",
        ["Usar recomendación automática editable", "Ingresar manualmente"],
        index=0,
        horizontal=False,
        help="La app recomienda presets regionales según ubicación aproximada; todos los coeficientes quedan editables antes de calcular."
    )
    vk_options = list(VERNI_KING_PRESETS.keys())
    suggested_vk = region_guess["cuenca_sugerida"] if region_guess["cuenca_sugerida"] in vk_options else "Manual"
    if regional_mode == "Ingresar manualmente":
        suggested_vk = "Manual"
    vk_preset_name = st.selectbox("Preset Verni-King recomendado/editable", vk_options, index=vk_options.index(suggested_vk))
    dga_options = list(DGA_AC_PRESETS.keys())
    zona_txt = str(region_guess.get("zona_dga_sugerida", "")) + " " + str(region_guess.get("cuenca_sugerida", ""))
    if "Jp" in zona_txt or "Limar" in zona_txt:
        dga_default = "Zona Jp pluvial Limarí - envolvente máxima informe Punitaqui (editable)"
    elif "Choapa" in zona_txt or "Kp" in zona_txt:
        dga_default = "Zona Kp pluvial Choapa (editable)"
    elif region_guess["region"] == "Coquimbo":
        dga_default = "Zona Ip pluvial III-IV / Elqui-Huasco-Copiapo (editable)"
    else:
        dga_default = "Manual"
    if regional_mode == "Ingresar manualmente":
        dga_default = "Manual"
    dga_preset_name = st.selectbox("Preset DGA-AC recomendado/editable", dga_options, index=dga_options.index(dga_default) if dga_default in dga_options else 0)
    st.caption("Los factores y coeficientes cargados son recomendaciones operacionales para automatizar la corrida. En v6.3 se corrigió Limarí Jp para permitir la envolvente Máx. con α=2,14 y se conserva alternativa media regional; validar con zona homogénea/cuenca oficial DGA.")
    has_fluvio = st.checkbox("Existe estación fluviométrica DGA representativa", value=False)

    st.header("6. Lluvia automática DGA/MC")
    auto_rain_mode = st.selectbox(
        "Modo de automatización de lluvia",
        [
            "No usar",
            "Generar IDF y P24 desde P10D + CD/CF/K",
            "Usar tabla IDF directa de estación"
        ],
        index=1,
    )
    # Por defecto se usa Rivadavia-Elqui para automatizar la lluvia en Coquimbo; el usuario puede cambiarla manualmente.
    idf_station = st.selectbox("Estación/localidad de referencia", list(IDF_REFERENCE_STATIONS.keys()), index=1)
    use_isohyet_p10d = st.checkbox("Usar KMZ de isoyetas para P10D si está disponible", value=True)
    p10d_mm = st.number_input("P10D diaria 10 años (mm)", min_value=0.0, value=45.0, step=1.0, help="Valor manual de respaldo. Si se activa el KMZ de isoyetas y se detecta una curva válida, la app usa la isoyeta más cercana al punto de control.")
    k_24h = st.number_input("K corrección 24h / diaria", min_value=0.80, max_value=1.30, value=1.10, step=0.01)
    area_reduction = st.number_input("Coeficiente de abatimiento espacial CA", min_value=0.10, max_value=1.00, value=1.00, step=0.01)
    duration_override_h = st.number_input("Duración IDF manual si no hay Tc (h)", min_value=0.083, max_value=24.0, value=1.0, step=0.083)
    override_manual_rain = st.checkbox("Sobrescribir valores de lluvia de la tabla con la automatización", value=False)

    st.header("7. Base histórica DGA")
    use_historical_data = st.checkbox("Usar bases históricas regionales incluidas", value=True, help="Activa estaciones cercanas, Pmax24 anual, precipitación mensual, temperaturas y sedimentos.")
    hist_radius_km = st.number_input("Radio búsqueda estaciones históricas (km)", min_value=10.0, max_value=500.0, value=150.0, step=10.0)
    hist_n_stations = st.number_input("N° estaciones cercanas a mostrar", min_value=1, max_value=20, value=8, step=1)
    hist_pmax24_mode = st.selectbox(
        "Uso de Pmax24 histórica",
        ["Solo comparar", "Rellenar P24 faltantes", "Sobrescribir P24"],
        index=0,
        help="Usa análisis de frecuencia Gumbel de la estación Pmax24 anual más cercana. Para diseño debe validarse representatividad y largo de registro."
    )
    hist_freq_model = st.selectbox(
        "Modelo de frecuencia histórica para P24",
        ["Gumbel", "Normal", "LogNormal", "LogPearsonIII", "MediaModelos", "MaxModelos"],
        index=0,
        help="Permite comparar modelos usados en análisis de frecuencia. Para diseño definitivo debe justificarse el modelo adoptado."
    )

    st.header("8. Tiempo de concentración avanzado")
    texas_condition = st.selectbox(
        "Condición método Velocidad Texas",
        ["Poca vegetación", "Bosque partes altas", "Cauce natural no muy definido"],
        index=0,
        help="Velocidades referenciales para estimar Tc por método de velocidad, según pendiente y cobertura."
    )

    st.header("9. Línea de nieve / área pluvial")
    use_pluvial_area = st.checkbox("Aplicar área pluvial efectiva por línea de nieve", value=False, help="Útil para cuencas nival/pluviales. Si está desactivado, se usa el área total de la cuenca.")
    snowline_m = st.number_input("Línea de nieve adoptada (m s.n.m.)", min_value=0.0, max_value=7000.0, value=2500.0, step=50.0)
    st.caption("La estimación automática usa una hipsometría lineal entre cota mínima y máxima. Para memoria definitiva debe reemplazarse por línea de nieve/área pluvial validada.")

    st.header("10. Riesgo y diseño hidráulico")
    design_life_years = st.number_input("Vida útil para riesgo de falla (años)", min_value=1.0, max_value=200.0, value=50.0, step=1.0)
    target_risk = st.number_input("Riesgo objetivo de referencia", min_value=0.001, max_value=0.95, value=0.10, step=0.01, format="%.3f")
    hydraulic_enabled = st.checkbox("Activar verificación hidráulica preliminar", value=True)
    channel_b_m = st.number_input("Ancho basal canal/sección b (m)", min_value=0.10, max_value=50.0, value=1.00, step=0.10)
    channel_z_hv = st.number_input("Talud lateral z H:V", min_value=0.00, max_value=5.00, value=1.50, step=0.25)
    manning_n = st.number_input("Rugosidad Manning n", min_value=0.010, max_value=0.150, value=0.040, step=0.005, format="%.3f")
    hydraulic_slope_pct_manual = st.number_input("Pendiente hidráulica manual si no hay perfil (%)", min_value=0.01, max_value=100.0, value=2.00, step=0.10)
    freeboard_m = st.number_input("Revancha preliminar (m)", min_value=0.0, max_value=5.0, value=0.30, step=0.05)
    design_depth_m = st.number_input("Altura constructiva disponible opcional (m)", min_value=0.0, max_value=20.0, value=0.0, step=0.10, help="Si se deja en 0, no se evalúa capacidad con altura fija.")

# -----------------------
# Mapa y dibujo interactivo
# -----------------------
center = [metrics.centroid_lat, metrics.centroid_lon]
m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
Fullscreen().add_to(m)
MeasureControl(position="bottomleft").add_to(m)
gj = safe_map_geometry(basin_f.geometry, simplify_deg=0.00002)
if gj is not None:
    folium.GeoJson(gj, name="Cuenca", style_function=lambda x: {"fillColor":"#99ccff", "color":"blue", "weight":2, "fillOpacity":0.15}).add_to(m)
for c in contours[:650]:
    gj = safe_map_geometry(c.geometry, simplify_deg=0.00005)
    if gj is not None:
        folium.GeoJson(gj, name="Curva", style_function=lambda x: {"color":"#6b6b6b", "weight":1, "fillOpacity":0}).add_to(m)
if 'show_isohyets_map' in locals() and show_isohyets_map:
    iso_to_draw = isohyets[:int(max_isohyets_map)]
    if len(isohyets) > len(iso_to_draw):
        st.info(f"El KMZ contiene {len(isohyets)} isoyetas. Para proteger la app se dibujan solo {len(iso_to_draw)}; el cálculo usa todas.")
    for iso in iso_to_draw:
        val = iso.properties.get("p_mm")
        try:
            name = f"Isoyeta {float(val):g} mm" if val is not None and not pd.isna(val) else (iso.name or "Isoyeta")
        except Exception:
            name = iso.name or "Isoyeta"
        gj = safe_map_geometry(iso.geometry, simplify_deg=0.00008)
        if gj is not None:
            folium.GeoJson(gj, name=name, style_function=lambda x: {"color":"#008000", "weight":2, "fillOpacity":0}).add_to(m)
for ch in channels:
    gj = safe_map_geometry(ch.geometry, simplify_deg=0.00003)
    if gj is not None:
        folium.GeoJson(gj, name="Cauce cargado", style_function=lambda x: {"color":"#0077ff", "weight":4}).add_to(m)
if pre_channel_geom is not None and not channels:
    gj = safe_map_geometry(pre_channel_geom, simplify_deg=0.00003)
    if gj is not None:
        folium.GeoJson(gj, name="Cauce preliminar automático", style_function=lambda x: {"color":"#00a8ff", "weight":3, "dashArray":"6,6"}).add_to(m)
for o in outlets:
    folium.Marker(location=[o.geometry.y, o.geometry.x], popup=o.name or "Punto de salida", icon=folium.Icon(color="red", icon="tint", prefix="fa")).add_to(m)
if estimated_outlet is not None:
    folium.Marker(location=[estimated_outlet.y, estimated_outlet.x], popup="Salida probable por curva más baja", icon=folium.Icon(color="orange", icon="flag", prefix="fa")).add_to(m)

Draw(
    export=False,
    position="topleft",
    draw_options={"polyline": True, "polygon": False, "rectangle": False, "circle": False, "marker": True, "circlemarker": False},
    edit_options={"edit": True},
).add_to(m)
folium.LayerControl().add_to(m)

st.subheader("Mapa de trabajo")
st.caption("Puede dibujar una polilínea para reemplazar el cauce preliminar y un marcador para fijar la descarga/control.")
map_data = st_folium(m, width=None, height=590, returned_objects=["all_drawings"])

channel_geom = pre_channel_geom
outlet_geom = pre_outlet_geom
if map_data and map_data.get("all_drawings"):
    for feat in map_data["all_drawings"]:
        try:
            geom = shape(feat["geometry"])
            if geom.geom_type == "LineString":
                channel_geom = geom
            elif geom.geom_type == "Point":
                outlet_geom = geom
        except Exception:
            pass

nearest_iso = nearest_isohyet_to_point(isohyets, outlet_geom, metrics.epsg_utm) if isohyets else {"ok": False, "message": "No se cargó KMZ de isoyetas.", "p_mm": np.nan, "distance_m": np.nan, "n_isohyets": 0}

if channel_auto_info and not channels:
    st.warning("Eje de cauce preliminar automático: debe validarse o reemplazarse por el eje real. " + channel_auto_info.get("message", ""))

if isohyets:
    if nearest_iso.get("ok"):
        st.success(f"Isoyeta más cercana al punto de control: {nearest_iso['p_mm']:.2f} mm; distancia {nearest_iso['distance_m']:.1f} m; curva: {nearest_iso['name']}. Fuente: {isohyet_source_name}.")
        if auto_rain_mode == "No usar":
            st.info("Se detectó P10D desde isoyetas, pero la automatización de lluvia está desactivada. Para completar IDF y P24 active: 'Generar IDF y P24 desde P10D + CD/CF/K'.")
    else:
        st.warning("KMZ de isoyetas cargado, pero no se pudo obtener P automáticamente: " + nearest_iso.get("message", ""))

# -----------------------
# Tablas editables de lluvia y parámetros regionales
# -----------------------
st.subheader("Lluvia de diseño y parámetros regionales editables")
col_rain, col_vk, col_dga = st.columns(3)
with col_rain:
    st.markdown("**Lluvia por periodo**")
    rain_rows = []
    for T in periods:
        T_float = float(T)
        rain_rows.append({
            "T_anios": T_float,
            "Intensidad_IDF_mm_h": DEMO_INTENSITY.get(T_float, np.nan) if (demo_active and use_demo_hydrology) else np.nan,
            "P24_mm": DEMO_P24.get(T_float, np.nan) if (demo_active and use_demo_hydrology) else np.nan,
        })
    rain_df = pd.DataFrame(rain_rows)
    rain_edit = st.data_editor(rain_df, num_rows="dynamic", use_container_width=True, key="rain_table")
    intensity_by_T = {float(r["T_anios"]): float(r["Intensidad_IDF_mm_h"]) for _, r in rain_edit.iterrows() if pd.notna(r.get("Intensidad_IDF_mm_h"))}
    p24_by_T = {float(r["T_anios"]): float(r["P24_mm"]) for _, r in rain_edit.iterrows() if pd.notna(r.get("P24_mm"))}

with col_vk:
    st.markdown("**Verni-King**")
    vk_preset = VERNI_KING_PRESETS.get(vk_preset_name, VERNI_KING_PRESETS["Manual"])
    c10 = st.number_input("C(10) Verni-King", value=float(0 if pd.isna(vk_preset.get("c10")) else vk_preset.get("c10")), min_value=0.0, step=0.001, format="%.4f")
    ratios = vk_preset.get("ratios", {}) or {}
    vk_rows = []
    for T in periods:
        ratio = ratios.get(float(T), ratios.get(int(T), np.nan))
        vk_rows.append({"T_anios": float(T), "factor_CT_C10": ratio})
    vk_df = pd.DataFrame(vk_rows)
    vk_edit = st.data_editor(vk_df, num_rows="dynamic", use_container_width=True, key="vk_table")
    vk_ratios = parse_factors_from_df(vk_edit, col="factor_CT_C10")
    vk_params = {"mode": "verni_king_mod", "c10": c10 if c10 > 0 else np.nan, "ratios": vk_ratios, "const": 0.00618, "m": 1.24, "n": 0.88, "preset": vk_preset_name}

with col_dga:
    st.markdown("**DGA-AC**")
    dga_preset = DGA_AC_PRESETS.get(dga_preset_name, DGA_AC_PRESETS["Manual"])
    alpha_default = dga_preset.get("alpha_inst", np.nan)
    alpha_inst = st.number_input("Alfa instantáneo DGA-AC", value=float(1.25 if pd.isna(alpha_default) else alpha_default), min_value=0.0, step=0.05, format="%.3f")
    dga_rows = []
    dga_f = dga_preset.get("factors", {}) or {}
    for T in periods:
        factor = dga_f.get(float(T), dga_f.get(int(T), np.nan))
        dga_rows.append({"T_anios": float(T), "F_T_QT_Q10": factor})
    dga_df = pd.DataFrame(dga_rows)
    dga_edit = st.data_editor(dga_df, num_rows="dynamic", use_container_width=True, key="dga_table")
    dga_factors = parse_factors_from_df(dga_edit, col="F_T_QT_Q10")
    if dga_preset.get("mode") == "iii_iv_formula":
        dga_params = {"mode": "iii_iv_formula", "coef": dga_preset.get("coef"), "area_exp": dga_preset.get("area_exp"), "p_exp": dga_preset.get("p_exp"), "factors": dga_factors, "alpha_inst": alpha_inst, "preset": dga_preset_name}
    else:
        dga_params = {"mode": "generic", "qref": np.nan, "alpha": np.nan, "kinst": np.nan, "factors": dga_factors}

# -----------------------
# Perfil y cálculos
# -----------------------
profile_df = pd.DataFrame()
if channel_geom is not None and contours:
    try:
        step = max(20.0, min(100.0, metrics.max_geom_length_km * 1000 / 150))
        profile_df = sample_line_profile(channel_geom, contours, metrics.epsg_utm, step_m=step)
    except Exception as e:
        st.warning(f"No fue posible generar perfil longitudinal: {e}")

prof_summary = profile_summary(metrics, contour_metrics, profile_df)

# Módulos metodológicos avanzados: pendiente Mociornita y múltiples tiempos de concentración.
mociornita_info = compute_mociornita_slope_pct(basin_f.geometry, contours, metrics.epsg_utm)
tc_methods_df = time_concentration_methods(
    metrics.area_km2,
    prof_summary.get("L_km", np.nan),
    prof_summary.get("H_m", np.nan),
    prof_summary.get("S_pct", np.nan),
    texas_condition if 'texas_condition' in locals() else "Poca vegetación",
)
try:
    tc_adopted_row = tc_methods_df[tc_methods_df["metodo_Tc"] == "Adoptado automático"].iloc[0]
    prof_summary["Tc_adoptado_min"] = float(tc_adopted_row["Tc_min"])
    prof_summary["Tc_adoptado_h"] = float(tc_adopted_row["Tc_h"])
except Exception:
    prof_summary["Tc_adoptado_min"] = prof_summary.get("Tc_kirpich_min", np.nan)
    prof_summary["Tc_adoptado_h"] = prof_summary.get("Tc_kirpich_min", np.nan) / 60.0 if pd.notna(prof_summary.get("Tc_kirpich_min", np.nan)) else np.nan
prof_summary["S_mociornita_pct"] = mociornita_info.get("S_mociornita_pct", np.nan)
prof_summary["Mociornita_estado"] = mociornita_info.get("estado", "")

# Línea de nieve / área pluvial efectiva para cuencas nivo-pluviales.
pluvial_area_info = pluvial_area_linear(
    metrics.area_km2,
    contour_metrics.z_min if contour_metrics.z_min is not None else np.nan,
    contour_metrics.z_max if contour_metrics.z_max is not None else np.nan,
    snowline_m if 'snowline_m' in locals() else np.nan,
)
effective_area_km2 = metrics.area_km2
if 'use_pluvial_area' in locals() and use_pluvial_area and pd.notna(pluvial_area_info.get("area_pluvial_km2", np.nan)):
    effective_area_km2 = max(float(pluvial_area_info["area_pluvial_km2"]), 0.001)
prof_summary["Area_total_km2"] = metrics.area_km2
prof_summary["Area_pluvial_adoptada_km2"] = effective_area_km2
prof_summary["Fraccion_pluvial"] = pluvial_area_info.get("fraccion_pluvial", np.nan)
prof_summary["Linea_nieve_m"] = pluvial_area_info.get("linea_nieve_m", np.nan)
prof_summary["Area_pluvial_estado"] = pluvial_area_info.get("estado", "")

# Base histórica regional: estaciones cercanas y estimación P24 por frecuencia.
historical_files_df = pd.DataFrame()
historical_catalog_df = pd.DataFrame()
nearest_historical_df = pd.DataFrame()
pmax24_station_summary_df = pd.DataFrame()
pmax24_freq_df = pd.DataFrame()
sed_hist_df = pd.DataFrame()
climate_hist_df = pd.DataFrame()
if 'use_historical_data' in locals() and use_historical_data and HISTORICAL_DATA_DIR.exists():
    try:
        historical_files_df = available_historical_files(HISTORICAL_DATA_DIR)
        historical_catalog_df = build_station_catalog(str(HISTORICAL_DATA_DIR.resolve()))
        if not historical_catalog_df.empty:
            ref_pt = outlet_geom if outlet_geom is not None else basin_f.geometry.centroid
            nearest_historical_df = nearest_stations(
                historical_catalog_df, float(ref_pt.x), float(ref_pt.y),
                n=int(hist_n_stations), max_km=float(hist_radius_km)
            )
            pmax24_station_summary_df, pmax24_freq_df = pmax24_frequency_estimates(
                HISTORICAL_DATA_DIR, float(ref_pt.x), float(ref_pt.y), periods,
                n=max(3, int(hist_n_stations)), max_km=float(hist_radius_km)
            )
            sed_hist_df = sediment_nearest_summary(HISTORICAL_DATA_DIR, float(ref_pt.x), float(ref_pt.y), n=int(hist_n_stations), max_km=max(float(hist_radius_km), 200.0))
            climate_hist_df = climate_nearest_summary(HISTORICAL_DATA_DIR, float(ref_pt.x), float(ref_pt.y), n=int(hist_n_stations), max_km=float(hist_radius_km))
    except Exception as e:
        st.warning(f"No fue posible procesar la base histórica regional: {e}")

# Automatización de lluvia: genera intensidades IDF y P24 cuando no se ingresan manualmente.
# La duración usada para IDF corresponde al Tc Kirpich si existe; si no, usa la duración manual.
auto_rain_rows = []
tc_for_idf_h = prof_summary.get("Tc_adoptado_h", np.nan) if pd.notna(prof_summary.get("Tc_adoptado_h", np.nan)) else (prof_summary.get("Tc_kirpich_min", np.nan) / 60.0 if pd.notna(prof_summary.get("Tc_kirpich_min", np.nan)) else np.nan)
duration_for_idf_h = float(tc_for_idf_h) if pd.notna(tc_for_idf_h) and tc_for_idf_h > 0 else float(duration_override_h)
p10d_calc = float(p10d_mm)
p10d_source = "Manual"
if use_isohyet_p10d and nearest_iso.get("ok") and pd.notna(nearest_iso.get("p_mm")):
    p10d_calc = float(nearest_iso["p_mm"])
    p10d_source = f"KMZ isoyetas: {nearest_iso.get('name','')}"
if auto_rain_mode != "No usar" and idf_station != "Manual":
    for T in periods:
        T = float(T)
        I_auto = np.nan
        P24_auto = np.nan
        fuente = auto_rain_mode
        if auto_rain_mode == "Generar IDF y P24 desde P10D + CD/CF/K":
            I_auto = intensity_from_p10d(idf_station, p10d_calc, T, duration_for_idf_h, k_24h=k_24h, area_reduction=area_reduction)
            P24_auto = p24_from_p10d(idf_station, p10d_calc, T, k_24h=k_24h, area_reduction=area_reduction)
        elif auto_rain_mode == "Usar tabla IDF directa de estación":
            I_auto = direct_idf_intensity(idf_station, T, duration_for_idf_h)
            # Para P24 de métodos de volumen se mantiene la estimación CD/CF si existe P10D.
            P24_auto = p24_from_p10d(idf_station, p10d_calc, T, k_24h=k_24h, area_reduction=area_reduction) if p10d_calc > 0 else np.nan
        if pd.notna(I_auto) and (override_manual_rain or T not in intensity_by_T or pd.isna(intensity_by_T.get(T))):
            intensity_by_T[T] = float(I_auto)
        if pd.notna(P24_auto) and (override_manual_rain or T not in p24_by_T or pd.isna(p24_by_T.get(T))):
            p24_by_T[T] = float(P24_auto)
        auto_rain_rows.append({"T_anios": T, "duracion_IDF_h": duration_for_idf_h, "P10D_usado_mm": p10d_calc, "fuente_P10D": p10d_source, "I_auto_mm_h": I_auto, "P24_auto_mm": P24_auto, "fuente": fuente, "estacion": idf_station})
auto_rain_df = pd.DataFrame(auto_rain_rows)
if auto_rain_mode != "No usar" and not auto_rain_df.empty:
    st.success(f"Lluvia automatizada activa: {len(p24_by_T)} periodos con P24 y {len(intensity_by_T)} periodos con IDF generados/rellenados desde P10D.")

# Complemento: análisis de frecuencia de Pmax24 anual desde estación histórica cercana.
historical_p24_used_df = pd.DataFrame()
if 'hist_pmax24_mode' in locals() and hist_pmax24_mode != "Solo comparar" and pmax24_freq_df is not None and not pmax24_freq_df.empty:
    first_code = str(pmax24_freq_df.sort_values("distancia_km").iloc[0]["codigo"])
    hist_sel = pmax24_freq_df[pmax24_freq_df["codigo"].astype(str) == first_code].copy()
    model_col = f"P24_{hist_freq_model}_mm" if 'hist_freq_model' in locals() else "P24_Gumbel_mm"
    if model_col not in hist_sel.columns:
        model_col = "P24_Gumbel_mm"
    for _, r in hist_sel.iterrows():
        T = float(r["T_anios"])
        val = float(r[model_col]) if pd.notna(r.get(model_col, np.nan)) else np.nan
        if pd.notna(val):
            if hist_pmax24_mode == "Sobrescribir P24" or T not in p24_by_T or pd.isna(p24_by_T.get(T)):
                p24_by_T[T] = val
    historical_p24_used_df = hist_sel
    st.info(f"P24 histórica aplicada en modo '{hist_pmax24_mode}' con modelo {model_col.replace('P24_','').replace('_mm','')} usando estación {hist_sel.iloc[0]['estacion']} ({first_code}).")

methods_df = detect_methods(
    effective_area_km2,
    has_idf=len(intensity_by_T) > 0,
    has_p24=len(p24_by_T) > 0,
    has_contours=len(contours) > 0,
    has_channel=channel_geom is not None,
    has_fluvio=has_fluvio,
    region=region_guess.get("region", ""),
    T_max=max(periods) if periods else 100,
)

diagnostics_df = input_diagnostics(metrics, contours, channel_geom, outlet_geom, intensity_by_T, p24_by_T, dga_params, vk_params)

# Para métodos hidrológicos se usa una copia de métricas. No se modifica el área real desplegada de la cuenca.
hydro_metrics = replace(metrics, area_km2=effective_area_km2)

hydro_inputs = HydrologyInputs(
    objective=objective,
    return_periods=periods,
    land_use=land_use,
    C=C,
    CN=CN,
    intensity_by_T=intensity_by_T,
    p24_by_T=p24_by_T,
    storm_duration_h=storm_duration_h,
    dga_ac_params=dga_params,
    verni_king_params=vk_params,
)
results_df, hydrographs = compute_hydrology(hydro_metrics, contour_metrics, profile_df, hydro_inputs)
design_flow_df = design_flow_recommendation(results_df, effective_area_km2)
rain_crosscheck_df = rainfall_crosscheck_summary(p10d_calc if 'p10d_calc' in locals() else np.nan, pmax24_freq_df)
risk_df = risk_table(periods, design_life_years if 'design_life_years' in locals() else 50.0)
try:
    T_requerido_riesgo = return_period_for_risk(target_risk, design_life_years)
except Exception:
    T_requerido_riesgo = np.nan

hydraulic_df = pd.DataFrame()
if 'hydraulic_enabled' in locals() and hydraulic_enabled:
    source_flows_for_hyd = design_flow_df if design_flow_df is not None and not design_flow_df.empty else results_df.dropna(subset=["Q_m3s"])
    slope_hyd_pct = prof_summary.get("S_pct", np.nan)
    if pd.isna(slope_hyd_pct) or float(slope_hyd_pct) <= 0:
        slope_hyd_pct = hydraulic_slope_pct_manual
    hydraulic_df = hydraulic_verification_table(
        source_flows_for_hyd,
        b_m=channel_b_m,
        z_hv=channel_z_hv,
        n=manning_n,
        slope_pct=slope_hyd_pct,
        freeboard_m=freeboard_m,
        design_depth_m=(design_depth_m if 'design_depth_m' in locals() and design_depth_m > 0 else None),
    )

# -----------------------
# Módulo detrítico / sedimentológico
# -----------------------
st.subheader("Módulo de caudal detrítico, gasto sólido y gasto líquido")
with st.expander("Parámetros sedimentológicos y aluvionales", expanded=False):
    st.caption("El módulo transforma el caudal líquido hidrológico en caudal sólido y caudal total detrítico mediante concentración volumétrica Cv, y agrega contrastes por volumen sedimentario disponible.")
    debris_enabled = st.checkbox("Activar cálculo detrítico/sedimentológico", value=True)
    terrain_presets = {
        "Terreno tipo desfavorable (recomendado para revisión conservadora)": {"cv_low": 0.35, "cv_mid": 0.45, "cv_high": 0.55, "material": "Alto", "vegetation": "Baja / escasa", "rho": 1.35, "sigma": 2.65, "phi": 30.0, "prod_area": 20000.0, "prod_cauce": 30000.0},
        "Terreno semiárido medio": {"cv_low": 0.20, "cv_mid": 0.35, "cv_high": 0.45, "material": "Medio", "vegetation": "Media", "rho": 1.20, "sigma": 2.65, "phi": 32.0, "prod_area": 8000.0, "prod_cauce": 15000.0},
        "Terreno natural bajo aporte sólido": {"cv_low": 0.10, "cv_mid": 0.20, "cv_high": 0.30, "material": "Bajo", "vegetation": "Alta", "rho": 1.10, "sigma": 2.65, "phi": 35.0, "prod_area": 3000.0, "prod_cauce": 5000.0},
        "Manual": {"cv_low": 0.20, "cv_mid": 0.35, "cv_high": 0.50, "material": "Medio", "vegetation": "Media", "rho": 1.35, "sigma": 2.65, "phi": 30.0, "prod_area": 8000.0, "prod_cauce": 15000.0},
    }
    terrain_type = st.selectbox("Terreno tipo / condición sedimentológica", list(terrain_presets.keys()), index=0)
    sed_default = terrain_presets[terrain_type]
    st.caption("El preset desfavorable asume alta disponibilidad de material, baja cobertura vegetal y concentraciones volumétricas conservadoras. Todos los parámetros pueden modificarse manualmente abajo.")
    col_cv1, col_cv2, col_cv3 = st.columns(3)
    with col_cv1:
        cv_low = st.number_input("Cv bajo / avenida con sedimento", min_value=0.01, max_value=0.79, value=float(sed_default["cv_low"]), step=0.01, format="%.2f")
        material_available = st.selectbox("Disponibilidad de material", ["Bajo", "Medio", "Alto"], index=["Bajo", "Medio", "Alto"].index(sed_default["material"]))
    with col_cv2:
        cv_mid = st.number_input("Cv medio / hiperconcentrado", min_value=0.01, max_value=0.79, value=float(sed_default["cv_mid"]), step=0.01, format="%.2f")
        vegetation_cover = st.selectbox("Cobertura vegetal", ["Baja / escasa", "Media", "Alta"], index=["Baja / escasa", "Media", "Alta"].index(sed_default["vegetation"]))
    with col_cv3:
        cv_high = st.number_input("Cv alto / flujo de barro", min_value=0.01, max_value=0.79, value=float(sed_default["cv_high"]), step=0.01, format="%.2f")
        use_takahashi = st.checkbox("Agregar Cv Takahashi desde pendiente", value=True)

    st.markdown("**Parámetros Takahashi**")
    tcol1, tcol2, tcol3 = st.columns(3)
    with tcol1:
        rho_fluid = st.number_input("ρ fluido / mezcla fina (g/cm³)", min_value=0.80, max_value=2.30, value=float(sed_default["rho"]), step=0.05)
    with tcol2:
        sigma_solid = st.number_input("σ partículas sólidas (g/cm³)", min_value=1.50, max_value=3.50, value=float(sed_default["sigma"]), step=0.05)
    with tcol3:
        phi_deg = st.number_input("φ fricción interna sedimento (°)", min_value=20.0, max_value=45.0, value=float(sed_default["phi"]), step=1.0)

    slope_for_taka = prof_summary.get("S_pct", np.nan)
    cv_taka = takahashi_cv(slope_for_taka, rho_fluid=rho_fluid, sigma_solid=sigma_solid, phi_deg=phi_deg)
    st.caption(f"Pendiente usada para Takahashi: {slope_for_taka:.2f} %" if pd.notna(slope_for_taka) else "Pendiente no disponible para Takahashi.")

    st.markdown("**Volumen sedimentario disponible / productividad**")
    vcol1, vcol2, vcol3 = st.columns(3)
    with vcol1:
        prod_area = st.number_input("Productividad por área (m³/km²)", min_value=0.0, value=float(sed_default.get("prod_area", 8000.0)), step=1000.0)
    with vcol2:
        prod_cauce = st.number_input("Productividad por cauce (m³/km)", min_value=0.0, value=float(sed_default.get("prod_cauce", 15000.0)), step=1000.0)
    with vcol3:
        manual_sed_vol = st.number_input("Volumen sedimento manual M (m³)", min_value=0.0, value=0.0, step=1000.0)

cv_scenarios = {"Bajo": cv_low, "Medio": cv_mid, "Alto": cv_high}
if 'cv_taka' in locals() and pd.notna(cv_taka) and use_takahashi:
    cv_scenarios["Takahashi"] = float(cv_taka)

if 'debris_enabled' not in locals():
    debris_enabled = False

if debris_enabled:
    susceptibility = debris_susceptibility(metrics, contour_metrics, prof_summary, material_available if 'material_available' in locals() else "Medio", vegetation_cover if 'vegetation_cover' in locals() else "Media")
    debris_df = debris_discharge_table(results_df, cv_scenarios, only_methods=None)
    volume_df = sediment_volume_estimates(metrics.area_km2, prof_summary.get("L_km", np.nan), prod_area if 'prod_area' in locals() else 0.0, prod_cauce if 'prod_cauce' in locals() else 0.0, manual_sed_vol if 'manual_sed_vol' in locals() else 0.0)
    empirical_peak_df = empirical_debris_peak_from_volume(volume_df)
else:
    susceptibility = {"nivel": "No evaluado", "puntaje": 0, "fundamentos": "Módulo desactivado"}
    debris_df = pd.DataFrame()
    volume_df = pd.DataFrame()
    empirical_peak_df = pd.DataFrame()

# -----------------------
# Resultados
# -----------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Área cuenca", f"{metrics.area_km2:.3f} km²")
col1.metric("Perímetro", f"{metrics.perimeter_km:.3f} km")
col2.metric("Curvas detectadas", f"{contour_metrics.n_contours}")
col2.metric("Rango altimétrico", f"{contour_metrics.z_min}–{contour_metrics.z_max} m" if contour_metrics.z_min is not None else "—")
col3.metric("Longitud hidráulica", f"{prof_summary['L_km']:.3f} km")
col3.metric("Pendiente media", f"{prof_summary['S_pct']:.2f} %" if pd.notna(prof_summary['S_pct']) else "—")
col4.metric("Tc adoptado", f"{prof_summary.get('Tc_adoptado_min', np.nan):.1f} min" if pd.notna(prof_summary.get('Tc_adoptado_min', np.nan)) else "—")
col4.metric("Cuenca sugerida", region_guess.get("cuenca_sugerida", "Manual"))

with st.expander("Diagnóstico automático de inputs", expanded=True):
    st.dataframe(diagnostics_df, use_container_width=True)
    if auto_rain_df is not None and not auto_rain_df.empty:
        st.markdown("**Lluvia generada automáticamente desde referencias DGA/Manual de Carreteras**")
        st.dataframe(auto_rain_df, use_container_width=True)
        st.caption(f"Duración IDF usada: {duration_for_idf_h:.3f} h. Fuente: {idf_station}. Revise/valide P10D, isoyeta detectada, CD/CF, K y CA antes de diseño definitivo.")

with st.expander("Tiempo de concentración avanzado y pendiente Mociornita", expanded=True):
    st.markdown("**Pendiente media de cuenca según Mociornita**")
    st.write(mociornita_info)
    st.caption("Mociornita usa la equidistancia de curvas, área de cuenca y longitud de curvas dentro del polígono. Es sensible a la calidad del KMZ de curvas.")
    st.markdown("**Comparación de tiempos de concentración**")
    st.dataframe(tc_methods_df, use_container_width=True)
    st.caption("La app usa la mediana de métodos válidos como Tc adoptado automático para generar IDF; el especialista puede revisar y corregir manualmente la duración IDF.")

with st.expander("Mapa de isoyetas KMZ", expanded=bool(isohyets)):
    if isohyets:
        iso_rows = []
        for iso in isohyets:
            iso_rows.append({"nombre": iso.name, "P_mm": iso.properties.get("p_mm"), "vertices": int(len(list(iso.geometry.coords))) if getattr(iso.geometry, "geom_type", "") == "LineString" else np.nan})
        st.dataframe(pd.DataFrame(iso_rows), use_container_width=True)
        st.write({k: v for k, v in nearest_iso.items() if k != "feature"})
        st.caption("La precipitación P10D se adopta desde la isoyeta más cercana al punto de control cuando la opción está activada. Si el punto queda entre dos curvas, el valor debe revisarse o reemplazarse por interpolación técnica/manual.")
    else:
        st.info("No se cargó mapa KMZ/KML de isoyetas.")

with st.expander("Base histórica regional DGA incorporada", expanded=False):
    if 'use_historical_data' in locals() and use_historical_data and HISTORICAL_DATA_DIR.exists():
        st.markdown("**Archivos históricos cargados en la aplicación**")
        st.dataframe(historical_files_df, use_container_width=True)
        if not nearest_historical_df.empty:
            st.markdown("**Estaciones históricas más cercanas al punto de control**")
            st.dataframe(nearest_historical_df, use_container_width=True)
        else:
            st.info("No se encontraron estaciones en el radio definido o la base histórica no pudo construir catálogo.")

        st.markdown("**Frecuencia preliminar de precipitación máxima anual 24 h**")
        if not pmax24_freq_df.empty:
            st.caption("Estimación preliminar por distribución Gumbel sobre máximos anuales 24 h. Debe revisarse largo de registro, representatividad, calidad de datos e independencia de la serie.")
            st.dataframe(pmax24_freq_df, use_container_width=True)
            if 'rain_crosscheck_df' in locals() and rain_crosscheck_df is not None and not rain_crosscheck_df.empty:
                st.markdown("**Comparación P10D de isoyeta versus estación histórica cercana**")
                st.dataframe(rain_crosscheck_df, use_container_width=True)
            try:
                first_code = str(pmax24_freq_df.sort_values("distancia_km").iloc[0]["codigo"])
                pchart = pmax24_freq_df[pmax24_freq_df["codigo"].astype(str) == first_code].pivot_table(index="T_anios", values="P24_Gumbel_mm", aggfunc="first")
                st.line_chart(pchart)
            except Exception:
                pass
        else:
            st.info("No hay estimación Pmax24 anual disponible en el radio definido.")

        st.markdown("**Sedimentos históricos cercanos**")
        if not sed_hist_df.empty:
            st.dataframe(sed_hist_df, use_container_width=True)
            st.caption("Estas concentraciones corresponden a muestreos históricos en estaciones DGA; sirven como referencia de disponibilidad/variabilidad sedimentaria, no reemplazan granulometría ni mediciones locales de la quebrada.")
        else:
            st.info("No se encontraron estaciones sedimentológicas cercanas en el radio definido.")

        st.markdown("**Precipitación mensual y temperatura histórica cercana**")
        if not climate_hist_df.empty:
            st.dataframe(climate_hist_df, use_container_width=True)
            st.caption("Estas series ayudan a caracterizar régimen térmico, estacionalidad de precipitaciones, línea de nieve aproximada y condición antecedente de humedad.")
        else:
            st.info("No se encontraron estaciones climáticas cercanas en el radio definido.")
    else:
        st.info("La base histórica regional está desactivada o no está incluida en el repositorio.")

with st.expander("Parámetros morfométricos", expanded=True):
    st.dataframe(pd.DataFrame([metrics.__dict__]).T.rename(columns={0: "valor"}), use_container_width=True)

with st.expander("Curvas de nivel y perfil longitudinal", expanded=True):
    st.dataframe(pd.DataFrame([contour_metrics.__dict__]).T.rename(columns={0: "valor"}), use_container_width=True)
    if profile_df is not None and not profile_df.empty:
        st.line_chart(profile_df.set_index("dist_m")["cota_m"])
        st.dataframe(profile_df, use_container_width=True)
    else:
        st.info("No hay perfil longitudinal. Cargue/dibuje un cauce principal o active la estimación automática.")

with st.expander("Detector automático de metodologías", expanded=True):
    st.dataframe(methods_df, use_container_width=True)
    recommended = methods_df[methods_df["estado"].isin(["Verde", "Amarillo"])]
    if not recommended.empty:
        st.success("Metodologías recomendadas/condicionadas: " + ", ".join(recommended["metodo"].tolist()))

with st.expander("Referencias técnicas incorporadas a la automatización", expanded=False):
    st.dataframe(METHODOLOGY_REFERENCES, use_container_width=True)
    if idf_station != "Manual":
        st.markdown(f"**Coeficientes CD/CF disponibles para: {idf_station}**")
        st.dataframe(rainfall_reference_dataframe(idf_station), use_container_width=True)
    st.info("Los catálogos internos son presets editables. Para diseño definitivo se debe verificar la zona homogénea, P24, coeficientes regionales y pertinencia hidrológica de la estación de referencia.")

with st.expander("Riesgo de falla, área pluvial y línea de nieve", expanded=False):
    st.markdown("**Riesgo de excedencia durante vida útil**")
    if 'risk_df' in locals() and risk_df is not None and not risk_df.empty:
        st.dataframe(risk_df, use_container_width=True)
        if pd.notna(T_requerido_riesgo):
            st.info(f"Para riesgo objetivo {target_risk:.3f} y vida útil {design_life_years:.0f} años, T requerido ≈ {T_requerido_riesgo:.0f} años.")
    st.markdown("**Área pluvial efectiva**")
    st.dataframe(pd.DataFrame([pluvial_area_info]), use_container_width=True)
    st.caption("El área pluvial se usa para DGA-AC/Verni-King/HUS/SCS cuando se activa la opción de línea de nieve. Es una aproximación inicial; para cuencas nivo-pluviales debe validarse con línea de nieve, temperatura, altitud y/o información satelital.")

with st.expander("Verificación hidráulica preliminar por Manning", expanded=False):
    st.info(objective_design_guidance(objective))
    if hydraulic_df is not None and not hydraulic_df.empty:
        st.dataframe(hydraulic_df, use_container_width=True)
        try:
            st.line_chart(hydraulic_df.pivot_table(index="T_anios", values=["y_normal_m", "altura_total_requerida_m", "velocidad_m_s"], aggfunc="max"))
        except Exception:
            pass
        st.warning("La verificación hidráulica es preliminar. Para diseño definitivo debe revisarse control hidráulico, pérdidas singulares, transiciones, curvas, material de revestimiento, socavación, arrastre sólido y eventual modelación HEC-RAS/2D si corresponde.")
    else:
        st.info("No hay caudales o parámetros suficientes para verificar hidráulicamente la sección.")

with st.expander("Resultados por metodología", expanded=True):
    st.dataframe(results_df, use_container_width=True)
    calc = results_df.dropna(subset=["Q_m3s"])
    if not calc.empty:
        pivot = calc.pivot_table(index="T_anios", columns="metodo", values="Q_m3s", aggfunc="first")
        st.line_chart(pivot)
    else:
        st.warning("Aún no hay caudales calculados. Ingrese intensidades IDF, P24 o parámetros regionales según corresponda.")

with st.expander("Caudal recomendado automático y criterio de adopción", expanded=True):
    if design_flow_df is not None and not design_flow_df.empty:
        st.dataframe(design_flow_df, use_container_width=True)
        try:
            st.line_chart(design_flow_df.set_index("T_anios")[["Q_recomendado_m3s", "Q_mediana_m3s", "Q_max_m3s"]])
        except Exception:
            pass
        st.warning("El caudal recomendado es una ayuda de revisión; el caudal adoptado definitivo debe quedar justificado por el especialista.")
    else:
        st.info("No hay caudal recomendado porque aún no existen resultados calculados.")

with st.expander("Resultados del módulo detrítico/sedimentológico", expanded=debris_enabled):
    if debris_enabled:
        st.markdown(f"**Susceptibilidad aluvional/detrítica preliminar:** {susceptibility.get('nivel')} | puntaje {susceptibility.get('puntaje')}")
        st.caption(susceptibility.get("fundamentos", ""))
        st.markdown("**Clasificación por concentración volumétrica Cv**")
        st.dataframe(debris_cv_classes_dataframe(), use_container_width=True)
        if 'cv_taka' in locals() and pd.notna(cv_taka):
            st.info(f"Cv Takahashi estimado desde pendiente = {cv_taka:.3f} ({cv_taka*100:.1f} %).")
        st.markdown("**Conversión Q líquido → Q sólido → Q detrítico**")
        if not debris_df.empty:
            st.dataframe(debris_df, use_container_width=True)
            qd_piv = debris_df.pivot_table(index="T_anios", columns="escenario_Cv", values="QD_m3s", aggfunc="max")
            st.line_chart(qd_piv)
        else:
            st.warning("No hay caudales detríticos calculados porque faltan caudales líquidos o Cv válidos.")
        st.markdown("**Volumen sedimentario y relaciones empíricas Qp=f(M)**")
        if not volume_df.empty:
            st.dataframe(volume_df, use_container_width=True)
            st.dataframe(empirical_peak_df, use_container_width=True)
        else:
            st.info("Ingrese productividad sedimentaria o volumen manual para calcular relaciones empíricas Qp=f(M).")
        st.warning("El módulo detrítico entrega valores preliminares de prediseño/contraste. Debe validarse con granulometría, disponibilidad real de sedimentos, secciones, evidencias de terreno y/o modelación especializada.")
    else:
        st.info("Módulo detrítico desactivado.")

# -----------------------
# Memoria y exportaciones
# -----------------------
st.subheader("Memoria técnica")
notes = st.text_area("Notas del especialista para incorporar en la memoria", "")
if st.button("Generar memoria Word"):
    with st.spinner("Generando memoria técnica..."):
        with tempfile.TemporaryDirectory() as td:
            basin_img = os.path.join(td, "cuenca.png")
            create_basin_plot(basin_f, contours, channel_geom, outlet_geom, metrics.epsg_utm, basin_img)
            doc_bytes = generate_memoria_bytes(
                project_name=project_name,
                objective=objective,
                metrics=metrics.__dict__,
                contour_metrics=contour_metrics.__dict__,
                profile_summary=prof_summary,
                methods_df=methods_df,
                results_df=results_df,
                profile_df=profile_df,
                debris_df=debris_df,
                volume_df=volume_df,
                empirical_peak_df=empirical_peak_df,
                susceptibility=susceptibility,
                tc_methods_df=tc_methods_df,
                design_flow_df=design_flow_df,
                pmax24_freq_df=pmax24_freq_df,
                rain_crosscheck_df=rain_crosscheck_df,
                basin_image_path=basin_img,
                surface_image_path=None,
                notes=(
                    notes
                    + "\n\nDiagnóstico de inputs:\n" + diagnostics_df.to_string(index=False)
                    + ("\n\nLluvia generada automáticamente:\n" + auto_rain_df.to_string(index=False) if auto_rain_df is not None and not auto_rain_df.empty else "")
                    + "\n\nReferencias metodológicas incorporadas:\n" + METHODOLOGY_REFERENCES.to_string(index=False)
                ),
            )
        st.download_button(
            "Descargar memoria de cálculo .docx",
            data=doc_bytes,
            file_name=f"Memoria_Calculo_{project_name.replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

st.download_button("Descargar resultados CSV", data=results_df.to_csv(index=False).encode("utf-8"), file_name="resultados_hidrologicos.csv", mime="text/csv")
st.download_button("Descargar diagnóstico CSV", data=diagnostics_df.to_csv(index=False).encode("utf-8"), file_name="diagnostico_inputs.csv", mime="text/csv")
if debris_enabled and debris_df is not None and not debris_df.empty:
    st.download_button("Descargar resultados detríticos CSV", data=debris_df.to_csv(index=False).encode("utf-8"), file_name="resultados_detriticos.csv", mime="text/csv")
if debris_enabled and empirical_peak_df is not None and not empirical_peak_df.empty:
    st.download_button("Descargar relaciones Qp-M CSV", data=empirical_peak_df.to_csv(index=False).encode("utf-8"), file_name="relaciones_empiricas_Qp_M.csv", mime="text/csv")
if 'nearest_historical_df' in locals() and nearest_historical_df is not None and not nearest_historical_df.empty:
    st.download_button("Descargar estaciones históricas cercanas CSV", data=nearest_historical_df.to_csv(index=False).encode("utf-8"), file_name="estaciones_historicas_cercanas.csv", mime="text/csv")
if 'pmax24_freq_df' in locals() and pmax24_freq_df is not None and not pmax24_freq_df.empty:
    st.download_button("Descargar frecuencia Pmax24 histórica CSV", data=pmax24_freq_df.to_csv(index=False).encode("utf-8"), file_name="frecuencia_pmax24_historica.csv", mime="text/csv")

if 'tc_methods_df' in locals() and tc_methods_df is not None and not tc_methods_df.empty:
    st.download_button("Descargar tiempos de concentración CSV", data=tc_methods_df.to_csv(index=False).encode("utf-8"), file_name="tiempos_concentracion.csv", mime="text/csv")
if 'design_flow_df' in locals() and design_flow_df is not None and not design_flow_df.empty:
    st.download_button("Descargar caudal recomendado CSV", data=design_flow_df.to_csv(index=False).encode("utf-8"), file_name="caudal_recomendado.csv", mime="text/csv")
if 'rain_crosscheck_df' in locals() and rain_crosscheck_df is not None and not rain_crosscheck_df.empty:
    st.download_button("Descargar comparación lluvia histórica vs isoyeta CSV", data=rain_crosscheck_df.to_csv(index=False).encode("utf-8"), file_name="comparacion_lluvia_historica_isoyeta.csv", mime="text/csv")
if 'risk_df' in locals() and risk_df is not None and not risk_df.empty:
    st.download_button("Descargar riesgo de falla CSV", data=risk_df.to_csv(index=False).encode("utf-8"), file_name="riesgo_falla.csv", mime="text/csv")
if 'hydraulic_df' in locals() and hydraulic_df is not None and not hydraulic_df.empty:
    st.download_button("Descargar verificación hidráulica CSV", data=hydraulic_df.to_csv(index=False).encode("utf-8"), file_name="verificacion_hidraulica_manning.csv", mime="text/csv")
