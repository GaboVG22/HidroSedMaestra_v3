from __future__ import annotations

from pathlib import Path
import json

import pandas as pd
import streamlit as st

from modules_hidrosed.contour_generator import generate_contours_kmz
from modules_hidrosed.dem_downloader import (
    bbox_from_point,
    bbox_from_point_km,
    download_dem_cop30,
    download_dem_cop30_tiled,
    validate_geotiff,
)
from modules_hidrosed.hidrosed_core import (
    GlobalParams,
    SectionInput,
    apply_grain_stats_to_sections,
    compute_all,
    dataframe_to_sections,
    default_sections,
    exner_simulation,
    grain_stats,
    parse_grain_curve,
    sections_to_dataframe,
    technical_summary,
)
from modules_hidrosed.hidrosed_report import export_hidrosed_excel, export_hidrosed_html
from modules_hidrosed.kmz_reader import KMLParseError, read_control_point
from modules_hidrosed.modulo_integracion_secciones_hidrosed import (
    convertir_puntos_a_hidrosed,
    convertir_secciones_a_hidrosed,
    generar_datos_demo_v13,
    generar_excel_integracion_hidrosed,
    hidrosed_03_to_sections_df,
    leer_salidas_v13_desde_excel,
    transferir_a_session_state,
    validar_secciones_para_hidrosed,
)
from modules_hidrosed.session_utils import as_download_bytes, get_session_folder, save_state_json, status_badge
from modules_hidrosed.visualizacion_3d import (
    exportar_modelo_3d_html,
    generar_figura_3d_cauce_secciones,
)


from modules_hidrosed.kmz_geometria import (
    extract_kml,
    generate_preliminary_axis_from_point,
    merge_kml_documents,
    parse_kml_geometries,
    summarize_geometries,
    write_kmz_from_kml,
)
from modules_hidrosed.granulometria_avanzada import (
    assign_grain_to_sections,
    build_default_grain_session_payload,
    get_standard_grain_profile,
    get_standard_grain_stats,
    normalize_grain_table,
    parse_grain_text_multi,
    recommend_standard_profile_by_context,
    standard_profile_options,
    STANDARD_GRAIN_PROFILES,
    DEFAULT_STANDARD_PROFILE_KEY,
    stats_by_sample,
)
from modules_hidrosed.idf_generator import (
    export_idf_excel,
    generate_idf_from_p24,
    generate_idf_power_law,
    parse_p24_text,
)

from modules_hidrosed.modelacion_avanzada import (
    AdvancedModelParams,
    ConnectedProfileParams,
    build_audit_summary,
    calibrate_manning_multiplier,
    compute_connected_profile_v6,
    compute_irregular_hydraulics,
    confidence_score_v6,
    monte_carlo_uncertainty_v6,
    roughness_estimators,
    sensitivity_manning_irregular,
    water_density_kgm3,
)

# st.set_page_config se controla desde app.py maestro

DEFAULT_GRAIN = """0.25;2
0.50;5
2;12
8;32
16;50
32;68
64;84
128;93
200;100"""


def init_state() -> None:
    if "sections_df" not in st.session_state:
        st.session_state["sections_df"] = sections_to_dataframe(default_sections(n_sections=5))
    if "global_params" not in st.session_state:
        st.session_state["global_params"] = GlobalParams().__dict__
    if "grain_text" not in st.session_state:
        st.session_state["grain_text"] = DEFAULT_GRAIN


def get_opentopo_key() -> str:
    try:
        secret_key = st.secrets.get("OPENTOPO_API_KEY", "")
    except Exception:
        secret_key = ""
    return st.sidebar.text_input(
        "OpenTopography API Key",
        value=secret_key,
        type="password",
        help="En Streamlit Cloud puedes guardarla en Secrets como OPENTOPO_API_KEY.",
    ).strip()


def sidebar() -> str:
    st.sidebar.title("HidroSed Maestra Integrada v6.3")
    page = st.sidebar.radio(
        "Flujo técnico",
        [
            "1 · DEM COP30",
            "2 · Curvas de nivel y apoyo topográfico",
            "3 · Eje del cauce",
            "4 · Integración secciones v13",
            "5 · Modelo 3D cauce y secciones",
            "6 · Proyecto y secciones",
            "7 · Granulometría avanzada",
            "8 · Curvas IDF",
            "9 · Resultados HidroSed",
            "10 · Lecho móvil",
            "11 · Reporte y descargas",
        ],
    )
    st.sidebar.divider()
    st.sidebar.subheader("Estado")
    status_badge("dem_path", "DEM disponible")
    status_badge("curvas_kmz_path", "Curvas DEM disponibles")
    status_badge("curvas_apoyo_kmz_path", "Curvas apoyo topográfico")
    status_badge("eje_cauce_kmz_path", "Eje cauce")
    status_badge("sections_df", "Secciones cargadas")
    status_badge("secciones_transferidas", "Secciones v13 transferidas")
    status_badge("grain_stats", "Granulometría aplicada")
    status_badge("idf_df", "Curvas IDF")
    status_badge("hidrosed_results", "Resultados calculados")
    st.sidebar.caption("Main file path: app.py")
    return page


def page_dem(api_key: str) -> None:
    st.title("1 · DEM COP30 como etapa inicial de HidroSed Cauces")
    st.write(
        "Carga un KMZ/KML con el punto de control. La app descarga el DEM COP30 y lo deja en memoria de sesión "
        "para generar curvas de nivel sin volver a cargar archivos."
    )
    c1, c2 = st.columns([1.1, 0.9])
    with c1:
        uploaded = st.file_uploader("KMZ/KML con punto de control", type=["kmz", "kml"])
        margin_mode = st.radio("Tipo de margen", ["Grados", "Kilómetros"], horizontal=True)
        if margin_mode == "Grados":
            margin = st.select_slider("Margen de descarga [°]", options=[0.10, 0.25, 0.50, 1.00, 1.50, 2.00], value=0.50)
        else:
            margin = st.select_slider("Margen de descarga [km]", options=[5, 10, 25, 50, 100, 150, 200], value=50)
        mode = st.radio("Modo DEM", ["Descarga única", "Mosaico para cuencas grandes"], horizontal=True)
        n_tiles = 10
        if mode.startswith("Mosaico"):
            n_tiles = st.slider("Número de DEM parciales", 10, 40, 10)
    with c2:
        st.info(
            "Para cuencas grandes usa mosaico de 10 a 40 DEM parciales. Para evitar caídas, genera curvas cada 50–100 m "
            "y usa simplificación ≥60 m."
        )
        if st.session_state.get("control_point"):
            st.success("Punto de control detectado")
            st.json(st.session_state["control_point"])


    st.divider()
    st.subheader("Entrada opcional: eje del cauce")
    st.caption("El input mínimo es el KMZ/KML del punto de control. El eje del cauce puede cargarse ahora o generarse de forma preliminar desde la aplicación para revisión posterior.")
    eje_file = st.file_uploader("Opcional: KMZ/KML con eje del cauce validado", type=["kmz", "kml"], key="eje_cauce_upload_global")
    if eje_file is not None and st.button("Registrar eje del cauce cargado", type="secondary"):
        try:
            folder = get_session_folder()
            eje_path = folder / f"eje_cauce_{eje_file.name}"
            eje_path.write_bytes(eje_file.getvalue())
            geoms = parse_kml_geometries(eje_file, filename=eje_file.name)
            resumen = summarize_geometries(geoms, archivo=eje_file.name)
            st.session_state["eje_cauce_kmz_path"] = str(eje_path)
            st.session_state["eje_cauce_origen"] = "KMZ/KML cargado por usuario"
            st.session_state["eje_cauce_resumen"] = resumen.__dict__
            st.success("Eje del cauce registrado. Será usado con prioridad para generación/revisión de secciones.")
            st.json(resumen.__dict__)
        except Exception as exc:
            st.error(f"No se pudo leer el eje del cauce: {exc}")

    if uploaded and st.button("Leer punto de control", type="secondary"):
        try:
            cp = read_control_point(uploaded, filename=uploaded.name)
            st.session_state["control_point"] = {
                "nombre": cp.name,
                "latitud": cp.latitude,
                "longitud": cp.longitude,
                "altitud": cp.altitude,
                "archivo": uploaded.name,
            }
            st.success(f"Punto detectado: {cp.name} · lat {cp.latitude:.6f}, lon {cp.longitude:.6f}")
        except KMLParseError as exc:
            st.error(str(exc))

    cp = st.session_state.get("control_point")
    if cp:
        bbox = bbox_from_point(cp["latitud"], cp["longitud"], float(margin)) if margin_mode == "Grados" else bbox_from_point_km(cp["latitud"], cp["longitud"], float(margin))
        st.subheader("Bounding box")
        st.json(bbox.to_params())
        if st.button("Descargar DEM COP30", type="primary"):
            if not api_key:
                st.error("Falta la API Key de OpenTopography.")
                return
            folder = get_session_folder()
            dem_path = folder / "dem_cop30.tif"
            progress = st.progress(0)
            status = st.empty()
            try:
                if mode.startswith("Descarga"):
                    status.info("Descargando DEM desde OpenTopography...")
                    download_dem_cop30(bbox, api_key=api_key, output_path=dem_path)
                    progress.progress(100)
                else:
                    def cb(done, total, msg):
                        progress.progress(int((done / max(total, 1)) * 100))
                        status.info(msg)
                    download_dem_cop30_tiled(bbox, api_key=api_key, output_path=dem_path, tiles_dir=folder / "dem_tiles", n_tiles=n_tiles, progress_callback=cb)
                    progress.progress(100)
                meta = validate_geotiff(dem_path)
                st.session_state["dem_path"] = str(dem_path)
                st.session_state["bbox"] = bbox.to_params()
                st.session_state["dem_metadata"] = meta
                save_state_json({"control_point": cp, "bbox": bbox.to_params(), "dem": meta}, filename="estado_dem.json")
                st.success("DEM descargado y registrado internamente para el módulo de curvas.")
                st.json(meta)
            except Exception as exc:
                st.error(f"No se pudo descargar o validar el DEM: {exc}")


    if cp and not st.session_state.get("eje_cauce_kmz_path"):
        with st.expander("Generar eje preliminar automático si no existe eje cargado", expanded=False):
            st.warning("Este eje es preliminar y debe revisarse. Para cálculo definitivo se recomienda eje de terreno o eje editado en la app de secciones.")
            ax_len = st.number_input("Longitud preliminar del eje [km]", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
            ax_az = st.number_input("Azimut preliminar [°]", min_value=0.0, max_value=360.0, value=180.0, step=5.0)
            if st.button("Crear eje preliminar automático", type="secondary"):
                folder = get_session_folder()
                kml = generate_preliminary_axis_from_point(cp, st.session_state.get("bbox"), length_km=float(ax_len), azimuth_deg=float(ax_az))
                axis_path = folder / "eje_preliminar_automatico.kmz"
                write_kmz_from_kml(kml, axis_path)
                st.session_state["eje_cauce_kmz_path"] = str(axis_path)
                st.session_state["eje_cauce_origen"] = "Eje preliminar automático"
                st.session_state["eje_cauce_resumen"] = {"tipo": "preliminar", "longitud_km": float(ax_len), "azimut_deg": float(ax_az)}
                st.success("Eje preliminar creado y guardado en memoria. Revísalo antes de usarlo como geometría definitiva.")

    dem_path = st.session_state.get("dem_path")
    if dem_path and Path(dem_path).exists():
        st.download_button("Descargar DEM GeoTIFF", data=as_download_bytes(dem_path), file_name="dem_cop30.tif", mime="image/tiff")


def page_contours() -> None:
    st.title("2 · Curvas de nivel desde el DEM")
    dem_path = st.session_state.get("dem_path")
    if not dem_path or not Path(dem_path).exists():
        st.warning("Primero descarga el DEM en la etapa 1.")
        return
    st.success("DEM disponible internamente. No necesitas cargarlo otra vez.")
    st.code(dem_path)
    c1, c2 = st.columns([1, 1])
    with c1:
        interval = st.selectbox("Equidistancia [m]", [10, 25, 50, 100, 200], index=2)
        simplify = st.selectbox("Simplificación [m]", [0, 30, 60, 100, 150, 250], index=2)
        max_cells = st.selectbox("Máximo de celdas procesadas", [1_000_000, 2_000_000, 4_000_000, 8_000_000], index=2, format_func=lambda x: f"{x:,}".replace(",", "."))
    with c2:
        st.info("La app decima internamente DEM grandes para generar curvas estables en Streamlit Cloud. También puedes cargar curvas topográficas de apoyo para mejorar la geometría de secciones y la revisión del cauce.")

    st.divider()
    st.subheader("Curvas de nivel de apoyo topográfico")
    st.caption("Opcional: carga KMZ/KML con curvas levantadas por topografía. Se guardan con prioridad técnica sobre curvas DEM para revisión y se pueden combinar en un KMZ maestro.")
    apoyo_file = st.file_uploader("Opcional: KMZ/KML con curvas de nivel topográficas de apoyo", type=["kmz", "kml"], key="curvas_apoyo_upload")
    prioridad = st.radio("Prioridad de curvas para secciones", ["DEM + apoyo topográfico", "Solo apoyo topográfico", "Solo DEM"], horizontal=True, key="prioridad_curvas")
    st.session_state["prioridad_curvas"] = prioridad
    if apoyo_file is not None and st.button("Registrar curvas topográficas de apoyo", type="secondary"):
        try:
            folder = get_session_folder()
            apoyo_path = folder / f"curvas_apoyo_topografia_{apoyo_file.name}"
            apoyo_path.write_bytes(apoyo_file.getvalue())
            geoms = parse_kml_geometries(apoyo_file, filename=apoyo_file.name)
            resumen = summarize_geometries(geoms, archivo=apoyo_file.name)
            st.session_state["curvas_apoyo_kmz_path"] = str(apoyo_path)
            st.session_state["curvas_apoyo_resumen"] = resumen.__dict__
            st.session_state["curvas_apoyo_lineas"] = geoms.get("lineas", pd.DataFrame()).to_dict("records")
            st.success("Curvas topográficas de apoyo registradas.")
            st.json(resumen.__dict__)
        except Exception as exc:
            st.error(f"No se pudieron leer las curvas de apoyo: {exc}")

    if st.session_state.get("curvas_apoyo_resumen"):
        with st.expander("Resumen de curvas topográficas de apoyo", expanded=False):
            st.json(st.session_state["curvas_apoyo_resumen"])
            if st.session_state.get("curvas_apoyo_lineas"):
                st.dataframe(pd.DataFrame(st.session_state["curvas_apoyo_lineas"]).head(200), use_container_width=True)

    if st.session_state.get("curvas_kml_path") and st.session_state.get("curvas_apoyo_kmz_path"):
        if st.button("Crear KMZ maestro DEM + curvas topográficas", type="secondary"):
            try:
                folder = get_session_folder()
                dem_kml = extract_kml(st.session_state["curvas_kml_path"])
                apoyo_kml = extract_kml(st.session_state["curvas_apoyo_kmz_path"])
                out = folder / "curvas_maestras_dem_mas_topografia.kmz"
                merge_kml_documents([("Curvas DEM", dem_kml), ("Curvas topográficas de apoyo", apoyo_kml)], out, "Curvas maestras DEM + apoyo topográfico")
                st.session_state["curvas_maestras_kmz_path"] = str(out)
                st.success("KMZ maestro de curvas creado.")
            except Exception as exc:
                st.error(f"No se pudo combinar curvas: {exc}")

    if st.button("Generar curvas de nivel", type="primary"):
        folder = get_session_folder()
        progress = st.progress(0)
        status = st.empty()
        def cb(done, total, msg):
            progress.progress(int((done / max(total, 1)) * 100))
            status.info(msg)
        try:
            result = generate_contours_kmz(Path(dem_path), output_dir=folder, interval=float(interval), simplify_m=float(simplify), max_cells=int(max_cells), progress_callback=cb)
            progress.progress(100)
            st.session_state["curvas_kmz_path"] = str(result.kmz_path)
            st.session_state["curvas_kml_path"] = str(result.kml_path)
            st.session_state["curvas_metadata"] = {
                "n_lineas": result.n_lines,
                "z_min_m": result.min_elevation,
                "z_max_m": result.max_elevation,
                "equidistancia_m": result.interval,
            }
            st.success(f"Curvas generadas: {result.n_lines:,} líneas".replace(",", "."))
            st.json(st.session_state["curvas_metadata"])
            if result.preview_png and result.preview_png.exists():
                st.image(str(result.preview_png), caption="Vista preliminar")
        except Exception as exc:
            st.error(f"No se pudieron generar curvas: {exc}")

    curvas_path = st.session_state.get("curvas_kmz_path")
    if curvas_path and Path(curvas_path).exists():
        st.download_button("Descargar curvas KMZ", data=as_download_bytes(curvas_path), file_name="curvas_nivel.kmz", mime="application/vnd.google-earth.kmz")
    cm = st.session_state.get("curvas_maestras_kmz_path")
    if cm and Path(cm).exists():
        st.download_button("Descargar KMZ maestro DEM + topografía", data=as_download_bytes(cm), file_name="curvas_maestras_dem_topografia.kmz", mime="application/vnd.google-earth.kmz")



def _load_v13_sources_from_state() -> dict[str, pd.DataFrame]:
    """Obtiene salidas v13 desde session_state si la app generadora está integrada."""
    return {
        "df_secciones_validas": pd.DataFrame(st.session_state.get("df_secciones_validas", [])),
        "df_puntos_secciones": pd.DataFrame(st.session_state.get("df_puntos_secciones", [])),
        "df_secciones_descartadas": pd.DataFrame(st.session_state.get("df_secciones_descartadas", [])),
        "df_perfil_longitudinal": pd.DataFrame(st.session_state.get("df_perfil_longitudinal", [])),
    }


def page_eje_cauce() -> None:
    st.title("3 · Eje del cauce")
    st.markdown(
        """
        El eje del cauce puede provenir de tres fuentes: **KMZ/KML cargado**, **eje preliminar generado en la app** o
        **salidas de la aplicación de secciones**. El eje cargado por topografía o edición GIS tiene prioridad para la generación de secciones.
        """
    )
    eje_path = st.session_state.get("eje_cauce_kmz_path")
    c1, c2 = st.columns([1, 1])
    with c1:
        uploaded = st.file_uploader("Cargar o reemplazar KMZ/KML del eje del cauce", type=["kmz", "kml"], key="eje_cauce_upload_page")
        if uploaded is not None and st.button("Usar este eje del cauce", type="primary"):
            try:
                folder = get_session_folder()
                path = folder / f"eje_cauce_{uploaded.name}"
                path.write_bytes(uploaded.getvalue())
                geoms = parse_kml_geometries(uploaded, filename=uploaded.name)
                resumen = summarize_geometries(geoms, archivo=uploaded.name)
                st.session_state["eje_cauce_kmz_path"] = str(path)
                st.session_state["eje_cauce_origen"] = "KMZ/KML cargado por usuario"
                st.session_state["eje_cauce_resumen"] = resumen.__dict__
                st.success("Eje cargado y validado.")
            except Exception as exc:
                st.error(f"No se pudo registrar el eje: {exc}")
    with c2:
        st.info("Si no tienes eje, genera uno preliminar desde el punto de control y luego ajústalo en la etapa de secciones. No se recomienda usar un eje preliminar sin revisión para diseño definitivo.")
        cp = st.session_state.get("control_point")
        if cp:
            ax_len = st.number_input("Longitud eje preliminar [km]", 0.5, 100.0, 10.0, 0.5, key="axis_len_page")
            ax_az = st.number_input("Azimut eje preliminar [°]", 0.0, 360.0, 180.0, 5.0, key="axis_az_page")
            if st.button("Generar eje preliminar", type="secondary", key="gen_axis_page"):
                folder = get_session_folder()
                kml = generate_preliminary_axis_from_point(cp, st.session_state.get("bbox"), length_km=float(ax_len), azimuth_deg=float(ax_az))
                path = folder / "eje_preliminar_automatico.kmz"
                write_kmz_from_kml(kml, path)
                st.session_state["eje_cauce_kmz_path"] = str(path)
                st.session_state["eje_cauce_origen"] = "Eje preliminar automático"
                st.session_state["eje_cauce_resumen"] = {"tipo": "preliminar", "longitud_km": float(ax_len), "azimut_deg": float(ax_az)}
                st.success("Eje preliminar generado.")
        else:
            st.warning("Primero lee el punto de control en el módulo DEM.")

    eje_path = st.session_state.get("eje_cauce_kmz_path")
    if eje_path and Path(eje_path).exists():
        st.success(f"Eje activo: {st.session_state.get('eje_cauce_origen', 'Sin origen')}")
        st.json(st.session_state.get("eje_cauce_resumen", {}))
        st.download_button("Descargar eje activo KMZ/KML", data=as_download_bytes(eje_path), file_name=Path(eje_path).name, mime="application/vnd.google-earth.kmz")
    else:
        st.warning("Aún no hay eje de cauce registrado.")


def page_integracion_secciones() -> None:
    st.title("3 · Integración de secciones v13 → HidroSed Maestra")
    st.markdown(
        """
        Este módulo enlaza la aplicación **v13_fix_km_final_utm19s_3d** con HidroSed Maestra.
        Recibe `df_secciones_validas` y `df_puntos_secciones`, valida la geometría, genera las tablas
        **03_Secciones** y **04_Puntos_Seccion**, y las deja disponibles para calcular hidráulica,
        transporte de sedimentos y socavación sin descargar y recargar archivos manualmente.
        """
    )

    fuente = st.radio(
        "Fuente de secciones",
        [
            "Usar salidas v13 ya presentes en memoria",
            "Cargar Excel de respaldo v13 / HidroSed",
            "Usar datos demo para prueba técnica",
        ],
        horizontal=False,
    )

    data = _load_v13_sources_from_state()
    if fuente.startswith("Cargar"):
        uploaded = st.file_uploader(
            "Excel con hojas df_secciones_validas/df_puntos_secciones o 03_Secciones/04_Puntos_Seccion",
            type=["xlsx", "xls"],
            key="excel_v13_upload",
        )
        if uploaded is not None:
            try:
                data = leer_salidas_v13_desde_excel(uploaded)
                st.success("Excel leído correctamente.")
            except Exception as exc:
                st.error(f"No se pudo leer el Excel: {exc}")
                return
    elif fuente.startswith("Usar datos demo"):
        data = generar_datos_demo_v13(n_secciones=5, puntos_por_seccion=11)
        st.info("Se cargó un set demo para probar la transferencia completa.")

    df_sec = data.get("df_secciones_validas", pd.DataFrame())
    df_pts = data.get("df_puntos_secciones", pd.DataFrame())
    df_desc = data.get("df_secciones_descartadas", pd.DataFrame())
    df_perfil = data.get("df_perfil_longitudinal", pd.DataFrame())

    c1, c2, c3 = st.columns(3)
    c1.metric("Secciones válidas origen", len(df_sec) if df_sec is not None else 0)
    c2.metric("Puntos transversales", len(df_pts) if df_pts is not None else 0)
    c3.metric("Descartadas origen", len(df_desc) if df_desc is not None else 0)

    with st.expander("Vista previa y edición de salidas v13", expanded=True):
        t1, t2, t3 = st.tabs(["df_secciones_validas", "df_puntos_secciones", "df_secciones_descartadas"])
        with t1:
            if df_sec is None or df_sec.empty:
                st.warning("No hay df_secciones_validas en memoria. Ejecuta la app de secciones o carga un Excel.")
            else:
                df_sec = st.data_editor(df_sec, use_container_width=True, num_rows="dynamic", key="editor_df_secciones_validas_v13")
        with t2:
            if df_pts is None or df_pts.empty:
                st.warning("No hay df_puntos_secciones en memoria.")
            else:
                df_pts = st.data_editor(df_pts, use_container_width=True, num_rows="dynamic", key="editor_df_puntos_v13")
        with t3:
            if df_desc is None or df_desc.empty:
                st.caption("Sin secciones descartadas de origen.")
            else:
                st.dataframe(df_desc, use_container_width=True)

    st.subheader("Validación y transferencia")
    col_a, col_b, col_c = st.columns(3)
    caudal_default = col_a.number_input("Caudal inicial si no existe [m³/s]", min_value=0.001, value=50.0, step=5.0)
    manning_default = col_b.number_input("Manning n por defecto", min_value=0.005, value=0.035, step=0.005, format="%.3f")
    aplicar_q = col_c.checkbox("Rellenar Caudal_m3s ahora", value=True)

    if st.button("Validar secciones para HidroSed", type="secondary"):
        errores, advertencias, df_validado = validar_secciones_para_hidrosed(pd.DataFrame(df_sec), pd.DataFrame(df_pts))
        st.session_state["integracion_errores"] = errores
        st.session_state["integracion_advertencias"] = advertencias
        st.session_state["integracion_df_validado"] = df_validado.to_dict("records")
        if errores:
            st.error("Existen errores críticos de integración.")
            st.dataframe(pd.DataFrame(errores), use_container_width=True)
        if advertencias:
            st.warning("Hay secciones excluidas o advertencias de calidad.")
            st.dataframe(pd.DataFrame(advertencias), use_container_width=True)
        if not errores and not df_validado.empty:
            st.success(f"Validación aprobada: {len(df_validado)} secciones modelables.")
            st.dataframe(df_validado, use_container_width=True)

    errores, advertencias, df_validado = validar_secciones_para_hidrosed(pd.DataFrame(df_sec), pd.DataFrame(df_pts)) if not pd.DataFrame(df_sec).empty and not pd.DataFrame(df_pts).empty else ([], [], pd.DataFrame())
    if errores:
        st.error("No es posible transferir automáticamente hasta corregir los errores críticos.")
        st.dataframe(pd.DataFrame(errores), use_container_width=True)
    elif not df_validado.empty:
        st.success(f"Listo para transferencia: {len(df_validado)} secciones modelables.")
        if advertencias:
            with st.expander("Advertencias / secciones no modelables", expanded=False):
                st.dataframe(pd.DataFrame(advertencias), use_container_width=True)

        df_h03 = convertir_secciones_a_hidrosed(
            df_validado,
            caudal_default=float(caudal_default) if aplicar_q else None,
            manning_default=float(manning_default),
        )
        df_h04 = convertir_puntos_a_hidrosed(pd.DataFrame(df_pts), df_validado)

        t1, t2 = st.tabs(["03_Secciones", "04_Puntos_Seccion"])
        with t1:
            st.dataframe(df_h03, use_container_width=True)
        with t2:
            st.dataframe(df_h04, use_container_width=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Transferir secciones válidas a HidroSed Maestra", type="primary"):
                transferir_a_session_state(df_h03, df_h04, df_descartadas=pd.DataFrame(df_desc), advertencias=advertencias)
                st.success("Secciones transferidas correctamente. El módulo hidráulico ya puede calcular con la geometría importada.")
                st.info("Continúa en '6 · Resultados HidroSed' o revisa/edita los datos en '4 · Proyecto y secciones'.")
        with col2:
            folder = get_session_folder()
            out_xlsx = folder / "02_Excel_Secciones_HidroSed.xlsx"
            generar_excel_integracion_hidrosed(
                df_h03,
                df_h04,
                df_descartadas=pd.DataFrame(df_desc),
                df_perfil_longitudinal=pd.DataFrame(df_perfil),
                metadatos={
                    "origen": "v13_fix_km_final_utm19s_3d",
                    "sistema_coordenadas": "WGS84 / UTM huso 19S - EPSG:32719",
                    "n_secciones_transferibles": len(df_h03),
                    "n_puntos_transferibles": len(df_h04),
                },
                output_path=out_xlsx,
            )
            st.download_button(
                "Descargar Excel integración HidroSed",
                data=as_download_bytes(out_xlsx),
                file_name="02_Excel_Secciones_HidroSed.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    if st.session_state.get("secciones_transferidas"):
        st.divider()
        st.subheader("Estado de transferencia actual")
        h03 = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
        h04 = pd.DataFrame(st.session_state.get("hidrosed_puntos_seccion", []))
        c1, c2, c3 = st.columns(3)
        c1.metric("03_Secciones", len(h03))
        c2.metric("04_Puntos_Seccion", len(h04))
        c3.metric("Origen", st.session_state.get("origen_geometria", "N/D"))
        with st.expander("Ver tablas transferidas", expanded=False):
            st.dataframe(h03, use_container_width=True)
            st.dataframe(h04, use_container_width=True)



def page_modelo_3d() -> None:
    st.title("4 · Modelo 3D del cauce y secciones")
    st.write(
        "Visor técnico para revisar el eje longitudinal estimado, las secciones transversales, "
        "la superficie simplificada del cauce y, cuando existan resultados, los niveles de agua y la socavación."
    )

    fuente = st.radio(
        "Fuente de geometría 3D",
        [
            "Secciones transferidas a HidroSed",
            "Salidas v13 en memoria",
            "Datos demo de prueba",
        ],
        horizontal=True,
    )

    resultados = pd.DataFrame(st.session_state.get("hidrosed_results", []))

    if fuente.startswith("Secciones transferidas"):
        df_sec = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
        df_pts = pd.DataFrame(st.session_state.get("hidrosed_puntos_seccion", []))
        if df_sec.empty or df_pts.empty:
            st.warning("Aún no hay secciones transferidas. Usa la etapa 3 o selecciona datos demo.")
            return
    elif fuente.startswith("Salidas v13"):
        data = _load_v13_sources_from_state()
        df_sec = data.get("df_secciones_validas", pd.DataFrame())
        df_pts = data.get("df_puntos_secciones", pd.DataFrame())
        if df_sec.empty or df_pts.empty:
            st.warning("No hay salidas v13 en memoria. Ejecuta/carga la generación de secciones o usa datos demo.")
            return
    else:
        data = generar_datos_demo_v13(n_secciones=8, puntos_por_seccion=15)
        errores, advertencias, df_validado = validar_secciones_para_hidrosed(data["df_secciones_validas"], data["df_puntos_secciones"])
        df_sec = convertir_secciones_a_hidrosed(df_validado, caudal_default=50.0, manning_default=0.035)
        df_pts = convertir_puntos_a_hidrosed(data["df_puntos_secciones"], df_validado)
        st.info("Modelo 3D demo cargado. Úsalo solo para verificar funcionamiento visual.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        exageracion = st.slider("Exageración vertical", 1.0, 10.0, 2.0, 0.5)
    with c2:
        max_sec = st.slider("Máx. secciones visibles", 5, 150, 80, 5)
    with c3:
        mostrar_superficie = st.checkbox("Superficie simplificada", value=True)
    with c4:
        mostrar_agua_soc = st.checkbox("Agua/socavación si existe", value=True)

    c5, c6 = st.columns([1.2, 0.8])
    with c5:
        vista_3d = st.selectbox(
            "Vista predefinida del modelo",
            ["Isométrica", "Planta / superior", "Lateral", "Aguas abajo", "Aguas arriba", "Rotación libre"],
            index=0,
            help="Define la cámara inicial. Luego puedes girar libremente el modelo con el mouse o el dedo.",
        )
    with c6:
        proyeccion = st.radio("Proyección", ["Perspectiva", "Ortográfica"], horizontal=True)

    st.caption(
        "Azul: secciones modelables. Negro: eje longitudinal estimado. Gris: malla simplificada del cauce. "
        "Celeste/naranjo aparecen solo si ya existen resultados HidroSed. Puedes girar el modelo con click izquierdo + arrastrar; "
        "zoom con rueda/pellizco; desplazamiento con click derecho o Shift + arrastrar."
    )

    try:
        fig = generar_figura_3d_cauce_secciones(
            df_puntos=df_pts,
            df_secciones=df_sec,
            resultados=resultados,
            max_sections=int(max_sec),
            max_cross_points=60,
            exageracion_vertical=float(exageracion),
            mostrar_superficie=bool(mostrar_superficie),
            mostrar_agua=bool(mostrar_agua_soc),
            mostrar_socavacion=bool(mostrar_agua_soc),
            vista_3d=vista_3d,
            projection_type="orthographic" if proyeccion == "Ortográfica" else "perspective",
        )
        st.plotly_chart(fig, use_container_width=True)

        folder = get_session_folder()
        vista_slug = str(vista_3d).lower().replace(" ", "_").replace("/", "_")
        html_path = folder / f"Modelo_3D_Cauce_Secciones_{vista_slug}.html"
        exportar_modelo_3d_html(fig, html_path)
        st.session_state["modelo_3d_html"] = str(html_path)
        st.download_button(
            "Descargar modelo 3D en HTML",
            data=as_download_bytes(html_path),
            file_name="Modelo_3D_Cauce_Secciones.html",
            mime="text/html",
        )

        with st.expander("Control de datos usados en el modelo 3D", expanded=False):
            st.write("Secciones:", len(df_sec))
            st.write("Puntos:", len(df_pts))
            st.dataframe(pd.DataFrame(df_sec).head(50), use_container_width=True)
            st.dataframe(pd.DataFrame(df_pts).head(100), use_container_width=True)
    except Exception as exc:
        st.error(f"No fue posible generar el modelo 3D: {exc}")


def _params_from_form() -> GlobalParams:
    d = st.session_state.get("global_params", {}).copy()
    return GlobalParams(**d)


def page_project_sections() -> None:
    st.title("5 · Proyecto y secciones HidroSed Cauces")
    st.write("Este módulo conserva la lógica de HidroSed: proyecto → N secciones → hidráulica → transporte → socavación.")
    if st.session_state.get("secciones_transferidas"):
        st.success("Hay secciones transferidas desde v13. Puedes revisarlas o editarlas antes del cálculo.")
        with st.expander("Ver tabla 03_Secciones transferida", expanded=False):
            st.dataframe(pd.DataFrame(st.session_state.get("hidrosed_secciones", [])), use_container_width=True)
    with st.expander("Datos globales del proyecto", expanded=True):
        params = _params_from_form()
        c1, c2, c3 = st.columns(3)
        with c1:
            params.project_name = st.text_input("Proyecto", params.project_name)
            params.river_name = st.text_input("Cauce", params.river_name)
            params.location_name = st.text_input("Ubicación", params.location_name)
            params.condition = st.selectbox("Condición", ["Sin proyecto", "Con proyecto", "Alternativa", "Diagnóstico"], index=["Sin proyecto", "Con proyecto", "Alternativa", "Diagnóstico"].index(params.condition) if params.condition in ["Sin proyecto", "Con proyecto", "Alternativa", "Diagnóstico"] else 0)
        with c2:
            params.return_period_years = st.number_input("Período de retorno Tr [años]", min_value=1.01, value=float(params.return_period_years), step=1.0)
            params.rho_w = st.number_input("Densidad agua ρw [kg/m³]", min_value=500.0, value=float(params.rho_w), step=10.0)
            params.rho_s = st.number_input("Densidad sedimento ρs [kg/m³]", min_value=1000.0, value=float(params.rho_s), step=10.0)
            params.porosity = st.slider("Porosidad lecho λ", 0.10, 0.60, float(params.porosity), 0.01)
        with c3:
            params.theta_c = st.number_input("Shields crítico θc", min_value=0.001, value=float(params.theta_c), step=0.001, format="%.3f")
            params.sediment_supply_factor = st.number_input("Factor aporte sólido", min_value=0.0, value=float(params.sediment_supply_factor), step=0.1)
            params.scour_safety_factor = st.number_input("Factor seguridad socavación", min_value=0.0, value=float(params.scour_safety_factor), step=0.1)
            params.use_roughness_correction = st.checkbox("Corregir θ por rugosidad MPM", value=bool(params.use_roughness_correction))
        st.session_state["global_params"] = params.__dict__

    st.subheader("Crear o editar secciones")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.number_input("N° secciones", min_value=1, max_value=200, value=5, step=1)
        dx = st.number_input("Separación Δx [m]", min_value=0.1, value=100.0, step=10.0)
    with c2:
        q = st.number_input("Q global [m³/s]", min_value=0.001, value=50.0, step=5.0)
        slope = st.number_input("Pendiente S [m/m]", min_value=0.000001, value=0.010, step=0.001, format="%.5f")
    with c3:
        manning = st.number_input("Manning n", min_value=0.005, value=0.040, step=0.005, format="%.3f")
        b = st.number_input("Ancho basal B [m]", min_value=0.1, value=12.0, step=1.0)
    with c4:
        y = st.number_input("Tirante y [m]", min_value=0.01, value=2.0, step=0.1)
        z = st.number_input("Talud z H/V", min_value=0.0, value=1.5, step=0.1)

    if st.button("Generar tabla base de secciones", type="secondary"):
        sections = default_sections(n_sections=int(n), q_m3s=float(q), slope=float(slope), manning_n=float(manning), dx_m=float(dx), bottom_width_m=float(b), depth_m=float(y), side_slope_z=float(z))
        if st.session_state.get("grain_stats"):
            sections = apply_grain_stats_to_sections(sections, st.session_state["grain_stats"])
        st.session_state["sections_df"] = sections_to_dataframe(sections)
        st.success("Tabla base generada.")

    st.caption("Puedes editar celdas directamente. Campos principales: q_m3s, slope, manning_n, bottom_width_m, depth_m, bed_elevation_m, curva y diámetros.")
    edited = st.data_editor(st.session_state["sections_df"], use_container_width=True, num_rows="dynamic", key="sections_editor")
    if st.button("Guardar cambios de secciones", type="primary"):
        st.session_state["sections_df"] = pd.DataFrame(edited)
        st.success("Secciones guardadas.")

    csv_upload = st.file_uploader("Opcional: cargar secciones desde CSV", type=["csv"])
    if csv_upload is not None and st.button("Importar CSV de secciones"):
        try:
            df = pd.read_csv(csv_upload)
            st.session_state["sections_df"] = df
            st.success("CSV importado. Revisa la tabla y guarda cambios.")
        except Exception as exc:
            st.error(f"No se pudo importar CSV: {exc}")


def _apply_default_granulometry_to_state(profile_key: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga un perfil estándar cuando no existe granulometría medida y lo aplica a secciones disponibles."""
    key = profile_key or st.session_state.get("grain_default_profile_key") or DEFAULT_STANDARD_PROFILE_KEY
    payload = build_default_grain_session_payload(key)
    st.session_state.update(payload)
    stats_df = pd.DataFrame(payload["grain_stats_multi"])

    # Aplicar a secciones simplificadas si existen.
    sections_df = pd.DataFrame(st.session_state.get("sections_df", []))
    if not sections_df.empty:
        st.session_state["sections_df"] = assign_grain_to_sections(sections_df, stats_df)

    # Aplicar a 03_Secciones si ya fueron transferidas.
    h03 = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
    if not h03.empty:
        st.session_state["hidrosed_secciones"] = assign_grain_to_sections(h03, stats_df).to_dict("records")

    return pd.DataFrame(payload["grain_df"]), stats_df


def _ensure_grain_for_calculation() -> None:
    """Garantiza que el motor nunca calcule sin D50/D84/D90 explícitos."""
    has_user_or_default = bool(st.session_state.get("grain_stats_multi")) or bool(st.session_state.get("grain_stats"))
    if not has_user_or_default:
        # Recomendación por pendiente mediana si existe.
        h03 = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
        sections_df = pd.DataFrame(st.session_state.get("sections_df", []))
        slope = None
        for df, cols in [(h03, ["Pendiente_m_m", "slope", "slope_m_m"]), (sections_df, ["slope", "Pendiente_m_m"] )]:
            if not df.empty:
                for c in cols:
                    if c in df.columns:
                        vals = pd.to_numeric(df[c], errors="coerce").dropna()
                        if len(vals):
                            slope = float(vals.median())
                            break
                if slope is not None:
                    break
        key = recommend_standard_profile_by_context(slope=slope)
        _apply_default_granulometry_to_state(key)
        st.session_state["grain_autoloaded_for_calculation"] = True
    else:
        stats_df = pd.DataFrame(st.session_state.get("grain_stats_multi", []))
        if not stats_df.empty:
            h03 = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
            if not h03.empty and not any(c in h03.columns for c in ["D50_mm", "d50_mm"]):
                st.session_state["hidrosed_secciones"] = assign_grain_to_sections(h03, stats_df).to_dict("records")
            sections_df = pd.DataFrame(st.session_state.get("sections_df", []))
            if not sections_df.empty and not any(c in sections_df.columns for c in ["D50_mm", "d50_mm"]):
                st.session_state["sections_df"] = assign_grain_to_sections(sections_df, stats_df)


def page_grain() -> None:
    st.title("7 · Granulometría avanzada")
    st.markdown(
        """
        Ingresa una o varias granulometrías del cauce. La aplicación interpola automáticamente los diámetros
        característicos **D5, D10, D16, D25, D30, D35, D50, D60, D65, D75, D84, D90 y D95** mediante interpolación logarítmica.
        Si las muestras tienen PK, los diámetros se interpolan espacialmente hacia cada sección.

        Si no cuentas con granulometría medida, activa un **perfil estándar de respaldo**. El sistema dejará trazabilidad y reducirá la confianza técnica, porque no reemplaza una muestra de terreno.
        """
    )
    modo = st.radio("Fuente granulométrica", ["Perfil estándar por defecto", "Texto", "CSV/Excel"], horizontal=True)
    grain_df = pd.DataFrame()

    if modo == "Perfil estándar por defecto":
        opts = standard_profile_options()
        labels = {k: v for k, v in opts}
        default_key = st.session_state.get("grain_default_profile_key", DEFAULT_STANDARD_PROFILE_KEY)
        if default_key not in labels:
            default_key = DEFAULT_STANDARD_PROFILE_KEY
        selected = st.selectbox(
            "Perfil estándar de lecho",
            options=list(labels.keys()),
            index=list(labels.keys()).index(default_key),
            format_func=lambda k: labels[k],
            help="Usar solo como respaldo cuando no existe granulometría. Para diseño definitivo cargar muestras de terreno.",
        )
        prof = STANDARD_GRAIN_PROFILES[selected]
        st.session_state["grain_default_profile_key"] = selected
        st.info(f"Perfil seleccionado: {prof['nombre']} · Clase: {prof['clase_wentworth']} · D50 objetivo aprox.: {prof['d50_objetivo_mm']} mm")
        st.caption(prof["descripcion"])
        c1, c2 = st.columns([1, 1])
        with c1:
            curva_preview = get_standard_grain_profile(selected)
            st.subheader("Curva estándar acumulada")
            st.dataframe(curva_preview, use_container_width=True)
        with c2:
            stats_preview = get_standard_grain_stats(selected)
            st.subheader("Diámetros estimados")
            st.dataframe(stats_preview, use_container_width=True)
        if st.button("Aplicar perfil estándar a HidroSed", type="primary"):
            grain_df, stats_df = _apply_default_granulometry_to_state(selected)
            st.warning("Se aplicó granulometría estándar de respaldo. La app continuará el cálculo, pero el informe marcará esta condición como estimativa y no equivalente a muestreo real.")
            st.dataframe(stats_df, use_container_width=True)

    elif modo == "Texto":
        st.caption("Formato simple: diametro_mm;porcentaje_pasa. Formato múltiple: id_muestra;PK_m;diametro_mm;porcentaje_pasa")
        text = st.text_area("Curva(s) granulométrica(s)", value=st.session_state.get("grain_text", DEFAULT_GRAIN), height=240)
        st.session_state["grain_text"] = text
        if st.button("Calcular/interpolar diámetros", type="primary"):
            try:
                grain_df = parse_grain_text_multi(text)
                st.session_state["grain_default_profile_active"] = False
                st.session_state["grain_source_status"] = "GRANULOMETRIA_USUARIO_TEXTO"
            except Exception as exc:
                st.error(f"No se pudo leer la granulometría: {exc}")
                return
    else:
        upload = st.file_uploader("Cargar granulometría CSV/XLSX", type=["csv", "xlsx", "xls"], key="grain_multi_upload")
        if upload is not None and st.button("Leer archivo granulométrico", type="primary"):
            try:
                if upload.name.lower().endswith(".csv"):
                    raw = pd.read_csv(upload)
                else:
                    raw = pd.read_excel(upload)
                grain_df = normalize_grain_table(raw)
                st.session_state["grain_default_profile_active"] = False
                st.session_state["grain_source_status"] = f"GRANULOMETRIA_USUARIO_ARCHIVO:{upload.name}"
            except Exception as exc:
                st.error(f"No se pudo leer el archivo: {exc}")
                return

    if not grain_df.empty:
        try:
            stats_df = stats_by_sample(grain_df)
            if not bool(st.session_state.get("grain_default_profile_active", False)):
                stats_df["Fuente_granulometria"] = "Usuario / muestreo ingresado"
                stats_df["Advertencia_granulometria"] = "Datos ingresados por el usuario. Verificar representatividad espacial y temporal."
            st.session_state["grain_df"] = grain_df.to_dict("records")
            st.session_state["grain_stats_multi"] = stats_df.to_dict("records")
            first = stats_df.iloc[0].to_dict()
            first["Es_granulometria_default"] = bool(st.session_state.get("grain_default_profile_active", False))
            st.session_state["grain_stats"] = first
            st.success("Granulometría calculada e interpolada.")
            st.subheader("Curvas ingresadas")
            st.dataframe(grain_df, use_container_width=True)
            st.subheader("Diámetros característicos interpolados")
            st.dataframe(stats_df, use_container_width=True)
        except Exception as exc:
            st.error(f"No se pudo calcular estadística granulométrica: {exc}")

    if st.session_state.get("grain_stats_multi"):
        stats_df = pd.DataFrame(st.session_state["grain_stats_multi"])
        st.subheader("Granulometrías guardadas")
        if st.session_state.get("grain_default_profile_active"):
            st.warning("Estado actual: se está usando granulometría estándar por defecto. Reemplázala por muestras de terreno cuando estén disponibles.")
        st.dataframe(stats_df, use_container_width=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Aplicar/interpolar granulometría a secciones", type="primary"):
                try:
                    sections_df = pd.DataFrame(st.session_state.get("sections_df", []))
                    out = assign_grain_to_sections(sections_df, stats_df)
                    st.session_state["sections_df"] = out
                    # Si ya existen tablas 03_Secciones, anexar D por PK para el motor avanzado.
                    h03 = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
                    if not h03.empty:
                        h03b = assign_grain_to_sections(h03, stats_df)
                        st.session_state["hidrosed_secciones"] = h03b.to_dict("records")
                    st.success("Granulometría asignada a las secciones por PK/distancia.")
                    st.dataframe(out, use_container_width=True)
                except Exception as exc:
                    st.error(f"No se pudo aplicar a secciones: {exc}")
        with c2:
            folder = get_session_folder()
            xlsx = folder / "Granulometria_interpolada_HidroSed.xlsx"
            with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
                pd.DataFrame(st.session_state.get("grain_df", [])).to_excel(writer, index=False, sheet_name="Curvas_granulometricas")
                stats_df.to_excel(writer, index=False, sheet_name="Diametros_interpolados")
                pd.DataFrame([{
                    "Estado": st.session_state.get("grain_source_status", "N/D"),
                    "Perfil_default": st.session_state.get("grain_default_profile_key", ""),
                    "Usa_granulometria_default": bool(st.session_state.get("grain_default_profile_active", False)),
                    "Advertencia": "Perfil estimativo; no reemplaza granulometría de terreno." if st.session_state.get("grain_default_profile_active") else "Datos ingresados por usuario.",
                }]).to_excel(writer, index=False, sheet_name="Metadatos")
            st.download_button("Descargar Excel granulometría", data=as_download_bytes(xlsx), file_name="Granulometria_interpolada_HidroSed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

def page_idf() -> None:
    st.title("8 · Curvas IDF")
    st.markdown(
        """
        Genera curvas **Intensidad–Duración–Frecuencia** accesibles para el cálculo hidrológico.
        Puedes usar coeficientes IDF calibrados o una tabla P24 por período de retorno para generar curvas sintéticas de prediseño.
        """
    )
    durations_default = "5,10,15,30,60,120,180,360,720,1440"
    trs_default = "2,5,10,25,50,100,200"
    modo = st.radio("Método de generación", ["Coeficientes IDF i=a·Tr^b/(t+c)^d", "Desde P24 por período de retorno"], horizontal=False)
    c1, c2 = st.columns([1, 1])
    with c1:
        durations_txt = st.text_input("Duraciones [min] separadas por coma", value=durations_default)
    with c2:
        trs_txt = st.text_input("Períodos de retorno [años]", value=trs_default)
    durations = [float(x.strip()) for x in durations_txt.split(',') if x.strip()]
    trs = [float(x.strip()) for x in trs_txt.split(',') if x.strip()]

    idf_df = pd.DataFrame()
    if modo.startswith("Coeficientes"):
        ca, cb, cc, cd = st.columns(4)
        a = ca.number_input("a", value=950.0, step=10.0)
        b = cb.number_input("b", value=0.18, step=0.01, format="%.3f")
        c = cc.number_input("c [min]", value=10.0, step=1.0)
        d = cd.number_input("d", value=0.75, step=0.01, format="%.3f")
        if st.button("Generar curvas IDF", type="primary"):
            idf_df = generate_idf_power_law(float(a), float(b), float(c), float(d), durations, trs)
            st.session_state["idf_metodo"] = modo
            st.session_state["idf_parametros"] = {"a": float(a), "b": float(b), "c": float(c), "d": float(d)}
    else:
        st.caption("Formato: Tr;P24_mm. Ejemplo: 100;85")
        p24_txt = st.text_area("P24 por período", value="2;35\n5;45\n10;55\n25;68\n50;78\n100;90\n200;105", height=180)
        if st.button("Generar IDF desde P24", type="primary"):
            try:
                p24 = parse_p24_text(p24_txt)
                idf_df = generate_idf_from_p24(p24, durations)
                st.session_state["idf_metodo"] = modo
                st.session_state["idf_parametros"] = {"P24_T": p24}
            except Exception as exc:
                st.error(f"No se pudo generar IDF desde P24: {exc}")
                return

    if not idf_df.empty:
        st.session_state["idf_df"] = idf_df.to_dict("records")
        st.success("Curvas IDF generadas y guardadas en memoria.")

    if st.session_state.get("idf_df"):
        df = pd.DataFrame(st.session_state["idf_df"])
        st.subheader("Tabla IDF")
        st.dataframe(df, use_container_width=True)
        st.subheader("Intensidad IDF [mm/h]")
        try:
            piv = df.pivot_table(index="Duracion_min", columns="Tr_anios", values="Intensidad_mm_h", aggfunc="mean")
            st.line_chart(piv)
        except Exception:
            pass
        folder = get_session_folder()
        xlsx = folder / "Curvas_IDF_HidroSed.xlsx"
        export_idf_excel(df, xlsx)
        st.download_button("Descargar curvas IDF Excel", data=as_download_bytes(xlsx), file_name="Curvas_IDF_HidroSed.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("Descargar curvas IDF CSV", data=df.to_csv(index=False).encode('utf-8'), file_name="Curvas_IDF_HidroSed.csv", mime="text/csv")


def page_results() -> None:
    st.title("9 · Resultados HidroSed · Motor experto v6.3")
    params = _params_from_form()
    _ensure_grain_for_calculation()

    st.markdown(
        """
        Esta versión incorpora dos rutas de cálculo:

        **A. Motor avanzado con sección irregular real**: usa las tablas `03_Secciones` y `04_Puntos_Seccion`,
        resuelve automáticamente la lámina normal por Manning, calcula calado crítico, Froude, energía,
        Shields, Meyer-Peter-Müller, Engelund-Hansen, socavación y auditoría QA.

        **B. Motor simplificado trapecial**: queda disponible como respaldo cuando no existen puntos transversales.
        """
    )

    h03 = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
    h04 = pd.DataFrame(st.session_state.get("hidrosed_puntos_seccion", []))
    has_irregular = not h03.empty and not h04.empty

    c0, c1, c2, c3 = st.columns(4)
    c0.metric("Motor irregular", "Disponible" if has_irregular else "No disponible")
    c1.metric("03_Secciones", len(h03))
    c2.metric("04_Puntos", len(h04))
    c3.metric("Tr [años]", f"{float(params.return_period_years):.0f}")
    if st.session_state.get("grain_default_profile_active"):
        st.warning("Granulometría: usando perfil estándar por defecto. El cálculo continúa con D50/D84/D90 estimados, pero el informe debe indicar que no existe muestreo de terreno.")

    modo = st.radio(
        "Modo de cálculo",
        [
            "Motor conectado v6 · perfil 1D + incertidumbre + confianza",
            "Motor avanzado v5 · secciones independientes",
            "Motor simplificado trapecial de respaldo",
        ],
        index=0 if has_irregular else 2,
        horizontal=False,
    )

    with st.expander("Parámetros expertos de auditoría", expanded=False):
        ca, cb, cc, cd = st.columns(4)
        temp_c = ca.number_input("Temperatura agua [°C]", min_value=0.0, max_value=40.0, value=20.0, step=1.0)
        rho_w_calc = water_density_kgm3(float(temp_c))
        ca.caption(f"ρw calculada: {rho_w_calc:.2f} kg/m³")
        usar_rho_temp = ca.checkbox("Usar ρw por temperatura", value=True)
        alpha_v = cb.number_input("α velocidad", min_value=0.5, max_value=2.0, value=1.0, step=0.05)
        contraction = cc.number_input("Coef. contracción", min_value=0.50, max_value=2.00, value=1.00, step=0.05)
        expansion = cd.number_input("Coef. expansión", min_value=0.50, max_value=2.00, value=1.00, step=0.05)
        sensibilidad = st.checkbox("Ejecutar sensibilidad Manning ±20%", value=True)
        mc_enabled = st.checkbox("Ejecutar incertidumbre Monte Carlo", value=True)
        cm1, cm2, cm3, cm4 = st.columns(4)
        n_mc = cm1.number_input("Corridas Monte Carlo", min_value=20, max_value=300, value=80, step=20)
        q_cv = cm2.number_input("CV caudal", min_value=0.0, max_value=0.5, value=0.10, step=0.01)
        n_cv = cm3.number_input("CV Manning", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
        d_cv = cm4.number_input("CV D50/pendiente", min_value=0.0, max_value=0.6, value=0.20, step=0.01)
        bc = st.selectbox("Condición de borde aguas abajo", ["Tirante normal", "Cota conocida"], index=0)
        known_wse = None
        if bc == "Cota conocida":
            known_wse = st.number_input("Cota lámina aguas abajo conocida [m]", value=0.0, step=0.10)
        st.caption("La sensibilidad e incertidumbre permiten detectar dependencia excesiva de rugosidad, caudal, condición de borde, granulometría o geometría insuficiente.")

    if modo.startswith("Motor conectado"):
        if not has_irregular:
            st.error("No hay tablas transferidas `03_Secciones` y `04_Puntos_Seccion`. Usa primero la etapa 3 de integración v13.")
            return
        with st.expander("Calibración opcional con datos observados", expanded=False):
            st.write("Carga una tabla CSV/XLSX con `ID_Seccion` o `PK_m` y `Cota_lamina_observada_m`. Si existe calibración, el sistema puede alcanzar 9/10; sin observaciones el techo ético queda en 8.6/10.")
            obs_file = st.file_uploader("Datos observados de lámina de agua", type=["csv", "xlsx"], key="obs_wse_calibracion")
            observed_df = pd.DataFrame()
            if obs_file is not None:
                try:
                    if obs_file.name.lower().endswith(".csv"):
                        observed_df = pd.read_csv(obs_file)
                    else:
                        observed_df = pd.read_excel(obs_file)
                    st.dataframe(observed_df, use_container_width=True)
                except Exception as exc:
                    st.error(f"No se pudo leer la tabla observada: {exc}")
        if st.button("Calcular con motor conectado v6 · confianza 9/10", type="primary"):
            try:
                conn_params = ConnectedProfileParams(
                    rho_w=float(rho_w_calc if usar_rho_temp else params.rho_w),
                    rho_s=float(params.rho_s),
                    theta_c=float(params.theta_c),
                    porosity=float(params.porosity),
                    temp_c=float(temp_c),
                    alpha_velocity=float(alpha_v),
                    sediment_supply_factor=float(params.sediment_supply_factor),
                    scour_safety_factor=float(params.scour_safety_factor),
                    use_roughness_correction=bool(params.use_roughness_correction),
                    contraction_coeff=float(contraction),
                    expansion_coeff=float(expansion),
                    downstream_boundary="known_wse" if bc == "Cota conocida" else "normal_depth",
                    downstream_wse_m=float(known_wse) if known_wse is not None else None,
                    max_reasonable_dx_m=300.0,
                )
                h03_calc = h03.copy()
                calib_df = pd.DataFrame()
                if not observed_df.empty:
                    factor, calib_df = calibrate_manning_multiplier(
                        h03_calc, h04, observed_df, params=conn_params, return_period_years=float(params.return_period_years)
                    )
                    # aplicar factor calibrado al cálculo final
                    if "Manning_n" in h03_calc.columns:
                        h03_calc["Manning_n"] = pd.to_numeric(h03_calc["Manning_n"], errors="coerce") * factor
                    st.info(f"Factor Manning calibrado aplicado: {factor:.3f}")
                results = compute_connected_profile_v6(
                    h03_calc,
                    h04,
                    params=conn_params,
                    return_period_years=float(params.return_period_years),
                )
                sens = pd.DataFrame()
                if sensibilidad:
                    sens = sensitivity_manning_irregular(
                        h03_calc,
                        h04,
                        params=conn_params,
                        n_factors=(0.8, 1.0, 1.2),
                        return_period_years=float(params.return_period_years),
                    )
                unc = pd.DataFrame()
                if mc_enabled:
                    unc = monte_carlo_uncertainty_v6(
                        h03_calc,
                        h04,
                        params=conn_params,
                        return_period_years=float(params.return_period_years),
                        n_runs=int(n_mc),
                        q_cv=float(q_cv),
                        n_cv=float(n_cv),
                        d50_cv=float(d_cv),
                        slope_cv=float(d_cv),
                    )
                audit = build_audit_summary(results, sens)
                conf_score, conf_df = confidence_score_v6(results, sens, unc, calib_df)
                if st.session_state.get("grain_default_profile_active"):
                    conf_score = min(float(conf_score), 8.2)
                    extra = pd.DataFrame([{
                        "criterio": "Granulometría de terreno",
                        "valor": f"no disponible · perfil estándar: {st.session_state.get('grain_default_profile_key', 'N/D')}",
                        "penalizacion": "techo 8.2",
                        "estado": "REVISAR",
                    }])
                    conf_df = pd.concat([conf_df, extra], ignore_index=True)
                    if not conf_df.empty and conf_df.loc[0, "criterio"] == "Nivel de confianza global":
                        conf_df.loc[0, "valor"] = round(float(conf_score), 1)
                        conf_df.loc[0, "estado"] = "REVISAR"
                st.session_state["hidrosed_results"] = results.to_dict("records")
                st.session_state["hidrosed_results_advanced"] = results.to_dict("records")
                st.session_state["hidrosed_sensitivity_manning"] = sens.to_dict("records") if not sens.empty else []
                st.session_state["hidrosed_uncertainty_v6"] = unc.to_dict("records") if not unc.empty else []
                st.session_state["hidrosed_calibration_v6"] = calib_df.to_dict("records") if not calib_df.empty else []
                st.session_state["hidrosed_confidence_v6"] = conf_df.to_dict("records")
                st.session_state["hidrosed_confidence_score"] = conf_score
                st.session_state["hidrosed_audit_summary"] = audit.to_dict("records")
                summary = technical_summary(results)
                st.session_state["hidrosed_summary"] = summary
                st.success(f"Cálculo conectado v6 terminado. Nivel de confianza técnico: {conf_score:.1f}/10.")
            except Exception as exc:
                st.error(f"No se pudo calcular con motor conectado v6: {exc}")

    elif modo.startswith("Motor avanzado"):
        if not has_irregular:
            st.error("No hay tablas transferidas `03_Secciones` y `04_Puntos_Seccion`. Usa primero la etapa 3 de integración v13.")
            return
        if st.button("Calcular con motor experto irregular v5", type="primary"):
            try:
                adv_params = AdvancedModelParams(
                    rho_w=float(rho_w_calc if usar_rho_temp else params.rho_w),
                    rho_s=float(params.rho_s),
                    theta_c=float(params.theta_c),
                    porosity=float(params.porosity),
                    temp_c=float(temp_c),
                    alpha_velocity=float(alpha_v),
                    sediment_supply_factor=float(params.sediment_supply_factor),
                    scour_safety_factor=float(params.scour_safety_factor),
                    use_roughness_correction=bool(params.use_roughness_correction),
                    contraction_coeff=float(contraction),
                    expansion_coeff=float(expansion),
                )
                results = compute_irregular_hydraulics(
                    h03,
                    h04,
                    params=adv_params,
                    return_period_years=float(params.return_period_years),
                )
                sens = pd.DataFrame()
                if sensibilidad:
                    sens = sensitivity_manning_irregular(
                        h03,
                        h04,
                        params=adv_params,
                        n_factors=(0.8, 1.0, 1.2),
                        return_period_years=float(params.return_period_years),
                    )
                audit = build_audit_summary(results, sens)
                st.session_state["hidrosed_results"] = results.to_dict("records")
                st.session_state["hidrosed_results_advanced"] = results.to_dict("records")
                st.session_state["hidrosed_sensitivity_manning"] = sens.to_dict("records") if not sens.empty else []
                st.session_state["hidrosed_audit_summary"] = audit.to_dict("records")
                summary = technical_summary(results)
                st.session_state["hidrosed_summary"] = summary
                st.success("Cálculo experto v5 terminado. Resultados asociados a cada sección y PK.")
            except Exception as exc:
                st.error(f"No se pudo calcular con motor experto irregular: {exc}")

    else:
        try:
            sections = dataframe_to_sections(pd.DataFrame(st.session_state["sections_df"]))
        except Exception as exc:
            st.error(f"La tabla de secciones no es válida: {exc}")
            return
        st.warning("Usando motor simplificado trapecial. Para revisión final, se recomienda transferir secciones v13 con puntos transversales.")
        if st.button("Calcular HidroSed Cauces simplificado", type="primary"):
            try:
                results = compute_all(sections, params)
                st.session_state["hidrosed_results"] = results.to_dict("records")
                summary = technical_summary(results)
                st.session_state["hidrosed_summary"] = summary
                st.success("Cálculo simplificado terminado.")
            except Exception as exc:
                st.error(f"No se pudo calcular: {exc}")

    if st.session_state.get("hidrosed_results"):
        df = pd.DataFrame(st.session_state["hidrosed_results"])
        summary = st.session_state.get("hidrosed_summary") or technical_summary(df)
        st.subheader("Resumen crítico")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Velocidad máx. [m/s]", f"{summary.get('velocidad_max_m_s', 0):.2f}", summary.get("seccion_velocidad_max", ""))
        c2.metric("Socavación máx. [m]", f"{summary.get('socavacion_max_m', 0):.2f}", summary.get("seccion_socavacion_max", ""))
        c3.metric("Índice movilidad máx.", f"{summary.get('indice_movilidad_max', 0):.2f}", summary.get("seccion_movilidad_max", ""))
        c4.metric("Secciones móviles", str(summary.get("secciones_moviles", 0)))

        if st.session_state.get("hidrosed_audit_summary"):
            st.subheader("Auditoría hidráulica QA")
            audit_df = pd.DataFrame(st.session_state["hidrosed_audit_summary"])
            st.dataframe(audit_df, use_container_width=True)
            ncrit = int((audit_df.get("estado", pd.Series(dtype=str)).astype(str) == "CRITICO").sum())
            if ncrit:
                st.error("Existen criterios críticos: no usar resultados sin corregir geometría, pendiente, caudal o rugosidad.")
            else:
                st.success("No se detectaron errores críticos de QA. Revisa igualmente las advertencias profesionales.")

        if st.session_state.get("hidrosed_confidence_v6"):
            st.subheader("Nivel de confianza técnico v6")
            conf_score = st.session_state.get("hidrosed_confidence_score", 0)
            st.metric("Confianza global", f"{float(conf_score):.1f}/10")
            st.dataframe(pd.DataFrame(st.session_state["hidrosed_confidence_v6"]), use_container_width=True)

        if st.session_state.get("hidrosed_uncertainty_v6"):
            st.subheader("Incertidumbre Monte Carlo v6")
            unc_df = pd.DataFrame(st.session_state["hidrosed_uncertainty_v6"])
            st.dataframe(unc_df, use_container_width=True)
            try:
                cols_unc = [c for c in ["Cota_lamina_agua_m_P10", "Cota_lamina_agua_m_P50", "Cota_lamina_agua_m_P90"] if c in unc_df.columns]
                if cols_unc:
                    st.line_chart(unc_df.set_index("PK_m")[cols_unc])
            except Exception:
                pass

        if st.session_state.get("hidrosed_calibration_v6"):
            st.subheader("Calibración Manning v6")
            st.dataframe(pd.DataFrame(st.session_state["hidrosed_calibration_v6"]), use_container_width=True)

        st.subheader("Tabla integrada por sección")
        st.dataframe(df, use_container_width=True)

        if "Advertencias_QA" in df.columns:
            adv = df.loc[df["Advertencias_QA"].astype(str).str.len() > 0, ["ID", "Distancia_m", "Advertencias_QA", "Estado_QA"]]
            if not adv.empty:
                with st.expander("Advertencias por sección", expanded=True):
                    st.dataframe(adv, use_container_width=True)

        st.subheader("Gráficos longitudinales")
        plot_cols = [c for c in ["V_Q_m_s", "Fr", "Indice_movilidad", "Socavacion_ajustada_m", "Cota_fondo_socavado_m", "Cota_lamina_agua_m"] if c in df.columns]
        if plot_cols and "Distancia_m" in df.columns:
            st.line_chart(df.set_index("Distancia_m")[plot_cols])

        if st.session_state.get("hidrosed_sensitivity_manning"):
            st.subheader("Sensibilidad a Manning ±20%")
            sens_df = pd.DataFrame(st.session_state["hidrosed_sensitivity_manning"])
            st.dataframe(sens_df, use_container_width=True)
            try:
                piv = sens_df.pivot_table(index="ID", columns="factor_n", values="Cota_lamina_agua_m", aggfunc="mean")
                st.line_chart(piv)
            except Exception:
                pass

        with st.expander("Rugosidad granulométrica recomendada", expanded=False):
            if "D50_mm" in df.columns:
                sample = df.iloc[0]
                st.dataframe(pd.DataFrame([roughness_estimators(sample.get("D50_mm", 32), sample.get("D84_mm", 64), sample.get("D90_mm", 90))]), use_container_width=True)
            st.caption("Compara este n granular con el Manning total: el Manning total debe incluir forma de fondo, vegetación, irregularidad, meandros y llanuras.")

def page_exner() -> None:
    st.title("10 · Lecho móvil conceptual")
    if not st.session_state.get("hidrosed_results"):
        st.warning("Primero calcula los resultados HidroSed.")
        return
    params = _params_from_form()
    results = pd.DataFrame(st.session_state["hidrosed_results"])
    c1, c2, c3 = st.columns(3)
    with c1:
        duration = st.number_input("Duración del evento [h]", min_value=0.01, value=6.0, step=1.0)
    with c2:
        dt = st.number_input("Paso temporal [h]", min_value=0.01, value=1.0, step=0.5)
    with c3:
        upstream_qs = st.number_input("Aporte sólido aguas arriba [m³/h]", min_value=0.0, value=0.0, step=1.0)
    if st.button("Simular tendencia Exner", type="primary"):
        try:
            exner = exner_simulation(results, params=params, duration_h=float(duration), dt_h=float(dt), upstream_qs_m3h=float(upstream_qs))
            st.session_state["exner_results"] = exner.to_dict("records")
            st.success("Simulación conceptual terminada.")
        except Exception as exc:
            st.error(f"No se pudo simular: {exc}")
    if st.session_state.get("exner_results"):
        exner_df = pd.DataFrame(st.session_state["exner_results"])
        st.dataframe(exner_df, use_container_width=True)
        st.line_chart(exner_df.set_index("Distancia_m")[["Delta_z_total_m", "Cota_fondo_final_m"]])


def page_report() -> None:
    st.title("11 · Reporte y descargas")
    folder = get_session_folder()
    params = _params_from_form()
    metadata = {
        "session_id": st.session_state.get("session_id"),
        "control_point": st.session_state.get("control_point"),
        "bbox": st.session_state.get("bbox"),
        "dem_path": st.session_state.get("dem_path"),
        "dem_metadata": st.session_state.get("dem_metadata"),
        "curvas_kmz_path": st.session_state.get("curvas_kmz_path"),
        "curvas_maestras_kmz_path": st.session_state.get("curvas_maestras_kmz_path"),
        "curvas_apoyo_kmz_path": st.session_state.get("curvas_apoyo_kmz_path"),
        "curvas_apoyo_resumen": st.session_state.get("curvas_apoyo_resumen"),
        "eje_cauce_kmz_path": st.session_state.get("eje_cauce_kmz_path"),
        "eje_cauce_origen": st.session_state.get("eje_cauce_origen"),
        "curvas_metadata": st.session_state.get("curvas_metadata"),
        "proyecto": params.project_name,
        "cauce": params.river_name,
        "ubicacion": params.location_name,
    }
    st.subheader("Estado del proyecto")
    st.json(metadata)

    if not st.session_state.get("hidrosed_results"):
        st.warning("Aún no hay resultados HidroSed para exportar.")
    else:
        sections_df = pd.DataFrame(st.session_state["sections_df"])
        results_df = pd.DataFrame(st.session_state["hidrosed_results"])
        grain_df = pd.DataFrame(st.session_state.get("grain_df", []))
        exner_df = pd.DataFrame(st.session_state.get("exner_results", []))
        audit_df = pd.DataFrame(st.session_state.get("hidrosed_audit_summary", []))
        sens_df = pd.DataFrame(st.session_state.get("hidrosed_sensitivity_manning", []))
        uncertainty_df = pd.DataFrame(st.session_state.get("hidrosed_uncertainty_v6", []))
        calibration_df = pd.DataFrame(st.session_state.get("hidrosed_calibration_v6", []))
        confidence_df = pd.DataFrame(st.session_state.get("hidrosed_confidence_v6", []))
        idf_df = pd.DataFrame(st.session_state.get("idf_df", []))
        grain_stats_multi_df = pd.DataFrame(st.session_state.get("grain_stats_multi", []))
        if st.button("Generar Excel + HTML + JSON", type="primary"):
            excel_path = folder / "HidroSed_Cauces_resultados_v6.xlsx"
            html_path = folder / "Reporte_HidroSed_Cauces_v6.html"
            json_path = folder / "Proyecto_HidroSed_Cauces_v6.json"
            export_hidrosed_excel(excel_path, metadata, params, sections_df, results_df, grain_df, exner_df)
            # Agregar hojas avanzadas sin romper el exportador base
            if not audit_df.empty or not sens_df.empty or not uncertainty_df.empty or not calibration_df.empty or not confidence_df.empty or not idf_df.empty or not grain_stats_multi_df.empty:
                with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                    if not audit_df.empty:
                        audit_df.to_excel(writer, index=False, sheet_name="Auditoria_QA")
                    if not sens_df.empty:
                        sens_df.to_excel(writer, index=False, sheet_name="Sensibilidad_Manning")
                    if not uncertainty_df.empty:
                        uncertainty_df.to_excel(writer, index=False, sheet_name="Incertidumbre_MC_v6")
                    if not calibration_df.empty:
                        calibration_df.to_excel(writer, index=False, sheet_name="Calibracion_v6")
                    if not confidence_df.empty:
                        confidence_df.to_excel(writer, index=False, sheet_name="Confianza_v6")
                    if not idf_df.empty:
                        idf_df.to_excel(writer, index=False, sheet_name="Curvas_IDF")
                    if not grain_stats_multi_df.empty:
                        grain_stats_multi_df.to_excel(writer, index=False, sheet_name="Granulometria_interp")
                    h03 = pd.DataFrame(st.session_state.get("hidrosed_secciones", []))
                    h04 = pd.DataFrame(st.session_state.get("hidrosed_puntos_seccion", []))
                    if not h03.empty:
                        h03.to_excel(writer, index=False, sheet_name="03_Secciones")
                    if not h04.empty:
                        h04.to_excel(writer, index=False, sheet_name="04_Puntos_Seccion")
            export_hidrosed_html(html_path, metadata, params, results_df, exner_df)
            payload = {
                "metadata": metadata,
                "global_params": params.__dict__,
                "sections": sections_df.to_dict("records"),
                "grain_curve": grain_df.to_dict("records"),
                "grain_stats": st.session_state.get("grain_stats"),
                "grain_stats_multi": st.session_state.get("grain_stats_multi"),
                "idf": st.session_state.get("idf_df"),
                "idf_parametros": st.session_state.get("idf_parametros"),
                "results": results_df.to_dict("records"),
                "audit_qa_v5": audit_df.to_dict("records"),
                "sensitivity_manning": sens_df.to_dict("records"),
                "uncertainty_mc_v6": uncertainty_df.to_dict("records"),
                "calibration_v6": calibration_df.to_dict("records"),
                "confidence_v6": confidence_df.to_dict("records"),
                "confidence_score": st.session_state.get("hidrosed_confidence_score"),
                "exner": exner_df.to_dict("records"),
            }
            json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            st.session_state["export_excel"] = str(excel_path)
            st.session_state["export_html"] = str(html_path)
            st.session_state["export_json"] = str(json_path)
            st.success("Archivos generados.")

    for key, label, fname, mime in [
        ("export_excel", "Descargar Excel HidroSed v6", "HidroSed_Cauces_resultados_v6.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("export_html", "Descargar reporte HTML v6", "Reporte_HidroSed_Cauces_v6.html", "text/html"),
        ("export_json", "Descargar proyecto JSON v6", "Proyecto_HidroSed_Cauces_v6.json", "application/json"),
        ("dem_path", "Descargar DEM GeoTIFF", "dem_cop30.tif", "image/tiff"),
        ("curvas_kmz_path", "Descargar curvas KMZ", "curvas_nivel.kmz", "application/vnd.google-earth.kmz"),
        ("curvas_maestras_kmz_path", "Descargar KMZ maestro DEM + topografía", "curvas_maestras_dem_topografia.kmz", "application/vnd.google-earth.kmz"),
        ("curvas_apoyo_kmz_path", "Descargar curvas topográficas de apoyo", "curvas_apoyo_topografia.kmz", "application/vnd.google-earth.kmz"),
        ("eje_cauce_kmz_path", "Descargar eje del cauce", "eje_cauce.kmz", "application/vnd.google-earth.kmz"),
    ]:
        path = st.session_state.get(key)
        if path and Path(path).exists():
            st.download_button(label, data=as_download_bytes(path), file_name=fname, mime=mime)


def main() -> None:
    init_state()
    page = sidebar()
    api_key = get_opentopo_key()
    if page.startswith("1"):
        page_dem(api_key)
    elif page.startswith("2"):
        page_contours()
    elif page.startswith("3"):
        page_eje_cauce()
    elif page.startswith("4"):
        page_integracion_secciones()
    elif page.startswith("5"):
        page_modelo_3d()
    elif page.startswith("6"):
        page_project_sections()
    elif page.startswith("7"):
        page_grain()
    elif page.startswith("8"):
        page_idf()
    elif page.startswith("9"):
        page_results()
    elif page.startswith("10"):
        page_exner()
    elif page.startswith("11"):
        page_report()


if __name__ == "__main__":
    main()
