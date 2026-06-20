
from __future__ import annotations

from pathlib import Path
import runpy

import streamlit as st

from modules_hidrosed.contour_generator import generate_contours_kmz
from modules_hidrosed.dem_downloader import (
    bbox_from_point,
    bbox_from_point_km,
    download_dem_cop30,
    download_dem_cop30_tiled,
    validate_geotiff,
)
from modules_hidrosed.kmz_reader import KMLParseError, read_control_point
from modules_hidrosed.watershed_delineation import delineate_watershed_from_dem
from modules_hidrosed.session_utils import as_download_bytes, get_session_folder, save_state_json, status_badge
from modules_master.kmz_workspace import create_combined_workspace_kmz

st.set_page_config(
    page_title="HidroSed Maestra Integrada v6.7",
    page_icon="🌊",
    layout="wide",
)

ROOT = Path(__file__).parent


def _get_opentopo_key() -> str:
    try:
        secret_key = st.secrets.get("OPENTOPO_API_KEY", "")
    except Exception:
        secret_key = ""
    return st.sidebar.text_input(
        "OpenTopography API Key",
        value=secret_key,
        type="password",
        help="En Streamlit Cloud guárdala en Secrets como OPENTOPO_API_KEY.",
    ).strip()


def _render_state_box() -> None:
    st.sidebar.subheader("Estado integrado")
    status_badge("control_point", "Punto de control")
    status_badge("dem_path", "DEM COP30")
    status_badge("basin_auto_kmz_path", "Cuenca delimitada")
    status_badge("curvas_kmz_path", "Curvas KMZ")
    status_badge("hidro_workspace_kmz_path", "KMZ combinado")
    status_badge("hidrosed_results", "HidroSed calculado")




def _auto_create_hydro_workspace_if_possible(project_name: str = "Proyecto HidroSed automático") -> Path | None:
    """Crea automáticamente el KMZ combinado si ya existen cuenca y curvas.

    Esto hace que la integración DEM → cuenca → curvas → hidrología sea realmente continua:
    el usuario no necesita descargar y recargar archivos entre módulos.
    """
    basin = st.session_state.get("basin_auto_kmz_path")
    contours = st.session_state.get("curvas_kml_path")
    if not basin or not contours:
        return None
    basin_path = Path(str(basin))
    contours_path = Path(str(contours))
    if not basin_path.exists() or not contours_path.exists():
        return None

    folder = get_session_folder()
    out_path = folder / "workspace_hidrologia_dem_cuenca_curvas.kmz"
    try:
        create_combined_workspace_kmz(basin_path, contours_path, out_path, project_name=project_name)
        st.session_state["hidro_workspace_kmz_path"] = str(out_path)
        st.session_state["hidro_workspace_source"] = "auto_dem_cuenca_curvas"
        return out_path
    except Exception as exc:
        st.warning(f"No se pudo crear automáticamente el KMZ combinado: {exc}")
        return None

def page_home() -> None:
    st.title("HidroSed Maestra Integrada v6.7")
    st.markdown(
        """
        Esta versión une tres flujos que antes estaban separados y agrega auditoría automática de informes externos:

        **1. Punto de control KMZ → 2. DEM COP30 → 3. delimitación automática de cuenca → 4. curvas DEM/topografía → 5. KMZ combinado automático → 6. hidrología/IDF → 7. secciones → 8. HidroSed Cauces.**

        Corrige la carga de curvas de nivel y el estado de sesión con DataFrames. Mantiene el módulo hidrológico avanzado de la versión 2.4 y agrega una etapa inicial DEM/curvas, además de una etapa HidroSed para hidráulica, transporte de sedimentos, socavación y lecho móvil conceptual.
        """
    )
    st.info(
        "La app ahora delimita la cuenca automáticamente desde el DEM y el punto de control, genera curvas de nivel y crea el KMZ combinado para el módulo hidrológico sin recarga manual. También permite reemplazar la cuenca automática por una cuenca validada externamente."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Entrada", "Punto KMZ/KML")
    c2.metric("Proceso", "DEM + curvas")
    c3.metric("Salida", "KMZ combinado")
    st.subheader("Uso recomendado")
    st.markdown(
        """
        1. Entra a **1 · DEM y curvas** y carga el KMZ/KML con el punto de control.
        2. Descarga DEM COP30 o usa mosaico para cuencas grandes.
        3. Delimita automáticamente la cuenca desde el DEM o carga una cuenca validada.
        4. Genera curvas de nivel.
        5. Opcionalmente carga curvas topográficas de apoyo y eje de cauce.
        6. Crea el **KMZ combinado** usando la cuenca automática o un KMZ/KML base validado.
        5. Entra a **2 · Hidrología KMZ** y activa **Usar KMZ combinado generado en etapa DEM/curvas**.
        6. Entra a **3 · HidroSed Maestra Integrada** para transferir secciones v13 a `03_Secciones` / `04_Puntos_Seccion`, revisar geometría y continuar con hidráulica, sedimentos y socavación.
        """
    )


def page_dem_curves(api_key: str) -> None:
    st.title("1 · DEM COP30 → Cuenca → Curvas → KMZ combinado")
    tab_dem, tab_basin, tab_curves, tab_workspace = st.tabs(["A. Descargar DEM", "B. Delimitar cuenca", "C. Generar curvas", "D. Armar KMZ combinado"])

    with tab_dem:
        c1, c2 = st.columns([1.1, 0.9])
        with c1:
            uploaded = st.file_uploader("KMZ/KML con punto de control", type=["kmz", "kml"], key="control_point_upload")
            margin_mode = st.radio("Tipo de margen", ["Grados", "Kilómetros"], horizontal=True, key="dem_margin_mode")
            if margin_mode == "Grados":
                margin = st.select_slider("Margen de descarga [°]", options=[0.10, 0.25, 0.50, 1.00, 1.50, 2.00], value=0.50)
            else:
                margin = st.select_slider("Margen de descarga [km]", options=[5, 10, 25, 50, 100, 150, 200], value=50)
            mode = st.radio("Modo DEM", ["Descarga única", "Mosaico para cuencas grandes"], horizontal=True)
            n_tiles = 10
            if mode.startswith("Mosaico"):
                n_tiles = st.slider("Número de DEM parciales", 10, 40, 10)
        with c2:
            st.info("Para cuencas grandes usa 10 a 40 DEM parciales. Para curvas estables: equidistancia 50–100 m y simplificación ≥60 m.")
            if st.session_state.get("control_point"):
                st.success("Punto registrado")
                st.json(st.session_state["control_point"])

        if uploaded and st.button("Leer punto de control", type="secondary"):
            try:
                cp = read_control_point(uploaded, filename=uploaded.name)
                st.session_state["control_point"] = {"nombre": cp.name, "latitud": cp.latitude, "longitud": cp.longitude, "altitud": cp.altitude, "archivo": uploaded.name}
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
                    st.success("DEM descargado y registrado internamente.")
                    st.json(meta)
                except Exception as exc:
                    st.error(f"No se pudo descargar o validar el DEM: {exc}")
        dem_path = st.session_state.get("dem_path")
        if dem_path and Path(dem_path).exists():
            st.download_button("Descargar DEM GeoTIFF", data=as_download_bytes(dem_path), file_name="dem_cop30.tif", mime="image/tiff")

    with tab_basin:
        st.subheader("Delimitación automática de cuenca desde DEM")
        dem_path = st.session_state.get("dem_path")
        cp = st.session_state.get("control_point")

        if not dem_path or not Path(dem_path).exists():
            st.warning("Primero descarga el DEM en la pestaña A.")
        elif not cp:
            st.warning("Primero lee el KMZ/KML del punto de control.")
        else:
            st.success("DEM y punto de control disponibles.")
            c1, c2 = st.columns([1, 1])
            with c1:
                snap_radius = st.select_slider(
                    "Radio de ajuste del punto al cauce [m]",
                    options=[100, 250, 500, 1000, 1500, 2500, 5000, 10000],
                    value=1500,
                    help="Si el punto no cae exactamente sobre la celda de mayor acumulación, se ajusta al cauce más cercano dentro de este radio.",
                )
                max_cells_ws = st.selectbox(
                    "Máximo de celdas para delimitación",
                    [500_000, 1_000_000, 1_500_000, 2_500_000, 4_000_000],
                    index=2,
                    format_func=lambda x: f"{x:,}".replace(",", "."),
                )
                simplify_ws = st.selectbox(
                    "Simplificación del polígono [m]",
                    [0, 30, 50, 80, 120, 200],
                    index=3,
                )
            with c2:
                st.info(
                    "El algoritmo usa relleno de depresiones, dirección de flujo D8, acumulación y ajuste del punto de salida. "
                    "Para diseño final, compara el resultado con cartografía/topografía o una cuenca validada."
                )
                st.write(f"Latitud punto: {cp['latitud']:.6f}")
                st.write(f"Longitud punto: {cp['longitud']:.6f}")

            if st.button("Delimitar cuenca automáticamente", type="primary"):
                folder = get_session_folder()
                progress = st.progress(0)
                status = st.empty()

                def cb(done, total, msg):
                    progress.progress(int((done / max(total, 1)) * 100))
                    status.info(msg)

                try:
                    result = delineate_watershed_from_dem(
                        dem_path=Path(dem_path),
                        outlet_lon=float(cp["longitud"]),
                        outlet_lat=float(cp["latitud"]),
                        output_dir=folder,
                        snap_radius_m=float(snap_radius),
                        max_cells=int(max_cells_ws),
                        simplify_m=float(simplify_ws),
                        project_name="Cuenca delimitada HidroSed",
                        progress_callback=cb,
                    )
                    st.session_state["basin_auto_kmz_path"] = str(result.kmz_path)
                    st.session_state["basin_auto_kml_path"] = str(result.kml_path)
                    st.session_state["basin_auto_metadata"] = {
                        "area_km2": result.area_km2,
                        "perimetro_km": result.perimeter_km,
                        "punto_original_lon": result.outlet_original_lon,
                        "punto_original_lat": result.outlet_original_lat,
                        "punto_ajustado_lon": result.outlet_snapped_lon,
                        "punto_ajustado_lat": result.outlet_snapped_lat,
                        "distancia_ajuste_m": result.snapped_distance_m,
                        "celdas_cuenca": result.n_cells_basin,
                        "tamano_celda_m": result.cell_size_m,
                        "acumulacion_salida_celdas": result.accumulation_at_outlet_cells,
                        "advertencias": result.quality_flags,
                    }
                    st.success("Cuenca delimitada y registrada internamente.")
                    st.json(st.session_state["basin_auto_metadata"])
                    if result.preview_png and Path(result.preview_png).exists():
                        st.image(str(result.preview_png), caption="Vista de control: cuenca delimitada y red de acumulación", use_container_width=True)
                    if result.quality_flags:
                        st.warning("Advertencias de calidad:")
                        for flag in result.quality_flags:
                            st.write(f"- {flag}")

                    auto_workspace = _auto_create_hydro_workspace_if_possible("Proyecto HidroSed automático")
                    if auto_workspace:
                        st.success("KMZ combinado creado automáticamente para el módulo hidrológico.")
                        st.code(str(auto_workspace))
                except Exception as exc:
                    st.error(f"No se pudo delimitar correctamente la cuenca: {exc}")
                    st.info("Prueba aumentar el margen del DEM, aumentar el radio de ajuste del punto de salida o cargar un KMZ de cuenca validado manualmente.")

            basin_auto = st.session_state.get("basin_auto_kmz_path")
            if basin_auto and Path(basin_auto).exists():
                st.download_button(
                    "Descargar cuenca delimitada KMZ",
                    data=as_download_bytes(basin_auto),
                    file_name="cuenca_delimitada_automatica.kmz",
                    mime="application/vnd.google-earth.kmz",
                )

    with tab_curves:
        dem_path = st.session_state.get("dem_path")
        if not dem_path or not Path(dem_path).exists():
            st.warning("Primero descarga el DEM en la pestaña A.")
        else:
            st.success("DEM disponible internamente. No necesitas cargarlo nuevamente.")
            st.code(dem_path)
            c1, c2 = st.columns([1, 1])
            with c1:
                interval = st.selectbox("Equidistancia [m]", [10, 25, 50, 100, 200], index=2)
                simplify = st.selectbox("Simplificación [m]", [0, 30, 60, 100, 150, 250], index=2)
                max_cells = st.selectbox("Máximo de celdas procesadas", [1_000_000, 2_000_000, 4_000_000, 8_000_000], index=2, format_func=lambda x: f"{x:,}".replace(",", "."))
            with c2:
                st.info("La app decima internamente DEM grandes para generar curvas estables en Streamlit Cloud.")
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
                    st.session_state["curvas_metadata"] = {"n_lineas": result.n_lines, "z_min_m": result.min_elevation, "z_max_m": result.max_elevation, "equidistancia_m": result.interval}
                    st.success(f"Curvas generadas: {result.n_lines:,} líneas".replace(",", "."))
                    st.json(st.session_state["curvas_metadata"])
                    if result.preview_png and result.preview_png.exists():
                        st.image(str(result.preview_png), caption="Vista preliminar")

                    auto_workspace = _auto_create_hydro_workspace_if_possible("Proyecto HidroSed automático")
                    if auto_workspace:
                        st.success("KMZ combinado creado automáticamente para el módulo hidrológico.")
                        st.code(str(auto_workspace))
                except Exception as exc:
                    st.error(f"No se pudieron generar curvas: {exc}")
            curvas_path = st.session_state.get("curvas_kmz_path")
            if curvas_path and Path(curvas_path).exists():
                st.download_button("Descargar curvas KMZ", data=as_download_bytes(curvas_path), file_name="curvas_nivel.kmz", mime="application/vnd.google-earth.kmz")

    with tab_workspace:
        st.subheader("Crear KMZ hidrológico combinado")
        st.write("Puedes usar la cuenca delimitada automáticamente o cargar un KMZ/KML base validado con polígono de cuenca. La app lo combinará con las curvas generadas desde DEM.")
        basin_source_mode = st.radio(
            "Fuente del polígono de cuenca",
            ["Usar cuenca delimitada automáticamente", "Cargar KMZ/KML de cuenca validada"],
            horizontal=True,
        )
        basin_base = None
        if basin_source_mode.startswith("Usar"):
            basin_auto = st.session_state.get("basin_auto_kmz_path")
            if basin_auto and Path(basin_auto).exists():
                basin_base = Path(basin_auto)
                st.success("Cuenca automática disponible para combinar.")
                if st.session_state.get("basin_auto_metadata"):
                    st.json(st.session_state["basin_auto_metadata"])
            else:
                st.warning("Primero delimita la cuenca en la pestaña B o cambia a carga manual.")
        else:
            basin_base = st.file_uploader("KMZ/KML base con polígono de cuenca", type=["kmz", "kml"], key="basin_base_for_workspace")

        project_name = st.text_input("Nombre del proyecto para KMZ combinado", value="Proyecto HidroSed Hidrología")
        curvas_kml_path = st.session_state.get("curvas_kml_path")
        if not curvas_kml_path or not Path(curvas_kml_path).exists():
            st.warning("Primero genera curvas de nivel en la pestaña C.")
        if basin_base is not None and curvas_kml_path and Path(curvas_kml_path).exists():
            if st.button("Crear KMZ combinado para módulo hidrológico", type="primary"):
                try:
                    folder = get_session_folder()
                    out_path = folder / "workspace_hidrologia_dem_curvas.kmz"
                    create_combined_workspace_kmz(basin_base, Path(curvas_kml_path), out_path, project_name=project_name)
                    st.session_state["hidro_workspace_kmz_path"] = str(out_path)
                    st.success("KMZ combinado creado. Ahora entra a '2 · Hidrología KMZ' y usa la opción de KMZ combinado.")
                except Exception as exc:
                    st.error(f"No se pudo crear el KMZ combinado: {exc}")
        workspace = st.session_state.get("hidro_workspace_kmz_path")
        if workspace and Path(workspace).exists():
            st.download_button("Descargar KMZ combinado", data=as_download_bytes(workspace), file_name="workspace_hidrologia_dem_curvas.kmz", mime="application/vnd.google-earth.kmz")


def page_auditoria_informe() -> None:
    st.title("4 · Auditoría de informe y diferencias de cálculo")
    st.caption("Control incorporado a partir de las observaciones detectadas en el informe de extracción Estero Punitaqui.")

    from modules_hidrosed.auditoria_informe import (
        AuditoriaPunitaquiInput,
        matriz_comparacion_punitaqui,
        excel_auditoria_bytes,
    )

    st.markdown(
        """
        Este módulo no reemplaza la revisión técnica completa del expediente: permite **comparar automáticamente** los datos clave del informe con los cálculos internos de la aplicación.

        Controles incorporados:
        - diferencia entre **P24 textual** y **P24 usado en cálculo**;
        - factor **α DGA-AC**;
        - selección de curva regional **Jp Media / Máx. / editable**;
        - coherencia territorial del proyecto;
        - diferenciación entre **volumen de evento** y **volumen anual** de sedimentos;
        - bloqueo de dictamen definitivo si no existe geometría digital de secciones.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.number_input("Área cuenca [km²]", min_value=0.001, value=173.17, step=0.01, format="%.3f")
        p24_calc = st.number_input("P24,10 usado en cálculo [mm]", min_value=0.0, value=80.7, step=0.1, format="%.2f")
        p24_text = st.number_input("P24,10 declarado en texto [mm]", min_value=0.0, value=120.0, step=0.1, format="%.2f")
    with col2:
        alpha_inf = st.number_input("α informe / adoptado", min_value=0.0, value=2.14, step=0.01, format="%.3f")
        alpha_ant = st.number_input("α app anterior / control", min_value=0.0, value=1.25, step=0.01, format="%.3f")
        dur_sed = st.number_input("Duración usada para sedimentos [h]", min_value=0.0, value=48.0, step=1.0, format="%.1f")
    with col3:
        comuna_portada = st.text_input("Comuna en portada/objetivo", value="Punitaqui")
        comuna_general = st.text_input("Comuna en generalidades", value="Salamanca")
        geom_digital = st.checkbox("Existe geometría digital verificable de secciones", value=False)

    inp = AuditoriaPunitaquiInput(
        area_km2=float(area),
        p24_10_calculo_mm=float(p24_calc),
        p24_10_texto_mm=float(p24_text),
        alpha_informe=float(alpha_inf),
        alpha_app_anterior=float(alpha_ant),
        comuna_portada=comuna_portada,
        comuna_generalidades=comuna_general,
        duracion_evento_sedimentos_h=float(dur_sed),
        geometria_digital_disponible=bool(geom_digital),
    )

    sheets = matriz_comparacion_punitaqui(inp)

    st.subheader("Matriz de observaciones incorporadas")
    st.dataframe(sheets["Observaciones"], use_container_width=True)

    st.subheader("Reproducción DGA-AC del informe")
    st.info("La comparación usa la serie Máx. Q(T)/Q10 y α=2,14 para reproducir el informe Punitaqui. La app v6.7 deja este preset disponible y editable.")
    st.dataframe(sheets["Qmax_Reproducido"], use_container_width=True)

    with st.expander("Comparación con serie media regional Jp", expanded=False):
        st.dataframe(sheets["Q_DGA_Jp_Media"], use_container_width=True)

    xls = excel_auditoria_bytes(sheets)
    st.download_button(
        "Descargar auditoría Excel",
        data=xls,
        file_name="Auditoria_Informe_Punitaqui_HidroSed_v6_7.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def page_galeria_resultados() -> None:
    st.title("4 · Galería de imágenes y gráficos")
    st.caption("Ventana de revisión visual con lista desplegable de imágenes y gráficos relevantes del proyecto.")

    gallery = {
        "Resultados de socavación": {
            "path": ROOT / "assets" / "resultados_demo" / "resultado_socavacion.png",
            "caption": "Vista de resultados de socavación general y local para una sección seleccionada.",
            "detail": [
                "Muestra la sección transversal evaluada y la lámina de agua.",
                "Resalta fondo socavado, socavación general y socavación local.",
                "Útil para revisar profundidad máxima, estado y trazabilidad del cálculo.",
            ],
        },
        "Transporte de sedimentos": {
            "path": ROOT / "assets" / "resultados_demo" / "transporte_sedimentos.png",
            "caption": "Vista de transporte de sedimentos con perfil longitudinal, capacidad y tendencia erosión/deposición.",
            "detail": [
                "Incluye indicadores globales: capacidad de transporte, carga de fondo, velocidad y tensión de corte.",
                "Resume tendencias de erosión, equilibrio y deposición por tramo.",
                "Permite comparar tramos representativos y periodos de retorno.",
            ],
        },
        "Modelo 3D del cauce": {
            "path": ROOT / "assets" / "resultados_demo" / "modelo_3d_cauce.png",
            "caption": "Vista 3D del cauce y secciones con terreno, lámina de agua y eje principal.",
            "detail": [
                "Integra el eje del cauce, las secciones transversales y la topografía circundante.",
                "Permite comunicar visualmente la geometría del modelo y la lámina de agua.",
                "Sirve como referencia para exportación HTML 3D y capturas del proyecto.",
            ],
        },
    }

    c1, c2 = st.columns([0.35, 0.65])
    with c1:
        st.subheader("Selector visual")
        selected = st.selectbox("Imagen / gráfico", list(gallery.keys()), index=0)
        st.markdown("**Elementos disponibles**")
        for name in gallery.keys():
            st.write(f"• {name}")
        st.info("La app puede usar esta misma ventana como galería de resultados, incorporando capturas generadas por el proyecto y gráficos exportados.")
    with c2:
        item = gallery[selected]
        st.subheader(selected)
        if item["path"].exists():
            st.image(str(item["path"]), caption=item["caption"], use_container_width=True)
        else:
            st.warning("La imagen seleccionada no está disponible en la carpeta de assets.")
        st.markdown("**Qué muestra esta vista**")
        for bullet in item["detail"]:
            st.write(f"- {bullet}")
        with open(item["path"], "rb") as f:
            data = f.read()
        st.download_button(
            "Descargar imagen seleccionada",
            data=data,
            file_name=item["path"].name,
            mime="image/png",
        )

    st.subheader("Diseño recomendado para resultados")
    st.markdown(
        """
        - **Lista desplegable** para elegir la figura a mostrar.
        - **Ventana central de visualización** para imagen/gráfico seleccionado.
        - **Leyenda o resumen lateral** con interpretación técnica.
        - **Descarga directa** de la imagen y futura integración con el informe final.

        Estas tres imágenes ya quedan incorporadas como resultados de referencia dentro de la plataforma.
        """
    )


def page_guide() -> None:
    st.title("Guía técnica de integración")
    st.markdown(
        """
        ### Qué se perfeccionó
        - Etapa inicial **DEM COP30 → curvas de nivel**.
        - **KMZ combinado**: polígono de cuenca + curvas DEM.
        - El módulo hidrológico v2.4 puede tomar automáticamente el KMZ combinado desde `st.session_state`.
        - Corrección de mutación del área real de cuenca cuando se usa área pluvial efectiva.
        - Reemplazo visual por `use_container_width=True`.
        - Integración de **HidroSed Cauces** con secciones, granulometría, transporte MPM, socavación LL y Exner conceptual.
        - Nuevo módulo **v13 → HidroSed** con validación de secciones, transferencia automática a `03_Secciones` / `04_Puntos_Seccion` y trazabilidad de descartadas.
        - Nuevo módulo **Auditoría de informe** para comparar cálculos del informe Punitaqui con la app, corregir preset DGA-AC Jp y documentar inconsistencias.

        **Main file path:** `app.py`

        En Streamlit Cloud agrega en Secrets:
        ```toml
        OPENTOPO_API_KEY = "TU_API_KEY_DE_OPENTOPOGRAPHY"
        ```
        """
    )


def main() -> None:
    st.sidebar.title("Aplicación Maestra")
    page = st.sidebar.radio("Módulos", ["0 · Inicio", "1 · DEM y curvas", "2 · Hidrología KMZ", "3 · HidroSed Maestra Integrada", "4 · Galería de resultados", "5 · Auditoría de informe", "6 · Guía técnica"])
    api_key = _get_opentopo_key()
    st.sidebar.divider()
    _render_state_box()
    st.sidebar.caption("Main file path: app.py")
    if page.startswith("0"):
        page_home()
    elif page.startswith("1"):
        page_dem_curves(api_key)
    elif page.startswith("2"):
        runpy.run_path(str(ROOT / "app_hidrologia_kmz.py"), run_name="__main__")
    elif page.startswith("3"):
        runpy.run_path(str(ROOT / "app_hidrosed_cauces.py"), run_name="__main__")
    elif page.startswith("4"):
        page_galeria_resultados()
    elif page.startswith("5"):
        page_auditoria_informe()
    else:
        page_guide()


if __name__ == "__main__":
    main()
