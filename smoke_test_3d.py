from pathlib import Path
import pandas as pd

from modules_hidrosed.modulo_integracion_secciones_hidrosed import (
    generar_datos_demo_v13,
    validar_secciones_para_hidrosed,
    convertir_secciones_a_hidrosed,
    convertir_puntos_a_hidrosed,
)
from modules_hidrosed.visualizacion_3d import generar_figura_3d_cauce_secciones, exportar_modelo_3d_html


def main():
    data = generar_datos_demo_v13(n_secciones=6, puntos_por_seccion=13)
    errores, advertencias, df_validado = validar_secciones_para_hidrosed(data["df_secciones_validas"], data["df_puntos_secciones"])
    assert not errores, errores
    assert not df_validado.empty
    h03 = convertir_secciones_a_hidrosed(df_validado, caudal_default=50.0)
    h04 = convertir_puntos_a_hidrosed(data["df_puntos_secciones"], df_validado)
    for vista in ["Isométrica", "Planta / superior", "Lateral", "Aguas abajo", "Aguas arriba", "Rotación libre"]:
        fig = generar_figura_3d_cauce_secciones(
            h04,
            h03,
            max_sections=10,
            exageracion_vertical=2.0,
            vista_3d=vista,
            projection_type="orthographic" if vista == "Planta / superior" else "perspective",
        )
        assert len(fig.data) >= 3, f"Figura sin trazas suficientes: {len(fig.data)}"
        assert fig.layout.scene.camera is not None, f"Cámara no aplicada para {vista}"
    out = Path("outputs/test_modelo_3d_vistas.html")
    exportar_modelo_3d_html(fig, out)
    assert out.exists() and out.stat().st_size > 1000
    print("OK 3D vistas", len(fig.data), out)


if __name__ == "__main__":
    main()
