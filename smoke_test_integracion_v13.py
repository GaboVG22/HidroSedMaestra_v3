from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules_hidrosed.modulo_integracion_secciones_hidrosed import (
    convertir_puntos_a_hidrosed,
    convertir_secciones_a_hidrosed,
    generar_datos_demo_v13,
    generar_excel_integracion_hidrosed,
    hidrosed_03_to_sections_df,
    validar_secciones_para_hidrosed,
)
from modules_hidrosed.hidrosed_core import dataframe_to_sections, compute_all, GlobalParams


def main():
    data = generar_datos_demo_v13(n_secciones=5, puntos_por_seccion=11)
    errores, advertencias, df_validado = validar_secciones_para_hidrosed(
        data["df_secciones_validas"], data["df_puntos_secciones"]
    )
    assert not errores, errores
    assert len(df_validado) == 5
    h03 = convertir_secciones_a_hidrosed(df_validado, caudal_default=75.0)
    h04 = convertir_puntos_a_hidrosed(data["df_puntos_secciones"], df_validado)
    assert list(h03.columns)[0] == "PK_m"
    assert len(h04) == 55
    sections_df = hidrosed_03_to_sections_df(h03)
    sections = dataframe_to_sections(sections_df)
    results = compute_all(sections, GlobalParams())
    assert not results.empty
    out = ROOT / "outputs" / "test_integracion_v13.xlsx"
    generar_excel_integracion_hidrosed(h03, h04, output_path=out)
    assert out.exists() and out.stat().st_size > 0
    print("OK integración v13 -> HidroSed", len(results), "secciones calculadas")


if __name__ == "__main__":
    main()
