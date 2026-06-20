from pathlib import Path
import pandas as pd
from modules_hidrosed.granulometria_avanzada import parse_grain_text_multi, stats_by_sample, assign_grain_to_sections
from modules_hidrosed.idf_generator import generate_idf_power_law, generate_idf_from_p24, export_idf_excel
from modules_hidrosed.kmz_geometria import generate_preliminary_axis_from_point, parse_kml_geometries, summarize_geometries, write_kmz_from_kml


def main():
    # Granulometría multi-muestra + interpolación a secciones
    txt = """G1;0;0.5;5
G1;0;2;15
G1;0;8;40
G1;0;32;75
G1;0;90;100
G2;1000;0.5;3
G2;1000;2;10
G2;1000;16;45
G2;1000;64;90
G2;1000;128;100
"""
    g = parse_grain_text_multi(txt)
    stats = stats_by_sample(g)
    assert len(stats) == 2 and "D50_mm" in stats.columns and stats["D50_mm"].notna().all()
    sections = pd.DataFrame({"section_id":["S1","S2","S3"], "distance_m":[0,500,1000], "q_m3s":[10,10,10]})
    sec2 = assign_grain_to_sections(sections, stats)
    assert "d50_mm" in sec2.columns and sec2["d50_mm"].notna().all()

    # IDF
    idf1 = generate_idf_power_law(950, 0.18, 10, 0.75, [5,10,60], [2,10,100])
    idf2 = generate_idf_from_p24({2: 35, 100: 90}, [5,60,1440])
    assert not idf1.empty and not idf2.empty
    out = Path('outputs/test_idf_v6_1.xlsx')
    export_idf_excel(idf1, out)
    assert out.exists()

    # KML/KMZ eje preliminar
    kml = generate_preliminary_axis_from_point({"latitud": -29.9, "longitud": -71.25}, length_km=2, azimuth_deg=180)
    path = Path('outputs/test_eje_preliminar.kmz')
    write_kmz_from_kml(kml, path)
    geoms = parse_kml_geometries(path)
    summary = summarize_geometries(geoms, archivo='test_eje_preliminar.kmz')
    assert summary.n_lineas >= 1
    print('OK smoke v6.1 inputs/idf/grain')

if __name__ == '__main__':
    main()
