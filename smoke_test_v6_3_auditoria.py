from modules_hidrosed.auditoria_informe import (
    AuditoriaPunitaquiInput,
    matriz_comparacion_punitaqui,
    q10_dga_ac_iii_iv,
)
from src.hidro_kmz_core import DGA_AC_PRESETS

inp = AuditoriaPunitaquiInput()
sheets = matriz_comparacion_punitaqui(inp)
assert "Observaciones" in sheets and len(sheets["Observaciones"]) >= 5
q10 = q10_dga_ac_iii_iv(inp.area_km2, inp.p24_10_calculo_mm)
assert abs(q10 - 8.94) < 0.08, q10
df = sheets["Qmax_Reproducido"]
q100 = float(df[df["T_anios"] == 100]["Q_app_m3s"].iloc[0])
assert abs(q100 - 75.37) < 0.15, q100
assert DGA_AC_PRESETS["Zona Jp pluvial Limarí - envolvente máxima informe Punitaqui (editable)"]["alpha_inst"] == 2.14
assert DGA_AC_PRESETS["Zona Jp pluvial Limarí - envolvente máxima informe Punitaqui (editable)"]["factors"][100] == 3.94
print("OK v6.3 auditoria informe Punitaqui")
