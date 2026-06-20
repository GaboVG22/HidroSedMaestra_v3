"""Auditoría técnica de informes externos vs cálculo HidroSed.

Módulo agregado en v6.3 para incorporar observaciones detectadas en el informe
"Extracción Mecanizada de áridos Estero Punitaqui".

Objetivo:
- Reproducir la hidrología DGA-AC usada en el informe.
- Detectar diferencias de insumos, presets y denominaciones técnicas.
- Entregar una matriz de control trazable antes de emitir memoria final.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from io import BytesIO
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


PUNITAQUI_FACTORES_JP_MAX = {
    2: 0.30,
    5: 0.66,
    10: 1.00,
    20: 1.61,
    25: 1.85,
    50: 2.76,
    75: 3.42,
    100: 3.94,
}

PUNITAQUI_FACTORES_JP_MEDIA = {
    2: 0.24,
    5: 0.61,
    10: 1.00,
    20: 1.51,
    25: 1.71,
    50: 2.41,
    75: 2.91,
    100: 3.30,
}

PUNITAQUI_QMAX_INFORME = {
    2: 5.74,
    5: 12.63,
    10: 19.13,
    25: 35.40,
    50: 52.79,
    100: 75.37,
}


@dataclass
class AuditoriaPunitaquiInput:
    area_km2: float = 173.17
    p24_10_calculo_mm: float = 80.7
    p24_10_texto_mm: float = 120.0
    alpha_informe: float = 2.14
    alpha_app_anterior: float = 1.25
    comuna_portada: str = "Punitaqui"
    comuna_generalidades: str = "Salamanca"
    duracion_evento_sedimentos_h: float = 48.0
    geometria_digital_disponible: bool = False


def q10_dga_ac_iii_iv(area_km2: float, p24_10_mm: float, coef: float = 1.94e-7, area_exp: float = 0.776, p_exp: float = 3.108) -> float:
    """Caudal medio diario Q10 por fórmula DGA-AC III-IV regiones."""
    if area_km2 <= 0 or p24_10_mm <= 0:
        return float("nan")
    return float(coef * (area_km2 ** area_exp) * (p24_10_mm ** p_exp))


def dga_ac_instantaneous_flows(
    area_km2: float,
    p24_10_mm: float,
    alpha_inst: float,
    factors: Dict[int, float],
    report_q: Optional[Dict[int, float]] = None,
) -> pd.DataFrame:
    """Reproduce los caudales instantáneos DGA-AC por período de retorno."""
    q10 = q10_dga_ac_iii_iv(area_km2, p24_10_mm)
    rows = []
    for T in sorted(factors):
        f = float(factors[T])
        q = q10 * f * alpha_inst
        q_rep = np.nan if not report_q else report_q.get(int(T), np.nan)
        rows.append({
            "T_anios": int(T),
            "Q10_m3s": q10,
            "Factor_FT": f,
            "Alpha_inst": alpha_inst,
            "Q_app_m3s": q,
            "Q_informe_m3s": q_rep,
            "Diferencia_m3s": q - q_rep if pd.notna(q_rep) else np.nan,
            "Diferencia_pct": 100 * (q - q_rep) / q_rep if pd.notna(q_rep) and q_rep != 0 else np.nan,
        })
    return pd.DataFrame(rows)


def evaluar_observaciones_punitaqui(inp: AuditoriaPunitaquiInput | None = None) -> pd.DataFrame:
    """Genera matriz de observaciones detectadas para el informe Punitaqui."""
    inp = inp or AuditoriaPunitaquiInput()
    rows = []

    def add(item: str, informe: str, app: str, dictamen: str, accion: str, severidad: str = "Media"):
        rows.append({
            "Item": item,
            "Dato_en_informe": informe,
            "Dato_en_app_o_control": app,
            "Dictamen": dictamen,
            "Accion_en_aplicacion": accion,
            "Severidad": severidad,
        })

    # 1. P24 inconsistency
    if abs(inp.p24_10_texto_mm - inp.p24_10_calculo_mm) > 0.5:
        add(
            "P24,10 de diseño",
            f"Texto: {inp.p24_10_texto_mm:g} mm; cálculo/lámina: {inp.p24_10_calculo_mm:g} mm",
            "La app usa un campo separado para P24 textual y P24 de cálculo.",
            "Diferencia documental crítica: el caudal se reproduce con 80,7 mm, no con 120 mm.",
            "Agregar alerta automática cuando P24 textual difiere de P24 usado en cálculo.",
            "Alta",
        )

    # 2. alpha issue
    if abs(inp.alpha_informe - inp.alpha_app_anterior) > 1e-6:
        add(
            "Factor alfa DGA-AC zona Jp",
            f"Informe: α={inp.alpha_informe:g}",
            f"Preset anterior app: α={inp.alpha_app_anterior:g}; preset corregido v6.3: α={inp.alpha_informe:g}",
            "Error de preset corregido en la aplicación.",
            "Actualizar DGA_AC_PRESETS para Limarí Jp y dejar α editable con trazabilidad.",
            "Alta",
        )

    # 3. frequency factors issue
    add(
        "Curva regional DGA-AC Jp",
        "El informe calcula Qmax con la columna Máx. de Q(T)/Q10.",
        "v6.3 incorpora preset 'Jp Limarí - envolvente máxima informe Punitaqui' y conserva preset media regional.",
        "Sin este cambio, los caudales pueden subestimarse si se usa la serie media por defecto.",
        "Permitir seleccionar Media / Máx. / Mín. o editar factores por tabla.",
        "Alta",
    )

    # 4. commune inconsistency
    if inp.comuna_portada.strip().lower() != inp.comuna_generalidades.strip().lower():
        add(
            "Comuna del proyecto",
            f"Portada/objetivo: {inp.comuna_portada}; Generalidades: {inp.comuna_generalidades}",
            "La app agrega auditoría de coherencia territorial.",
            "Error documental del informe; no afecta el cálculo hidráulico, pero sí la trazabilidad administrativa.",
            "Mostrar advertencia y solicitar comuna oficial antes de emitir memoria final.",
            "Media",
        )

    # 5. sediment annual vs event duration
    if inp.duracion_evento_sedimentos_h and inp.duracion_evento_sedimentos_h < 365 * 24:
        add(
            "Arrastre de sedimentos",
            f"Se denomina 'anual' a un cálculo asociado a evento de {inp.duracion_evento_sedimentos_h:g} h.",
            "La app cambia la etiqueta a 'volumen equivalente de evento' cuando la duración es < 1 año.",
            "Error de denominación técnica si se presenta como volumen anual.",
            "Separar caudal sólido instantáneo, volumen de evento y volumen anualizado.",
            "Media",
        )

    # 6. geometry digital unavailable
    if not inp.geometria_digital_disponible:
        add(
            "Verificación hidráulica HEC-RAS / secciones",
            "Las secciones aparecen como imágenes/PDF; no como geometría digital trazable.",
            "La app exige KMZ/DXF/Excel de secciones o 04_Puntos_Seccion para recalcular.",
            "No se puede auditar completamente el eje hidráulico desde imágenes.",
            "Bloquear dictamen definitivo si no hay geometría digital suficiente.",
            "Alta",
        )

    return pd.DataFrame(rows)


def matriz_comparacion_punitaqui(inp: AuditoriaPunitaquiInput | None = None) -> Dict[str, pd.DataFrame]:
    """Entrega hojas de auditoría con observaciones y caudales reproducidos."""
    inp = inp or AuditoriaPunitaquiInput()
    df_obs = evaluar_observaciones_punitaqui(inp)
    df_qmax = dga_ac_instantaneous_flows(
        area_km2=inp.area_km2,
        p24_10_mm=inp.p24_10_calculo_mm,
        alpha_inst=inp.alpha_informe,
        factors=PUNITAQUI_FACTORES_JP_MAX,
        report_q=PUNITAQUI_QMAX_INFORME,
    )
    df_q_media = dga_ac_instantaneous_flows(
        area_km2=inp.area_km2,
        p24_10_mm=inp.p24_10_calculo_mm,
        alpha_inst=inp.alpha_informe,
        factors=PUNITAQUI_FACTORES_JP_MEDIA,
        report_q=None,
    )
    df_inputs = pd.DataFrame([asdict(inp)])
    return {
        "Observaciones": df_obs,
        "Qmax_Reproducido": df_qmax,
        "Q_DGA_Jp_Media": df_q_media,
        "Inputs_Auditoria": df_inputs,
    }


def excel_auditoria_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = name[:31]
            df.to_excel(writer, index=False, sheet_name=safe)
            ws = writer.sheets[safe]
            for col in ws.columns:
                max_len = 10
                col_letter = col[0].column_letter
                for cell in col:
                    val = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, min(len(val) + 2, 60))
                ws.column_dimensions[col_letter].width = max_len
    return bio.getvalue()
