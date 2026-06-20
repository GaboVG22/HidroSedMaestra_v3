from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .hidrosed_core import GlobalParams, df_to_html_report


def export_hidrosed_excel(output_path: Path, metadata: dict[str, Any], params: GlobalParams, sections_df: pd.DataFrame, results_df: pd.DataFrame, grain_df: pd.DataFrame | None = None, exner_df: pd.DataFrame | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([{"campo": k, "valor": str(v)} for k, v in metadata.items()]).to_excel(writer, index=False, sheet_name="Resumen")
        pd.DataFrame([params.__dict__]).to_excel(writer, index=False, sheet_name="Parametros")
        sections_df.to_excel(writer, index=False, sheet_name="Secciones")
        if grain_df is not None and not grain_df.empty:
            grain_df.to_excel(writer, index=False, sheet_name="Granulometria")
        results_df.to_excel(writer, index=False, sheet_name="Resultados")
        if exner_df is not None and not exner_df.empty:
            exner_df.to_excel(writer, index=False, sheet_name="Lecho_movil")
    return output_path


def export_hidrosed_html(output_path: Path, metadata: dict[str, Any], params: GlobalParams, results_df: pd.DataFrame, exner_df: pd.DataFrame | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = df_to_html_report(metadata, params, results_df, exner_df)
    output_path.write_text(html, encoding="utf-8")
    return output_path
