"""Exportadores simples de resultados."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def export_project_excel(
    output_path: Path,
    metadata: dict[str, Any],
    flows_df: pd.DataFrame | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_df = pd.DataFrame([{"campo": k, "valor": str(v)} for k, v in metadata.items()])
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        meta_df.to_excel(writer, index=False, sheet_name="Resumen")
        if flows_df is not None:
            flows_df.to_excel(writer, index=False, sheet_name="Crecidas")
    return output_path
