"""Utilidades de sesión y archivos temporales."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import streamlit as st


OUTPUT_ROOT = Path("outputs")


def ensure_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = str(uuid.uuid4())
    return st.session_state["session_id"]


def get_session_folder() -> Path:
    sid = ensure_session_id()
    folder = OUTPUT_ROOT / sid
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_state_json(metadata: dict[str, Any], filename: str = "estado_proyecto.json") -> Path:
    folder = get_session_folder()
    path = folder / filename
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _state_has_value(value: Any) -> bool:
    """Evalúa si un valor de st.session_state representa un dato cargado.

    Evita el error clásico de Pandas:
    "The truth value of a DataFrame is ambiguous".
    """
    if value is None:
        return False

    # Pandas DataFrame / Series
    if hasattr(value, "empty"):
        try:
            return not bool(value.empty)
        except Exception:
            return True

    # Numpy arrays
    if hasattr(value, "size") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return int(value.size) > 0
        except Exception:
            return True

    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0

    if isinstance(value, (str, bytes, bytearray)):
        return len(value) > 0

    try:
        return bool(value)
    except Exception:
        # Si el objeto no puede evaluarse como booleano, pero existe,
        # se considera cargado para no botar la aplicación.
        return True


def status_badge(key: str, label: str) -> None:
    value = st.session_state.get(key, None)
    if _state_has_value(value):
        st.sidebar.success(f"✓ {label}")
    else:
        st.sidebar.warning(f"○ {label}")


def as_download_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()
