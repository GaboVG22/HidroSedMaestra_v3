from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


TARGET_DIAMETERS = [5, 10, 16, 25, 30, 35, 50, 60, 65, 75, 84, 90, 95]


def _clean_num(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        if isinstance(x, str):
            x = x.replace(',', '.').strip()
            if x == '':
                return default
        return float(x)
    except Exception:
        return default


def normalize_grain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tablas con columnas flexibles: muestra/pk/diametro/%pasa."""
    if df is None or df.empty:
        raise ValueError('La tabla granulométrica está vacía.')
    cols = {c.lower().strip(): c for c in df.columns}
    def find(cands):
        for cand in cands:
            for low, orig in cols.items():
                if cand in low:
                    return orig
        return None
    col_d = find(['diametro', 'diámetro', 'd_mm', 'tamiz', 'mm'])
    col_p = find(['porcentaje_pasa', '% pasa', 'pasa', 'pasante', 'porc'])
    col_id = find(['id_muestra', 'muestra', 'id granulometria', 'id_granulometria', 'granulometria'])
    col_pk = find(['pk_m', 'progresiva', 'distancia', 'km', 'pk'])
    if col_d is None or col_p is None:
        raise ValueError('La tabla debe contener diámetro_mm y porcentaje_pasa.')
    out = pd.DataFrame()
    out['id_muestra'] = df[col_id].astype(str) if col_id else 'G1'
    out['PK_m'] = df[col_pk].map(_clean_num) if col_pk else np.nan
    out['diametro_mm'] = df[col_d].map(_clean_num)
    out['porcentaje_pasa'] = df[col_p].map(_clean_num)
    out = out.dropna(subset=['diametro_mm', 'porcentaje_pasa'])
    out = out[(out['diametro_mm'] > 0) & (out['porcentaje_pasa'] >= 0) & (out['porcentaje_pasa'] <= 100)]
    if out.empty:
        raise ValueError('No se encontraron pares diámetro/%pasa válidos.')
    return out.sort_values(['id_muestra', 'porcentaje_pasa', 'diametro_mm']).reset_index(drop=True)


def parse_grain_text_multi(text: str) -> pd.DataFrame:
    """Admite: id_muestra;PK_m;diametro_mm;porcentaje_pasa o diametro_mm;porcentaje_pasa."""
    rows = []
    current_id = 'G1'
    current_pk = np.nan
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        line = line.replace('\t', ';')
        parts = [p.strip() for p in line.split(';')]
        # ignora encabezados
        if any(word in line.lower() for word in ['diametro', 'diámetro', 'porcentaje', 'pasa']) and not any(ch.isdigit() for ch in line):
            continue
        if len(parts) >= 4:
            sid, pk, d, p = parts[0], _clean_num(parts[1]), _clean_num(parts[2]), _clean_num(parts[3])
            current_id, current_pk = sid or current_id, pk
        elif len(parts) >= 2:
            sid, pk = current_id, current_pk
            d, p = _clean_num(parts[0]), _clean_num(parts[1])
        else:
            continue
        if not np.isnan(d) and not np.isnan(p) and d > 0 and 0 <= p <= 100:
            rows.append({'id_muestra': str(sid), 'PK_m': pk, 'diametro_mm': d, 'porcentaje_pasa': p})
    if not rows:
        raise ValueError('No se pudo leer granulometría válida.')
    return normalize_grain_table(pd.DataFrame(rows))


def interpolate_dp(curve_df: pd.DataFrame, p: float) -> float:
    df = curve_df.dropna(subset=['diametro_mm', 'porcentaje_pasa']).sort_values('porcentaje_pasa')
    df = df.drop_duplicates(subset=['porcentaje_pasa'], keep='last')
    perc = df['porcentaje_pasa'].to_numpy(float)
    diam = df['diametro_mm'].to_numpy(float)
    if len(df) < 2:
        return float(diam[0]) if len(diam) else math.nan
    if p <= perc.min():
        return float(diam[0])
    if p >= perc.max():
        return float(diam[-1])
    i = np.searchsorted(perc, p)
    p1, p2 = perc[i-1], perc[i]
    d1, d2 = max(diam[i-1], 1e-9), max(diam[i], 1e-9)
    t = (p - p1) / max(p2 - p1, 1e-9)
    return float(10 ** (math.log10(d1) + t * (math.log10(d2) - math.log10(d1))))


def stats_by_sample(grain_df: pd.DataFrame, targets: Iterable[int] = TARGET_DIAMETERS) -> pd.DataFrame:
    df = normalize_grain_table(grain_df)
    rows = []
    for sid, grp in df.groupby('id_muestra'):
        row = {'id_muestra': sid}
        pks = pd.to_numeric(grp['PK_m'], errors='coerce').dropna()
        row['PK_m'] = float(pks.median()) if len(pks) else np.nan
        for p in targets:
            row[f'D{int(p)}_mm'] = interpolate_dp(grp, float(p))
        d10 = row.get('D10_mm', np.nan); d30 = row.get('D30_mm', np.nan); d50 = row.get('D50_mm', np.nan); d60 = row.get('D60_mm', np.nan)
        row['Cu'] = d60 / d10 if d10 and d10 > 0 else np.nan
        row['Cc'] = (d30 ** 2) / (d10 * d60) if d10 and d60 and d10 > 0 and d60 > 0 else np.nan
        row['Dm_mm'] = d50
        rows.append(row)
    return pd.DataFrame(rows).sort_values('PK_m', na_position='last').reset_index(drop=True)



# -----------------------------------------------------------------------------
# Perfiles granulométricos estándar de respaldo
# -----------------------------------------------------------------------------
# Estos perfiles NO reemplazan una granulometría de terreno. Están pensados como
# insumo conservador de prediagnóstico cuando el usuario aún no dispone de datos.
# Las clases se basan en la escala Wentworth/Udden-Wentworth: arcilla, limo,
# arena y grava por rangos de tamaño; las curvas son distribuciones sintéticas
# representativas para alimentar los modelos de transporte y socavación.

STANDARD_GRAIN_PROFILE_METADATA = {
    "referencia_clasificacion": "Escala Udden-Wentworth / Wentworth: arena ~0.0625-2 mm; grava >=2 mm.",
    "advertencia": "Perfil sintético para prediseño. Para diseño definitivo requiere muestreo de lecho y verificación en terreno.",
    "fuente_calculo": "Perfil estándar HidroSed v6.2 generado por curva acumulada representativa.",
}

# Formato: lista de pares (diametro_mm, porcentaje_pasa). Se usan rangos amplios
# y monotónicos para que la interpolación logarítmica derive D10, D50, D84, D90.
STANDARD_GRAIN_PROFILES = {
    "arena_fina": {
        "nombre": "Arena fina / lecho arenoso fino",
        "descripcion": "Uso preliminar en canales arenosos finos; no apto para cohesivos dominantes.",
        "d50_objetivo_mm": 0.22,
        "clase_wentworth": "arena fina",
        "curva": [(0.031, 0), (0.063, 5), (0.125, 25), (0.25, 60), (0.50, 88), (1.0, 98), (2.0, 100)],
    },
    "arena_media": {
        "nombre": "Arena media",
        "descripcion": "Perfil arenoso no cohesivo de energía baja a media.",
        "d50_objetivo_mm": 0.50,
        "clase_wentworth": "arena media",
        "curva": [(0.063, 0), (0.125, 7), (0.25, 28), (0.50, 55), (1.0, 86), (2.0, 98), (4.0, 100)],
    },
    "arena_gruesa": {
        "nombre": "Arena gruesa con gravilla",
        "descripcion": "Perfil de cauce arenoso grueso con presencia menor de grava fina.",
        "d50_objetivo_mm": 1.20,
        "clase_wentworth": "arena gruesa / grava muy fina",
        "curva": [(0.125, 0), (0.25, 5), (0.50, 18), (1.0, 44), (2.0, 72), (4.0, 92), (8.0, 100)],
    },
    "grava_fina": {
        "nombre": "Grava fina aluvial",
        "descripcion": "Perfil típico de cauce aluvial de grava fina y arena gruesa subordinada.",
        "d50_objetivo_mm": 8.0,
        "clase_wentworth": "grava fina",
        "curva": [(0.5, 0), (1.0, 4), (2.0, 10), (4.0, 28), (8.0, 52), (16.0, 76), (32.0, 94), (64.0, 100)],
    },
    "aluvial_semiarido_mixto": {
        "nombre": "Aluvial semiárido mixto: arenas + gravas",
        "descripcion": "Perfil por defecto recomendado para prediseño en cauces torrenciales/aluviales semiáridos cuando no existe muestreo.",
        "d50_objetivo_mm": 28.0,
        "clase_wentworth": "grava media con matriz arenosa",
        "curva": [(0.5, 0), (1.0, 2), (2.0, 6), (4.0, 12), (8.0, 24), (16.0, 42), (32.0, 58), (64.0, 78), (128.0, 94), (200.0, 100)],
    },
    "grava_media": {
        "nombre": "Grava media",
        "descripcion": "Perfil granular para cauces de grava de energía media.",
        "d50_objetivo_mm": 32.0,
        "clase_wentworth": "grava media",
        "curva": [(1.0, 0), (2.0, 4), (4.0, 10), (8.0, 24), (16.0, 40), (32.0, 56), (64.0, 78), (128.0, 94), (256.0, 100)],
    },
    "grava_gruesa_bolones": {
        "nombre": "Grava gruesa / bolones menores",
        "descripcion": "Perfil para cauces de alta energía con material grueso; revisar estabilidad y representatividad.",
        "d50_objetivo_mm": 75.0,
        "clase_wentworth": "grava gruesa a bolones pequeños",
        "curva": [(2.0, 0), (4.0, 3), (8.0, 8), (16.0, 18), (32.0, 34), (64.0, 48), (128.0, 72), (256.0, 92), (400.0, 100)],
    },
}

DEFAULT_STANDARD_PROFILE_KEY = "aluvial_semiarido_mixto"


def standard_profile_options() -> list[tuple[str, str]]:
    """Devuelve opciones (key, nombre) para UI."""
    return [(k, v["nombre"]) for k, v in STANDARD_GRAIN_PROFILES.items()]


def get_standard_grain_profile(profile_key: str = DEFAULT_STANDARD_PROFILE_KEY, pk_m: float | None = None, sample_id: str | None = None) -> pd.DataFrame:
    """Retorna curva granulométrica estándar en formato normalizado."""
    key = profile_key if profile_key in STANDARD_GRAIN_PROFILES else DEFAULT_STANDARD_PROFILE_KEY
    prof = STANDARD_GRAIN_PROFILES[key]
    sid = sample_id or key
    rows = []
    for d, p in prof["curva"]:
        rows.append({
            "id_muestra": sid,
            "PK_m": np.nan if pk_m is None else float(pk_m),
            "diametro_mm": float(d),
            "porcentaje_pasa": float(p),
            "perfil_estandar": key,
            "fuente_granulometria": "perfil_estandar_hidrosed_v6_2",
        })
    return normalize_grain_table(pd.DataFrame(rows))


def get_standard_grain_stats(profile_key: str = DEFAULT_STANDARD_PROFILE_KEY, pk_m: float | None = None) -> pd.DataFrame:
    """Calcula D característicos del perfil estándar."""
    df = get_standard_grain_profile(profile_key=profile_key, pk_m=pk_m)
    stats = stats_by_sample(df)
    key = profile_key if profile_key in STANDARD_GRAIN_PROFILES else DEFAULT_STANDARD_PROFILE_KEY
    stats["perfil_estandar"] = key
    stats["Nombre_perfil"] = STANDARD_GRAIN_PROFILES[key]["nombre"]
    stats["Clase_Wentworth"] = STANDARD_GRAIN_PROFILES[key]["clase_wentworth"]
    stats["Fuente_granulometria"] = "Perfil estándar HidroSed v6.2"
    stats["Advertencia_granulometria"] = STANDARD_GRAIN_PROFILE_METADATA["advertencia"]
    return stats


def recommend_standard_profile_by_context(slope: float | None = None, stream_power_hint: str | None = None) -> str:
    """Recomendación simple de perfil estándar por pendiente o pista cualitativa."""
    hint = (stream_power_hint or "").lower()
    if any(w in hint for w in ["arena fina", "fino", "limo", "baja energia", "baja energía"]):
        return "arena_fina"
    if "arena" in hint and "grues" not in hint:
        return "arena_media"
    if "arena grues" in hint or "gravilla" in hint:
        return "arena_gruesa"
    if "grava grues" in hint or "bolon" in hint or "bolón" in hint or "torrencial" in hint:
        return "grava_gruesa_bolones"
    try:
        s = float(slope)
    except Exception:
        s = math.nan
    if math.isfinite(s):
        if s < 0.001:
            return "arena_media"
        if s < 0.004:
            return "arena_gruesa"
        if s < 0.012:
            return "grava_fina"
        if s < 0.035:
            return "aluvial_semiarido_mixto"
        return "grava_gruesa_bolones"
    return DEFAULT_STANDARD_PROFILE_KEY


def build_default_grain_session_payload(profile_key: str = DEFAULT_STANDARD_PROFILE_KEY) -> dict:
    """Construye payload reutilizable para session_state cuando no hay granulometría de terreno."""
    curve = get_standard_grain_profile(profile_key)
    stats = get_standard_grain_stats(profile_key)
    first = stats.iloc[0].to_dict()
    first["Fuente_granulometria"] = "Perfil estándar HidroSed v6.2"
    first["Es_granulometria_default"] = True
    return {
        "grain_df": curve.to_dict("records"),
        "grain_stats_multi": stats.to_dict("records"),
        "grain_stats": first,
        "grain_default_profile_active": True,
        "grain_default_profile_key": profile_key if profile_key in STANDARD_GRAIN_PROFILES else DEFAULT_STANDARD_PROFILE_KEY,
        "grain_source_status": "SIN_MUESTREO_USA_PERFIL_ESTANDAR",
    }

def assign_grain_to_sections(sections_df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Asigna/interpola D característicos a secciones por PK/distancia. Si no hay PK usa primera muestra."""
    if sections_df is None or sections_df.empty or stats_df is None or stats_df.empty:
        return sections_df
    out = sections_df.copy()
    # Detectar columna de distancia/PK
    pk_col = None
    for cand in ['PK_m', 'distance_m', 'Distancia_m', 'Distancia_Transversal_m']:
        if cand in out.columns:
            pk_col = cand; break
    dcols = [c for c in stats_df.columns if c.startswith('D') and c.endswith('_mm')] + ['Dm_mm', 'Cu', 'Cc']
    s = stats_df.copy()
    if 'PK_m' in s.columns and pd.to_numeric(s['PK_m'], errors='coerce').notna().sum() >= 2 and pk_col is not None:
        s = s.dropna(subset=['PK_m']).sort_values('PK_m')
        x = s['PK_m'].to_numpy(float)
        xsec = pd.to_numeric(out[pk_col], errors='coerce').to_numpy(float)
        for c in dcols:
            if c not in s.columns: continue
            y = pd.to_numeric(s[c], errors='coerce').to_numpy(float)
            mask = np.isfinite(y)
            if mask.sum() >= 2:
                out[c.lower() if c in ['D50_mm','D84_mm','D90_mm','Dm_mm'] else c] = np.interp(xsec, x[mask], y[mask], left=y[mask][0], right=y[mask][-1])
                # compatibilidad dataclass lower
                if c == 'D50_mm': out['d50_mm'] = np.interp(xsec, x[mask], y[mask], left=y[mask][0], right=y[mask][-1])
                if c == 'D84_mm': out['d84_mm'] = np.interp(xsec, x[mask], y[mask], left=y[mask][0], right=y[mask][-1])
                if c == 'D90_mm': out['d90_mm'] = np.interp(xsec, x[mask], y[mask], left=y[mask][0], right=y[mask][-1])
                if c == 'Dm_mm': out['dm_mm'] = np.interp(xsec, x[mask], y[mask], left=y[mask][0], right=y[mask][-1])
    else:
        row = stats_df.iloc[0]
        for c in dcols:
            if c in row:
                val = _clean_num(row[c])
                out[c] = val
                if c == 'D50_mm': out['d50_mm'] = val
                if c == 'D84_mm': out['d84_mm'] = val
                if c == 'D90_mm': out['d90_mm'] = val
                if c == 'Dm_mm': out['dm_mm'] = val
    return out
