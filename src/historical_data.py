"""Utilities for regional historical DGA-style datasets.

The app uses compact ZIP+CSV/TXT files placed in data/historico.  The functions
are intentionally defensive because public exports often contain duplicated files,
old encodings, comma-separated TXT, DMS coordinates and occasional empty values.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import math
import zipfile

import numpy as np
import pandas as pd
from scipy import stats

DATASETS = {
    "pmax24_anual": {
        "file": "prep_maximas_anuales24_historico_4.zip",
        "label": "Precipitación máxima anual 24 h",
        "value_cols": ["Precipitación_max_anual_24horas"],
        "date_cols": ["ANIO"],
        "unit": "mm",
        "variable": "Pmax24 anual",
    },
    "precip_mensual": {
        "file": "prep_mensuales_historico_4.zip",
        "label": "Precipitación mensual",
        "value_cols": ["PREP_MENSUAL"],
        "date_cols": ["ANIO", "MES"],
        "unit": "mm",
        "variable": "Precipitación mensual",
    },
    "sed_rutinario": {
        "file": "sedimentos_muestreo_rutinario_historico_4.zip",
        "label": "Sedimentos muestreo rutinario",
        "value_cols": ["Concentración"],
        "date_cols": ["Año", "Mes", "Día"],
        "unit": "mg/L",
        "variable": "Concentración sedimentos rutinaria",
    },
    "sed_integrado": {
        "file": "sedimentos_muestreo_integrado_historico_4.zip",
        "label": "Sedimentos muestreo integrado",
        "value_cols": ["Concentración_Ponderada_Diaria"],
        "date_cols": ["Fecha"],
        "unit": "mg/L",
        "variable": "Concentración sedimentos integrada",
    },
    "tmax_diaria": {
        "file": "tmax_diarias_historico_4.zip",
        "label": "Temperatura máxima diaria",
        "value_cols": ["TMAX"],
        "date_cols": ["FECHA"],
        "unit": "°C",
        "variable": "Temperatura máxima diaria",
    },
    "tmin_diaria": {
        "file": "tmin_diarias_historico_4.zip",
        "label": "Temperatura mínima diaria",
        "value_cols": ["TMIN"],
        "date_cols": ["FECHA"],
        "unit": "°C",
        "variable": "Temperatura mínima diaria",
    },
    "tmed_diaria": {
        "file": "temp_med_diarias_historico_4.zip",
        "label": "Temperatura media diaria",
        "value_cols": ["TEMP_MEDIA_DIARIA"],
        "date_cols": ["FECHA"],
        "unit": "°C",
        "variable": "Temperatura media diaria",
    },
    "tmed_mensual": {
        "file": "temp_medias_mensuales_historico_4.zip",
        "label": "Temperatura media mensual",
        "value_cols": ["TEMP_MEDIA_MENSUAL"],
        "date_cols": ["ANIO", "MES"],
        "unit": "°C",
        "variable": "Temperatura media mensual",
    },
}


def _base_dir_str(data_dir: str | Path) -> str:
    return str(Path(data_dir).resolve())


def dms_text_to_dd(value: object, is_lon: bool = False) -> float:
    """Convert DGA compact DMS strings like 0292223 or 0710703 to decimal degrees.

    Coordinates in the Región de Coquimbo exports are south/west, so the returned
    latitude and longitude are negative. If the value is already decimal, it is
    returned with the expected sign.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    s = str(value).strip().replace(" ", "")
    if not s:
        return np.nan
    try:
        # Decimal coordinate already.
        if "." in s and len(s) < 9:
            x = float(s)
            if is_lon and x > 0:
                x = -abs(x)
            if not is_lon and x > 0:
                x = -abs(x)
            return x
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) < 6:
            return np.nan
        # DGA: latitude ddmmss, longitude dddmmss.
        if is_lon:
            deg = int(digits[:-4])
            minutes = int(digits[-4:-2])
            seconds = int(digits[-2:])
        else:
            deg = int(digits[:-4])
            minutes = int(digits[-4:-2])
            seconds = int(digits[-2:])
        dd = deg + minutes / 60.0 + seconds / 3600.0
        return -abs(dd)
    except Exception:
        return np.nan


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if any(pd.isna(v) for v in [lon1, lat1, lon2, lat2]):
        return np.nan
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * r * math.asin(math.sqrt(a))


@lru_cache(maxsize=32)
def read_dataset(data_dir: str, dataset_key: str) -> pd.DataFrame:
    meta = DATASETS[dataset_key]
    path = Path(data_dir) / meta["file"]
    if not path.exists():
        return pd.DataFrame()
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            if not names:
                return pd.DataFrame()
            with zf.open(names[0]) as f:
                df = pd.read_csv(f, sep=",", encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        with zipfile.ZipFile(path, "r") as zf:
            with zf.open(zf.namelist()[0]) as f:
                df = pd.read_csv(f, sep=",", encoding="latin-1", low_memory=False)
    except Exception:
        return pd.DataFrame()

    # Normalize coordinate/value columns.
    if "LAT" in df.columns:
        df["lat_dd"] = df["LAT"].apply(lambda v: dms_text_to_dd(v, is_lon=False))
    if "LONG" in df.columns:
        df["lon_dd"] = df["LONG"].apply(lambda v: dms_text_to_dd(v, is_lon=True))
    for col in meta.get("value_cols", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "ALTITUD" in df.columns:
        df["ALTITUD"] = pd.to_numeric(df["ALTITUD"], errors="coerce")
    return df


def available_historical_files(data_dir: str | Path) -> pd.DataFrame:
    rows = []
    base = Path(data_dir)
    for key, meta in DATASETS.items():
        p = base / meta["file"]
        rows.append({
            "dataset": key,
            "variable": meta["label"],
            "archivo": meta["file"],
            "disponible": p.exists(),
            "tamano_MB": round(p.stat().st_size / 1024**2, 3) if p.exists() else np.nan,
        })
    return pd.DataFrame(rows)


def _station_summary(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    if df.empty or "CODIGO ESTACION" not in df.columns:
        return pd.DataFrame()
    meta = DATASETS[dataset_key]
    value_col = next((c for c in meta["value_cols"] if c in df.columns), None)
    groups = []
    for code, g in df.groupby("CODIGO ESTACION", dropna=False):
        rec = {
            "codigo": str(code),
            "estacion": str(g["NOMBRE ESTACION"].iloc[0]) if "NOMBRE ESTACION" in g.columns else "",
            "comuna": str(g["COMUNA"].iloc[0]) if "COMUNA" in g.columns else "",
            "lat": float(g["lat_dd"].dropna().iloc[0]) if "lat_dd" in g.columns and not g["lat_dd"].dropna().empty else np.nan,
            "lon": float(g["lon_dd"].dropna().iloc[0]) if "lon_dd" in g.columns and not g["lon_dd"].dropna().empty else np.nan,
            "altitud_m": float(g["ALTITUD"].dropna().iloc[0]) if "ALTITUD" in g.columns and not g["ALTITUD"].dropna().empty else np.nan,
            "dataset": dataset_key,
            "variable": meta["variable"],
            "n_registros": int(len(g)),
        }
        # Date coverage.
        years = pd.Series(dtype=float)
        if "ANIO" in g.columns:
            years = pd.to_numeric(g["ANIO"], errors="coerce")
        elif "Año" in g.columns:
            years = pd.to_numeric(g["Año"], errors="coerce")
        elif "FECHA" in g.columns:
            years = pd.to_datetime(g["FECHA"], dayfirst=True, errors="coerce").dt.year
        elif "Fecha" in g.columns:
            years = pd.to_datetime(g["Fecha"], dayfirst=True, errors="coerce").dt.year
        rec["anio_inicio"] = int(years.dropna().min()) if not years.dropna().empty else np.nan
        rec["anio_fin"] = int(years.dropna().max()) if not years.dropna().empty else np.nan
        rec["anios_con_datos"] = int(years.dropna().nunique()) if not years.dropna().empty else np.nan
        if value_col:
            vals = pd.to_numeric(g[value_col], errors="coerce").dropna()
            rec["valor_medio"] = float(vals.mean()) if not vals.empty else np.nan
            rec["valor_max"] = float(vals.max()) if not vals.empty else np.nan
            rec["valor_p95"] = float(vals.quantile(0.95)) if len(vals) >= 5 else np.nan
            rec["unidad"] = meta["unit"]
        groups.append(rec)
    return pd.DataFrame(groups)


@lru_cache(maxsize=16)
def build_station_catalog(data_dir: str) -> pd.DataFrame:
    parts = []
    for key in DATASETS:
        df = read_dataset(data_dir, key)
        s = _station_summary(df, key)
        if not s.empty:
            parts.append(s)
    if not parts:
        return pd.DataFrame()
    cat = pd.concat(parts, ignore_index=True)
    # Collapse to one row per station, retaining availability by variable.
    rows = []
    for code, g in cat.groupby("codigo"):
        datasets = sorted(g["dataset"].unique().tolist())
        variables = sorted(g["variable"].unique().tolist())
        rows.append({
            "codigo": code,
            "estacion": g["estacion"].iloc[0],
            "comuna": g["comuna"].iloc[0],
            "lat": g["lat"].dropna().iloc[0] if not g["lat"].dropna().empty else np.nan,
            "lon": g["lon"].dropna().iloc[0] if not g["lon"].dropna().empty else np.nan,
            "altitud_m": g["altitud_m"].dropna().iloc[0] if not g["altitud_m"].dropna().empty else np.nan,
            "datasets": ", ".join(datasets),
            "variables": ", ".join(variables),
            "n_variables": len(variables),
            "anio_inicio_min": int(g["anio_inicio"].dropna().min()) if not g["anio_inicio"].dropna().empty else np.nan,
            "anio_fin_max": int(g["anio_fin"].dropna().max()) if not g["anio_fin"].dropna().empty else np.nan,
            "registros_total": int(g["n_registros"].sum()),
        })
    return pd.DataFrame(rows)


def nearest_stations(catalog: pd.DataFrame, lon: float, lat: float, n: int = 10, max_km: float = 100.0, dataset_contains: Optional[str] = None) -> pd.DataFrame:
    if catalog is None or catalog.empty:
        return pd.DataFrame()
    df = catalog.copy()
    if dataset_contains:
        df = df[df["datasets"].str.contains(dataset_contains, na=False)]
    if df.empty:
        return df
    df["distancia_km"] = [haversine_km(lon, lat, lo, la) for lo, la in zip(df["lon"], df["lat"])]
    df = df[df["distancia_km"].notna()]
    if max_km is not None and max_km > 0:
        df = df[df["distancia_km"] <= float(max_km)]
    return df.sort_values("distancia_km").head(int(n)).reset_index(drop=True)


def dataset_station_summary(data_dir: str | Path, dataset_key: str, lon: float, lat: float, n: int = 8, max_km: float = 120.0) -> pd.DataFrame:
    df = read_dataset(_base_dir_str(data_dir), dataset_key)
    s = _station_summary(df, dataset_key)
    if s.empty:
        return s
    s["distancia_km"] = [haversine_km(lon, lat, lo, la) for lo, la in zip(s["lon"], s["lat"])]
    if max_km is not None and max_km > 0:
        s = s[s["distancia_km"] <= float(max_km)]
    return s.sort_values("distancia_km").head(int(n)).reset_index(drop=True)




def frequency_multi_model(values: Iterable[float], return_periods: Iterable[float]) -> pd.DataFrame:
    """Fit several common frequency models to annual maxima.

    Models: Gumbel (EV1), Normal, LogNormal-2 and Log-Pearson III on log10 values.
    This implements a screening tool; the app keeps all estimates visible so the
    specialist can compare dispersion and choose/justify a model.
    """
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    rows = []
    if len(vals) < 5:
        for T in return_periods:
            rows.append({"T_anios": float(T), "P24_Gumbel_mm": np.nan, "P24_Normal_mm": np.nan, "P24_LogNormal_mm": np.nan, "P24_LogPearsonIII_mm": np.nan, "P24_MediaModelos_mm": np.nan, "n_datos": int(len(vals)), "asimetria_muestral": np.nan, "estado_frecuencia": "Insuficiente: menos de 5 datos"})
        return pd.DataFrame(rows)
    x = vals.astype(float)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    skew = float(stats.skew(x, bias=False)) if len(x) >= 3 else np.nan
    logx = np.log10(x[x > 0])
    logmean = float(logx.mean()) if len(logx) else np.nan
    logstd = float(logx.std(ddof=1)) if len(logx) >= 2 else np.nan
    logskew = float(stats.skew(logx, bias=False)) if len(logx) >= 3 else np.nan

    for T in return_periods:
        T = float(T)
        F = 1.0 - 1.0 / T if T > 1 else np.nan
        if not np.isfinite(F) or F <= 0 or F >= 1:
            gumbel = normal = lognorm = lp3 = np.nan
        else:
            # Gumbel via moments, same as previous app behavior.
            gamma = 0.5772156649
            sy = math.pi / math.sqrt(6)
            yT = -math.log(-math.log(F))
            gumbel = max(0.0, mean + (std / sy) * (yT - gamma)) if std > 0 else mean
            normal = max(0.0, float(stats.norm.ppf(F, loc=mean, scale=std))) if std > 0 else mean
            if len(logx) >= 5 and logstd > 0:
                lognorm = 10 ** float(stats.norm.ppf(F, loc=logmean, scale=logstd))
                try:
                    # Pearson III in log domain. Shape parameter is skewness.
                    if np.isfinite(logskew) and abs(logskew) > 1e-6:
                        lp_log = stats.pearson3.ppf(F, skew=logskew, loc=logmean, scale=logstd)
                    else:
                        lp_log = stats.norm.ppf(F, loc=logmean, scale=logstd)
                    lp3 = 10 ** float(lp_log)
                except Exception:
                    lp3 = np.nan
            else:
                lognorm = lp3 = np.nan
        estimates = [v for v in [gumbel, normal, lognorm, lp3] if np.isfinite(v)]
        rows.append({
            "T_anios": T,
            "P24_Gumbel_mm": float(gumbel) if np.isfinite(gumbel) else np.nan,
            "P24_Normal_mm": float(normal) if np.isfinite(normal) else np.nan,
            "P24_LogNormal_mm": float(lognorm) if np.isfinite(lognorm) else np.nan,
            "P24_LogPearsonIII_mm": float(lp3) if np.isfinite(lp3) else np.nan,
            "P24_MediaModelos_mm": float(np.mean(estimates)) if estimates else np.nan,
            "P24_MaxModelos_mm": float(np.max(estimates)) if estimates else np.nan,
            "n_datos": int(len(x)),
            "asimetria_muestral": skew,
            "asimetria_log": logskew,
            "estado_frecuencia": "OK" if len(x) >= 15 else "Referencial: serie corta",
        })
    return pd.DataFrame(rows)

def gumbel_frequency(values: Iterable[float], return_periods: Iterable[float]) -> Dict[float, float]:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    if len(vals) < 5:
        return {float(T): np.nan for T in return_periods}
    mean = float(vals.mean())
    std = float(vals.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return {float(T): mean for T in return_periods}
    gamma = 0.5772156649
    sy = math.pi / math.sqrt(6)
    out = {}
    for T in return_periods:
        T = float(T)
        if T <= 1:
            out[T] = np.nan
            continue
        yT = -math.log(-math.log(1 - 1 / T))
        xT = mean + (std / sy) * (yT - gamma)
        out[T] = max(0.0, float(xT))
    return out


def pmax24_frequency_estimates(data_dir: str | Path, lon: float, lat: float, return_periods: Iterable[float], n: int = 5, max_km: float = 150.0, min_years: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = _base_dir_str(data_dir)
    df = read_dataset(data_dir, "pmax24_anual")
    s = _station_summary(df, "pmax24_anual")
    if df.empty or s.empty:
        return pd.DataFrame(), pd.DataFrame()
    s["distancia_km"] = [haversine_km(lon, lat, lo, la) for lo, la in zip(s["lon"], s["lat"])]
    s = s[s["distancia_km"].notna()]
    if max_km:
        s = s[s["distancia_km"] <= float(max_km)]
    s = s.sort_values(["distancia_km", "anios_con_datos"], ascending=[True, False]).head(max(int(n), 1)).reset_index(drop=True)
    rows = []
    for _, st in s.iterrows():
        g = df[df["CODIGO ESTACION"].astype(str) == str(st["codigo"])]
        vals = pd.to_numeric(g["Precipitación_max_anual_24horas"], errors="coerce").dropna()
        vals = vals[vals >= 0]
        freq_df = frequency_multi_model(vals, return_periods)
        for _, fr in freq_df.iterrows():
            rec = {
                "codigo": st["codigo"],
                "estacion": st["estacion"],
                "distancia_km": st["distancia_km"],
                "altitud_m": st["altitud_m"],
                "anios_con_datos": int(st["anios_con_datos"]) if pd.notna(st["anios_con_datos"]) else np.nan,
                "T_anios": float(fr["T_anios"]),
                "P24_max_obs_mm": float(vals.max()) if not vals.empty else np.nan,
                "P24_media_obs_mm": float(vals.mean()) if not vals.empty else np.nan,
                "estado": "OK" if len(vals) >= min_years else "Referencial: pocos años",
            }
            for c in freq_df.columns:
                if c != "T_anios":
                    rec[c] = fr[c]
            rows.append(rec)
    return s, pd.DataFrame(rows)

def sediment_nearest_summary(data_dir: str | Path, lon: float, lat: float, n: int = 5, max_km: float = 200.0) -> pd.DataFrame:
    parts = []
    for key in ["sed_rutinario", "sed_integrado"]:
        ss = dataset_station_summary(data_dir, key, lon, lat, n=n, max_km=max_km)
        if not ss.empty:
            parts.append(ss)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("distancia_km").reset_index(drop=True)


def climate_nearest_summary(data_dir: str | Path, lon: float, lat: float, n: int = 5, max_km: float = 120.0) -> pd.DataFrame:
    parts = []
    for key in ["precip_mensual", "tmax_diaria", "tmin_diaria", "tmed_diaria", "tmed_mensual"]:
        ss = dataset_station_summary(data_dir, key, lon, lat, n=n, max_km=max_km)
        if not ss.empty:
            parts.append(ss)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["distancia_km", "dataset"]).reset_index(drop=True)
