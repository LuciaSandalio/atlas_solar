#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 1_b — Muestreo de radiancia GOES en grilla y cálculo de reflectancia TOA.

Entrada:
  - data/goes/YYYYMMDD/*.nc (ABI-L1b-RadF, bandas visibles C01/C02/C03)
  - data/grids/cordoba_grid_<res_km>km.geojson (Point o polígonos; se usa el centroide)
  - (opcional) data/goes/YYYYMMDD/_index.csv si usás --from-index

Salida:
  - data/processed/reflectancia_YYYYMMDD.parquet

Ejemplos:
  python scripts/fase_1_b.py --fecha 2020-01-15 --res_km 5 --canal C02 --sat goes16
  python scripts/fase_1_b.py --fecha 2024-09-17 --res_km 5 --canal C01 --sat goes16 --from-index
  python scripts/fase_1_b.py --fecha 2024-09-17 --res_km 5 --canal C02 --sat goes16 --chunks 800,800  # si tenés dask

Requisitos: numpy, pandas, xarray, geopandas, pvlib, pyarrow/fastparquet, netCDF4 (engine).
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import json
import hashlib

import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import pvlib # type: ignore

# Limitar hilos BLAS para estabilidad
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
xr.set_options(keep_attrs=True)

# Dask opcional
try:
    import dask  # type: ignore # noqa: F401
    HAS_DASK = True
except Exception:
    HAS_DASK = False

# Irradiancia extraterrestre espectral por banda visible (W m-2 um-1)
ESUN_BY_BAND: Dict[str, float] = {"C01": 466.8, "C02": 663.274, "C03": 441.868}

# ------------------ Utilidades de IO / paths ------------------
def _pick_engine():
    try:
        import netCDF4  # type: ignore # noqa: F401
        return "netcdf4"
    except Exception:
        raise RuntimeError("Falta netCDF4. Instalá: pip install netCDF4")

def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        return gdf.set_crs(epsg=4326)
    if gdf.crs.to_epsg() != 4326:
        return gdf.to_crs(epsg=4326)
    return gdf

def _grid_lonlat(grid: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
    geom = grid.geometry
    if not np.all(geom.geom_type.values == "Point"):
        geom = geom.centroid
    lon = geom.x.to_numpy(dtype="float64")
    lat = geom.y.to_numpy(dtype="float64")
    return lon, lat

def _list_nc_files(day: pd.Timestamp, band: str, goes_id: str) -> List[Path]:
    day_dir = Path("data/goes") / day.strftime("%Y%m%d")
    if not day_dir.exists():
        return []
    files = sorted(day_dir.glob("*.nc"))
    filt = []
    tag_band = f"-M6C{int(band):02d}_"
    tag_sat = f"_G{int(goes_id):02d}_"
    for p in files:
        name = p.name
        if tag_band in name and tag_sat in name and "ABI-L1b-RadF" in name:
            filt.append(p)
    return filt

# ------------------ Parseo de hora de escena ------------------
# Nombre típico: OR_ABI-L1b-RadF-M6C02_G16_s20200151500198_e20200151509506_c20200151509551.nc
ABI_NAME_RE = re.compile(
    r"OR_(?P<product>ABI-L1b-[A-Za-z0-9]+)-M6C(?P<chan>\d{2})_G(?P<goes>\d{2})_"
    r"s(?P<sy>\d{4})(?P<sdoy>\d{3})(?P<sh>\d{2})(?P<sm>\d{2})(?P<ss>\d{2})\d*_"
    r"e(?P<ey>\d{4})(?P<edoy>\d{3})(?P<eh>\d{2})(?P<em>\d{2})(?P<es>\d{2})\d*_"
    r"c(?P<cy>\d{4})(?P<cdoy>\d{3})(?P<ch>\d{2})(?P<cm>\d{2})(?P<cs>\d{2})\d*\.nc$"
)

def _ydoyhms_to_ts(y: str, doy: str, h: str, mi: str, s: str) -> pd.Timestamp:
    y, doy, h, mi, s = int(y), int(doy), int(h), int(mi), int(s) # type: ignore
    base = pd.Timestamp(y, 1, 1, tz="UTC") + pd.Timedelta(days=doy - 1) # type: ignore
    return base + pd.Timedelta(hours=h, minutes=mi, seconds=s)

def parse_scene_time(ds: xr.Dataset, nc_path: Path, fallback_day: str) -> pd.Timestamp:
    for cn in ("t", "time", "start_time"):
        if cn in ds.coords:
            tt = pd.to_datetime(ds.coords[cn].values)
            return tt.tz_localize("UTC") if getattr(tt, "tz", None) is None else tt.tz_convert("UTC")
    m = ABI_NAME_RE.search(nc_path.name)
    if m:
        gd = m.groupdict()
        return _ydoyhms_to_ts(gd["sy"], gd["sdoy"], gd["sh"], gd["sm"], gd["ss"])
    return pd.Timestamp(f"{fallback_day} 15:00:00", tz="UTC")

# ------------- Geometría GOES: lon/lat → (x,y) scan angles -------------
def lonlat_to_scan_angles(lon_deg: np.ndarray, lat_deg: np.ndarray, attrs: dict) -> Tuple[np.ndarray, np.ndarray]:
    lon0 = np.deg2rad(float(attrs["longitude_of_projection_origin"]))  # rad
    h = float(attrs["perspective_point_height"])
    a = float(attrs.get("semi_major_axis", 6378137.0))
    b = float(attrs.get("semi_minor_axis", 6356752.31414))
    H = h + a
    e2 = 1.0 - (b * b) / (a * a)

    lam = np.deg2rad(lon_deg)
    phi = np.deg2rad(lat_deg)

    r1 = a / np.sqrt(1.0 - e2 * (np.sin(phi) ** 2))
    sx = H - r1 * np.cos(phi) * np.cos(lam - lon0)
    sy = r1 * np.cos(phi) * np.sin(lam - lon0)
    sz = r1 * np.sin(phi)

    vis = (H * (H - sx)) > (sy * sy + sz * sz)

    x = np.arctan2(sy, sx).astype("float64")
    y = np.arctan2(sz, np.sqrt(sx * sx + sy * sy)).astype("float64")
    x[~vis] = np.nan
    y[~vis] = np.nan
    return x, y

def nearest_idx_anyorder(coord: np.ndarray, targets: np.ndarray) -> np.ndarray:
    asc = coord[0] <= coord[-1]
    arr = coord if asc else coord[::-1]
    pos = np.searchsorted(arr, targets)
    pos = np.clip(pos, 1, len(arr) - 1)
    left = arr[pos - 1]; right = arr[pos]
    choose = np.where(np.abs(right - targets) < np.abs(targets - left), pos, pos - 1)
    if not asc:
        choose = (len(arr) - 1) - choose
    return choose

# ------------------ Reflectancia ------------------
def compute_reflectance(L: np.ndarray, esun: float, cosz: np.ndarray, d2: np.ndarray, clip_max: float = 1.2) -> np.ndarray:
    out = np.full(L.shape, np.nan, dtype=np.float32)
    m = (cosz > 0) & np.isfinite(L)
    out[m] = (np.pi * L[m] * d2[m] / (esun * cosz[m])).astype("float32")
    if clip_max is not None:
        np.clip(out, 0, clip_max, out=out)
    return out

# ------------------ Caché (ix, iy) ------------------
def _hash_dict(d: Dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]

def build_idx_cache(grid_lon: np.ndarray, grid_lat: np.ndarray, ds: xr.Dataset, canal: str,
                    grid_tag: str, cache_dir: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    attrs = ds["goes_imager_projection"].attrs
    xs = ds["x"].values.astype("float64")
    ys = ds["y"].values.astype("float64")

    xq, yq = lonlat_to_scan_angles(grid_lon, grid_lat, attrs)
    ok = np.isfinite(xq) & np.isfinite(yq)
    inx = (xq >= np.nanmin(xs)) & (xq <= np.nanmax(xs))
    iny = (yq >= np.nanmin(ys)) & (yq <= np.nanmax(ys))
    ok_range = ok & inx & iny

    sig = {
        "canal": canal.upper(),
        "grid": grid_tag,
        "nx": int(xs.size), "ny": int(ys.size),
        "xmin": float(xs.min()), "xmax": float(xs.max()),
        "ymin": float(ys.min()), "ymax": float(ys.max()),
        "lon0": float(attrs["longitude_of_projection_origin"]),
        "h": float(attrs["perspective_point_height"]),
        "a": float(attrs.get("semi_major_axis", 6378137.0)),
        "b": float(attrs.get("semi_minor_axis", 6356752.31414)),
    }
    h = _hash_dict(sig)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"idx_{h}.npz"

    if cache_file.exists():
        z = np.load(cache_file)
        ix = z["ix"]; iy = z["iy"]
        if ix.shape == grid_lon.shape and iy.shape == grid_lon.shape:
            return ix, iy, cache_file

    ix = np.full(grid_lon.shape, -1, dtype=np.int32)
    iy = np.full(grid_lon.shape, -1, dtype=np.int32)
    if ok_range.any():
        ix_ok = nearest_idx_anyorder(xs, xq[ok_range])
        iy_ok = nearest_idx_anyorder(ys, yq[ok_range])
        ix[ok_range] = ix_ok.astype(np.int32)
        iy[ok_range] = iy_ok.astype(np.int32)

    np.savez_compressed(cache_file, ix=ix, iy=iy, **sig)
    return ix, iy, cache_file

# ------------------ Proceso por escena ------------------
def process_scene(nc_path: Path, grid: gpd.GeoDataFrame, lon: np.ndarray, lat: np.ndarray,
                  ix: np.ndarray, iy: np.ndarray, esun: float, chunks: Optional[str]) -> Optional[pd.DataFrame]:
    engine = _pick_engine()
    if HAS_DASK and chunks:
        cy, cx = (int(v) for v in chunks.split(","))
        ds = xr.open_dataset(nc_path, engine=engine, chunks={"y": cy, "x": cx})
    else:
        ds = xr.open_dataset(nc_path, engine=engine)

    var_name = "Rad"
    if var_name not in ds.data_vars:
        cand = [v for v in ds.data_vars if v.lower().startswith("rad")]
        if not cand:
            return None
        var_name = cand[0]
    rad = ds[var_name]

    # Hora de escena
    scene_time = parse_scene_time(ds, nc_path, "2000-01-01")

    # Muestreo radiancia (descarta índices -1)
    mask = (ix >= 0) & (iy >= 0)
    L = np.full(ix.shape, np.nan, dtype="float32")
    if mask.any():
        sel = rad.isel(x=("points", ix[mask]), y=("points", iy[mask]))
        L[mask] = (sel.compute().values if HAS_DASK else sel.values).astype("float32")

    # cos(θz) por bloques
    n = len(grid)
    cosz = np.empty(n, dtype="float32")
    BLOCK = 4000
    for i in range(0, n, BLOCK):
        j = min(i + BLOCK, n)
        t_idx = pd.DatetimeIndex([scene_time] * (j - i))
        sp = pvlib.solarposition.get_solarposition(t_idx, lat[i:j], lon[i:j])
        z = sp["zenith"].to_numpy(dtype="float64")
        cosz[i:j] = np.cos(np.radians(z)).astype("float32")

    # d^2 (distancia Tierra-Sol en UA)
    N = scene_time.dayofyear
    d = 1 - 0.01672 * np.cos(np.radians(0.9856 * (N - 4)))
    d2 = np.full(L.shape, d ** 2, dtype="float32")

    rho = compute_reflectance(L, esun, cosz, d2, clip_max=1.2)

    df = pd.DataFrame({
        "cell_id": grid["cell_id"].values.astype("int32"),
        "timestamp": np.repeat(scene_time, n),
        "radiance": L.astype("float32"),
        "cosz": cosz.astype("float32"),
        "reflectance": rho.astype("float32"),
        "nc_file": nc_path.name,
    })
    ds.close()
    return df

# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser(description="Fase 1_b — Muestreo GOES y reflectancia TOA")
    ap.add_argument("--fecha", required=True, help="YYYY-MM-DD")
    ap.add_argument("--res_km", type=float, default=5)
    ap.add_argument("--canal", default="C02", choices=["C01", "C02", "C03"])
    ap.add_argument("--sat", default="goes16", choices=["goes16", "goes18", "goes19"])
    ap.add_argument("--chunks", default="", help="y,x (si tenés dask), ej 800,800")
    ap.add_argument("--from-index", action="store_true",
                    help="Usa data/goes/YYYYMMDD/_index.csv para elegir escenas (recomendado si descargaste cada 30′)")
    args = ap.parse_args()

    fecha = pd.Timestamp(args.fecha).tz_localize("UTC")
    day_dir = Path("data/goes") / fecha.strftime("%Y%m%d")
    if not day_dir.exists():
        raise FileNotFoundError(f"No existe {day_dir}. Corré fase_1.py primero.")

    # Grilla
    grid_file = Path("data/grids") / f"cordoba_grid_{args.res_km:.0f}km.geojson"
    if not grid_file.exists():
        raise FileNotFoundError(grid_file)
    grid = gpd.read_file(grid_file)[["cell_id", "geometry"]].copy()
    grid = _ensure_wgs84(grid) # type: ignore
    lon, lat = _grid_lonlat(grid)

        # Selección de archivos (robusta + fallback)
    goes_id = {"goes16": "16", "goes18": "18", "goes19": "19"}[args.sat]
    band_num = int(args.canal.replace("C", "").replace("c", ""))
    files: List[Path] = []

    index_csv = day_dir / "_index.csv"
    if args.from_index and index_csv.exists():
        try:
            idx = pd.read_csv(index_csv)
            # normalización robusta de band/goes en el índice
            def _norm_band(x):
                s = str(x).strip().upper()
                if s.startswith("C"):
                    s = s[1:]
                return f"C{int(s):02d}"
            def _norm_goes(x):
                s = str(x).strip()
                return f"{int(s):02d}"

            if "band" in idx.columns:
                idx["band_norm"] = idx["band"].map(_norm_band)
            else:
                # si no existe 'band', intentar inferir desde filename
                idx["band_norm"] = idx["filename"].astype(str).str.extract(r"-M6C(\d{2})_")[0].map(lambda v: f"C{int(v):02d}")

            if "goes" in idx.columns:
                idx["goes_norm"] = idx["goes"].map(_norm_goes)
            else:
                idx["goes_norm"] = idx["filename"].astype(str).str.extract(r"_G(\d{2})_")[0].map(lambda v: f"{int(v):02d}")

            want_band = f"C{band_num:02d}"
            want_goes = goes_id

            idx_f = idx[(idx["band_norm"] == want_band) & (idx["goes_norm"] == want_goes)]
            if idx_f.empty:
                print(f"⚠️  _index.csv no tiene filas para band={want_band} goes={want_goes}. Hago fallback a listar .nc en disco…")
                files = _list_nc_files(fecha, str(band_num), goes_id)
            else:
                files = [day_dir / fn for fn in idx_f["filename"].astype(str).tolist()] # type: ignore
        except Exception as e:
            print(f"⚠️  Error leyendo {index_csv}: {e}. Fallback a listar .nc en disco…")
            files = _list_nc_files(fecha, str(band_num), goes_id)
    else:
        files = _list_nc_files(fecha, str(band_num), goes_id)

    if not files:
        raise FileNotFoundError(
            f"Sin escenas para {args.fecha} banda {args.canal} en {day_dir} (from_index={args.from_index})"
        )


    # Esun por banda
    esun = float(ESUN_BY_BAND[args.canal.upper()])

    # Caché de índices con el primer archivo
    engine = _pick_engine()
    ds0 = xr.open_dataset(files[0], engine=engine)
    grid_tag = f"cordoba_{args.res_km:.0f}km"
    cache_dir = Path("data/processed/cache_idx")
    ix, iy, cache_file = build_idx_cache(lon, lat, ds0, args.canal.upper(), grid_tag, cache_dir)
    ds0.close()
    print(f"🧭 idx cache: {cache_file.name} | puntos válidos: {int(((ix>=0)&(iy>=0)).sum())}/{len(ix)}")

    # Procesar escenas
    dfs = []
    chunks = args.chunks if (HAS_DASK and args.chunks.strip()) else None
    valid_r_samples_total = 0
    for nc in files:
        df = process_scene(nc, grid, lon, lat, ix, iy, esun, chunks)
        if df is None or df.empty:
            print(f"  - {nc.name}: vacío o Radiance no encontrada, salto.")
            continue
        n_valid_rad = int(np.isfinite(df["radiance"]).sum())
        valid_r_samples_total += n_valid_rad
        print(f"  - {nc.name}: radiancia válidas {n_valid_rad}/{len(df)}")
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No se generaron filas (todas escenas vacías).")

    out = pd.concat(dfs, ignore_index=True)
    out.sort_values(["timestamp", "cell_id"], inplace=True)
    out_file = Path("data/processed") / f"reflectancia_{fecha.strftime('%Y%m%d')}.parquet"
    out.to_parquet(out_file, compression="zstd", index=False)

    cosz_pos = int((out["cosz"] > 0).sum())
    print(f"✅ Reflectancia guardada en {out_file} ({len(out)} filas; escenas: {len(dfs)})")
    print(f"ℹ️  Radiancia válidas (día): {valid_r_samples_total}/{len(out)} | celdas con cosz>0: {cosz_pos}/{len(out)}")

if __name__ == "__main__":
    main()
