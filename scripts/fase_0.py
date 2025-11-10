#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fase 1-B (GOES -> Grid -> Reflectancia, robusto a signos/orden de x/y):
- Lee una escena GOES ABI (L1b Radiance)
- Mapea radiancia sobre la rejilla provincial (lon/lat -> scan angles x,y)
- Calcula cos(θz) con pvlib en la hora exacta
- Calcula reflectancia TOA y guarda Parquet
"""

from typing import Optional, Tuple
import argparse, os, re
from pathlib import Path
import numpy as np # type: ignore
import pandas as pd # type: ignore
import geopandas as gpd # type: ignore
import xarray as xr # type: ignore
import pvlib # type: ignore

os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS","1")
xr.set_options(keep_attrs=True)

try:
    import dask  # type: ignore # noqa
    HAS_DASK = True
except Exception:
    HAS_DASK = False

ESUN_BY_BAND = {"C01": 466.8, "C02": 663.274, "C03": 441.868}

# ---------------- utils ----------------
def find_goes_file(fecha_str: str) -> Optional[Path]:
    d = Path("data/goes") / pd.Timestamp(fecha_str).strftime("%Y%m%d")
    if not d.exists(): return None
    nc = sorted(d.glob("*.nc"))
    return nc[0] if nc else None

def _pick_engine() -> str:
    try:
        import netCDF4  # type: ignore # noqa
        return "netcdf4"
    except Exception:
        raise RuntimeError("Falta netCDF4. Instalá: pip install netCDF4")

def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)
        elif gdf.crs.to_epsg()!=4326:
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass
    return gdf

def _grid_lonlat(grid: gpd.GeoDataFrame) -> Tuple[np.ndarray, np.ndarray]:
    geom = grid.geometry
    try:
        if not np.all(geom.geom_type.values == "Point"):
            geom = geom.centroid
    except Exception:
        geom = geom.centroid
    lon = geom.x.astype("float64").values
    lat = geom.y.astype("float64").values
    return lon, lat

def _scene_time_from_ds_or_name(ds, goes_file: Path, fecha_fallback: str) -> pd.Timestamp:
    scene_time = None
    for cn in ["t","time","start_time"]:
        if cn in ds.coords:
            try:
                tt = pd.to_datetime(ds.coords[cn].values)
                scene_time = tt.tz_localize("UTC") if getattr(tt,"tz",None) is None else tt.tz_convert("UTC")
                break
            except Exception:
                pass
    if scene_time is None:
        m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})", Path(str(goes_file)).name)
        if m:
            y,jjj,hh,mm = m.groups()
            scene_time = pd.Timestamp(int(y),1,1,tz="UTC") + pd.Timedelta(days=int(jjj)-1, hours=int(hh), minutes=int(mm))
        else:
            scene_time = pd.Timestamp(f"{fecha_fallback} 15:00:00", tz="UTC")
    return scene_time

def compute_reflectance(L, esun, cosz, d2, clip_max=1.2):
    out = np.full(L.shape, np.nan, dtype=np.float32)
    m = (cosz>0) & np.isfinite(L)
    out[m] = (np.pi * L[m] * d2[m] / (esun * cosz[m])).astype(np.float32)
    if clip_max is not None: np.clip(out,0,clip_max,out=out)
    return out

def lonlat_to_scan_angles(lon_deg: np.ndarray, lat_deg: np.ndarray, attrs: dict) -> Tuple[np.ndarray, np.ndarray]:
    # Parámetros (preferir los del archivo)
    lon0 = np.deg2rad(attrs.get("longitude_of_projection_origin"))
    h    = float(attrs.get("perspective_point_height")) # type: ignore
    a    = float(attrs.get("semi_major_axis", 6378137.0))
    b    = float(attrs.get("semi_minor_axis", 6356752.31414))
    H    = h + a
    e2   = 1.0 - (b*b)/(a*a)

    lam = np.deg2rad(lon_deg)
    phi = np.deg2rad(lat_deg)

    r1 = a / np.sqrt(1.0 - e2*(np.sin(phi)**2))
    sx = H - r1*np.cos(phi)*np.cos(lam - lon0)
    sy =      r1*np.cos(phi)*np.sin(lam - lon0)
    sz =      r1*np.sin(phi)

    vis = (H*(H - sx)) > (sy*sy + sz*sz)
    x = np.arctan2(sy, sx).astype("float64")
    y = np.arctan2(sz, np.sqrt(sx*sx + sy*sy)).astype("float64")
    x[~vis] = np.nan
    y[~vis] = np.nan
    return x, y

def _nearest_idx_anyorder(coord: np.ndarray, targets: np.ndarray) -> np.ndarray:
    asc = coord[0] <= coord[-1]
    arr = coord if asc else coord[::-1]
    pos = np.searchsorted(arr, targets)
    pos = np.clip(pos, 1, len(arr)-1)
    left, right = arr[pos-1], arr[pos]
    choose = np.where(np.abs(right-targets) < np.abs(targets-left), pos, pos-1)
    if not asc: choose = (len(arr)-1) - choose
    return choose

def _in_range(xs, ys, xq, yq):
    okf = np.isfinite(xq) & np.isfinite(yq)
    inx = (xq >= np.nanmin(xs)) & (xq <= np.nanmax(xs))
    iny = (yq >= np.nanmin(ys)) & (yq <= np.nanmax(ys))
    ok  = okf & inx & iny
    return ok, ok.sum(), (~okf).sum(), (okf & ~inx).sum(), (okf & ~iny).sum()

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fecha", required=True)
    ap.add_argument("--res_km", type=float, default=5)
    ap.add_argument("--canal", default="C02")
    ap.add_argument("--sat", default="goes16")
    ap.add_argument("--chunks", default="1200,1200")
    args = ap.parse_args()

    fecha = pd.Timestamp(args.fecha); fecha_tag = fecha.strftime("%Y%m%d")
    grid_file = Path("data/grids") / f"cordoba_grid_{args.res_km:.0f}km.geojson"
    if not grid_file.exists(): raise FileNotFoundError(grid_file)

    goes_file = find_goes_file(args.fecha)
    if not goes_file: raise FileNotFoundError("No encontré NetCDF GOES en data/goes/FECHA.")

    grid = gpd.read_file(grid_file)[["cell_id","geometry"]].copy()
    grid = _ensure_wgs84(grid); grid["cell_id"]=grid["cell_id"].astype("int32") # type: ignore
    lon, lat = _grid_lonlat(grid)

    engine = _pick_engine()
    if HAS_DASK:
        cy,cx = (int(v) for v in args.chunks.split(","))
        ds = xr.open_dataset(goes_file, engine=engine, chunks={"y":cy,"x":cx})
    else:
        ds = xr.open_dataset(goes_file, engine=engine)

    var_name="Rad"
    if var_name not in ds.data_vars:
        cand=[v for v in ds.data_vars if v.lower().startswith("rad")]
        if not cand: raise KeyError("No encontré variable de radiancia.")
        var_name=cand[0]
    rad = ds[var_name]

    attrs = ds["goes_imager_projection"].attrs
    xs = ds["x"].values.astype("float64")
    ys = ds["y"].values.astype("float64")

    # 1) ángulos base
    x0,y0 = lonlat_to_scan_angles(lon.astype("float64"), lat.astype("float64"), attrs)

    # 2) probar 4 combinaciones de signo y elegir la mejor
    candidates = [
        ("+x,+y",  x0,    y0   ),
        ("-x,+y", -x0,    y0   ),
        ("+x,-y",  x0,   -y0   ),
        ("-x,-y", -x0,   -y0   ),
    ]
    scores = []
    for name, xq, yq in candidates:
        ok, n_ok, n_nan, n_ox, n_oy = _in_range(xs, ys, xq, yq)
        scores.append((n_ok, name, xq, yq, (n_nan, n_ox, n_oy)))
    scores.sort(reverse=True, key=lambda z: z[0])
    n_ok, chosen, xq, yq, diag = scores[0]
    print(f"ℹ️  Mejor combinación de signos: {chosen} | dentro de rango: {n_ok}/{len(lon)} "
          f"(NaN:{diag[0]}, fuera_x:{diag[1]}, fuera_y:{diag[2]})")

    # 3) muestreo con esa combinación
    ok, *_ = _in_range(xs, ys, xq, yq)
    L = np.full(lon.shape, np.nan, dtype="float32")
    if ok.any():
        ix = _nearest_idx_anyorder(xs, xq[ok])
        iy = _nearest_idx_anyorder(ys, yq[ok])
        da_x = xr.DataArray(xs[ix], dims=("points",))
        da_y = xr.DataArray(ys[iy], dims=("points",))
        sel = rad.sel(x=da_x, y=da_y, method="nearest")
        L[ok] = (sel.compute().values if HAS_DASK else sel.values).astype("float32")

    print(f"ℹ️  Muestreo radiancia: {int(np.isfinite(L).sum())}/{len(L)} válidas (no-NaN)")

    # Hora exacta de la escena
    scene_time = _scene_time_from_ds_or_name(ds, goes_file, args.fecha)

    # cos(θz) por bloques
    n = len(grid); cosz = np.empty(n, dtype="float32"); BLOCK=2000
    for i in range(0, n, BLOCK):
        j=min(i+BLOCK, n)
        t_idx = pd.DatetimeIndex([scene_time]*(j-i))
        sp = pvlib.solarposition.get_solarposition(t_idx, lat[i:j], lon[i:j])
        z = sp["zenith"].to_numpy(dtype="float64")
        cosz[i:j] = np.cos(np.radians(z)).astype("float32")

    # d^2 (AU^2)
    N = scene_time.dayofyear
    d = 1 - 0.01672*np.cos(np.radians(0.9856*(N-4)))
    d2 = np.full(L.shape, d**2, dtype="float32")

    esun = float(ESUN_BY_BAND.get(args.canal.upper(), ESUN_BY_BAND["C02"]))
    rho = compute_reflectance(L, esun, cosz, d2, clip_max=1.2)

    out = pd.DataFrame({
        "cell_id": grid["cell_id"].values.astype("int32"),
        "timestamp": np.repeat(scene_time, n),
        "radiance": L.astype("float32"),
        "cosz": cosz.astype("float32"),
        "reflectance": rho.astype("float32"),
    })
    out_file = Path("data/processed")/f"reflectancia_{fecha_tag}.parquet"
    out.to_parquet(out_file, compression="zstd", index=False)
    print(f"✅ Reflectancia guardada en {out_file} ({len(out)} filas)")
    print(f"ℹ️  Escena {scene_time.isoformat()} | Banda {args.canal.upper()} | celdas con reflectancia válida (cosz>0): "
          f"{int(np.isfinite(out['reflectance']).sum())}/{len(out)}")

if __name__ == "__main__":
    main()
