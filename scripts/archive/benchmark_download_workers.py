#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de Throughput Agregado de Descarga GOES-16 desde AWS S3
================================================================
Mide el ancho de banda efectivo total (MB/s) variando el número de
descargas concurrentes (1, 2, 4, 8 workers) para encontrar el punto óptimo
de saturación sin provocar sobrecarga.
"""

import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore import UNSIGNED
from botocore.config import Config

ROOT_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT_DIR / "data" / ".tmp_download_bench"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# 4 archivos de prueba reales de NOAA S3
SAMPLE_KEYS = [
    "ABI-L1b-RadF/2021/015/12/OR_ABI-L1b-RadF-M6C02_G16_s20210151200193_e20210151209501_c20210151209545.nc",
    "ABI-L1b-RadF/2021/015/13/OR_ABI-L1b-RadF-M6C02_G16_s20210151300193_e20210151309501_c20210151309534.nc",
    "ABI-L1b-RadF/2021/015/14/OR_ABI-L1b-RadF-M6C02_G16_s20210151400193_e20210151409501_c20210151409541.nc",
    "ABI-L1b-RadF/2021/015/15/OR_ABI-L1b-RadF-M6C02_G16_s20210151500193_e20210151509501_c20210151509540.nc",
    "ABI-L1b-RadF/2021/015/16/OR_ABI-L1b-RadF-M6C02_G16_s20210151600194_e20210151609502_c20210151609542.nc",
    "ABI-L1b-RadF/2021/015/17/OR_ABI-L1b-RadF-M6C02_G16_s20210151700194_e20210151709502_c20210151709544.nc",
    "ABI-L1b-RadF/2021/015/18/OR_ABI-L1b-RadF-M6C02_G16_s20210151800192_e20210151809500_c20210151809541.nc",
    "ABI-L1b-RadF/2021/015/19/OR_ABI-L1b-RadF-M6C02_G16_s20210151900192_e20210151909500_c20210151909541.nc",
]
BUCKET = "noaa-goes16"


def download_single(key: str, worker_id: int):
    s3c = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    dest = TMP_DIR / f"test_{worker_id}_{Path(key).name}"
    try:
        t0 = time.time()
        s3c.download_file(BUCKET, key, str(dest))
        el = time.time() - t0
        sz_mb = dest.stat().st_size / (1024 * 1024)
        return sz_mb, el
    finally:
        if dest.exists():
            dest.unlink(missing_ok=True)


def run_benchmark(n_workers: int, n_files: int = 4):
    keys = SAMPLE_KEYS[:n_files]
    print(f"\n⚡ Probando {n_workers} worker(s) descargando {len(keys)} archivos...")
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(download_single, key, i) for i, key in enumerate(keys)]
        results = [f.result() for f in futures]

    total_time = time.time() - t_start
    total_mb = sum(r[0] for r in results)
    throughput = total_mb / total_time

    print(f"   • Total transferido: {total_mb:.1f} MB en {total_time:.2f} s")
    print(f"   • Throughput AGREGADO ({n_workers} workers): {throughput:.2f} MB/s ({throughput*8:.1f} Mbps)")
    return throughput


def main():
    print("=" * 70)
    print("🚀 BENCHMARK DE CONCURRENCIA S3: MEDIDAS DE THROUGHPUT AGREGADO")
    print(f"📂 Directorio temporal seguro (132 GB disponibles): {TMP_DIR}")
    print("=" * 70)

    results = {}
    for n in [1, 2, 4]:
        results[n] = run_benchmark(n, n_files=4)

    print("\n" + "=" * 70)
    print("📊 RESUMEN DE ESCALABILIDAD:")
    print(f"{'Workers':<10} | {'Throughput Total':<20} | {'Ganancia vs 1 Worker':<20}")
    print("-" * 70)
    base = results[1]
    for n in sorted(results.keys()):
        tp = results[n]
        gain = tp / base if base > 0 else 0
        print(f"{n:<10} | {tp:6.2f} MB/s ({tp*8:5.1f} Mbps) | {gain:4.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
