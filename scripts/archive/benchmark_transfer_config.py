#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark de TransferConfig y Concurrencia de Red para Descargas S3 GOES-16
==========================================================================
Evalúa combinaciones de (workers x max_concurrency_por_archivo) para
maximizar el throughput agregado (MB/s) hacia AWS S3.
"""

import time
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
TMP_DIR = ROOT_DIR / "data" / ".tmp_bench_transfer"
TMP_DIR.mkdir(parents=True, exist_ok=True)

BUCKET = "noaa-goes16"

# 8 archivos NetCDF reales de NOAA S3
TEST_KEYS = [
    "ABI-L1b-RadF/2021/015/12/OR_ABI-L1b-RadF-M6C02_G16_s20210151200193_e20210151209501_c20210151209545.nc",
    "ABI-L1b-RadF/2021/015/13/OR_ABI-L1b-RadF-M6C02_G16_s20210151300193_e20210151309501_c20210151309534.nc",
    "ABI-L1b-RadF/2021/015/14/OR_ABI-L1b-RadF-M6C02_G16_s20210151400193_e20210151409501_c20210151409541.nc",
    "ABI-L1b-RadF/2021/015/15/OR_ABI-L1b-RadF-M6C02_G16_s20210151500193_e20210151509501_c20210151509540.nc",
    "ABI-L1b-RadF/2021/015/16/OR_ABI-L1b-RadF-M6C02_G16_s20210151600194_e20210151609502_c20210151609542.nc",
    "ABI-L1b-RadF/2021/015/17/OR_ABI-L1b-RadF-M6C02_G16_s20210151700194_e20210151709502_c20210151709544.nc",
    "ABI-L1b-RadF/2021/015/18/OR_ABI-L1b-RadF-M6C02_G16_s20210151800192_e20210151809500_c20210151809541.nc",
    "ABI-L1b-RadF/2021/015/19/OR_ABI-L1b-RadF-M6C02_G16_s20210151900192_e20210151909500_c20210151909541.nc",
]

_BENCH_CLIENT = None
_BENCH_TRANSFER_CONFIG = None


def init_bench_worker(concurrency_per_file: int):
    global _BENCH_CLIENT, _BENCH_TRANSFER_CONFIG
    _BENCH_CLIENT = boto3.client(
        "s3",
        config=Config(
            signature_version=UNSIGNED,
            retries={"max_attempts": 5, "mode": "adaptive"},
            max_pool_connections=32,
            tcp_keepalive=True,
        )
    )
    _BENCH_TRANSFER_CONFIG = TransferConfig(
        multipart_threshold=8 * 1024**2,
        multipart_chunksize=16 * 1024**2,
        max_concurrency=concurrency_per_file,
        use_threads=True,
    )


def download_task(key: str, task_id: int):
    global _BENCH_CLIENT, _BENCH_TRANSFER_CONFIG
    pid = os.getpid()
    dest = TMP_DIR / f"test_{pid}_{task_id}_{Path(key).name}"
    try:
        t0 = time.time()
        _BENCH_CLIENT.download_file(
            BUCKET,
            key,
            str(dest),
            Config=_BENCH_TRANSFER_CONFIG,
        )
        elapsed = time.time() - t0
        sz_mb = dest.stat().st_size / (1024 * 1024)
        return sz_mb, elapsed
    finally:
        if dest.exists():
            dest.unlink(missing_ok=True)


def test_configuration(n_workers: int, concurrency_per_file: int, n_files: int = 4):
    keys = TEST_KEYS[:n_files]
    print(f"\n⚡ Probando: {n_workers} workers × {concurrency_per_file} threads/archivo (Total: {n_workers * concurrency_per_file} conexiones)...")

    t_start = time.time()
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_bench_worker,
        initargs=(concurrency_per_file,)
    ) as executor:
        futures = [executor.submit(download_task, key, i) for i, key in enumerate(keys)]
        results = [f.result() for f in futures]

    total_time = time.time() - t_start
    total_mb = sum(r[0] for r in results)
    tp = total_mb / total_time
    print(f"   • {total_mb:.1f} MB en {total_time:.2f} s | Throughput: {tp:6.2f} MB/s ({tp*8:5.1f} Mbps)")
    return tp


def main():
    print("=" * 75)
    print("🚀 BENCHMARK DE S3 TRANSFER CONFIG (ANCHO DE BANDA AGREGADO)")
    print("=" * 75)

    matrix = [
        (2, 2),
        (2, 4),
        (4, 2),
        (4, 4),
        (6, 2),
    ]

    results = {}
    for w, c in matrix:
        tp = test_configuration(w, c, n_files=4)
        results[(w, c)] = tp

    print("\n" + "=" * 75)
    print("📊 MATRIZ DE RENDIMIENTO S3:")
    print(f"{'Workers':<8} | {'Threads/Archivo':<18} | {'Conexiones Totales':<20} | {'Throughput Agregado':<20}")
    print("-" * 75)
    best_config = None
    best_tp = 0.0
    for (w, c), tp in results.items():
        if tp > best_tp:
            best_tp = tp
            best_config = (w, c)
        print(f"{w:<8} | {c:<18} | {w*c:<20} | {tp:6.2f} MB/s ({tp*8:5.1f} Mbps)")
    print("=" * 75)
    print(f"🏆 CONFIGURACIÓN ÓPTIMA: {best_config[0]} workers × {best_config[1]} threads ({best_tp:.2f} MB/s)")


if __name__ == "__main__":
    main()
