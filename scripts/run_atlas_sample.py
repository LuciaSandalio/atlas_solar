#!/usr/bin/env python3
"""Run candidate native 2 km physics on sample days, preserving historical products.

Writes hourly/daily hard and blended results, scene-coverage diagnostics, station
comparisons and hash-linked run metadata. Never approves a definitive release.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pvlib
from threadpoolctl import threadpool_limits

from models.atlas_contract import ALIGNMENT, FEATURES, align_option_c, sha256, validate_models
from models.atlas_native import GridSolar, DailyEnergy, ENERGY_COLUMNS, PV_SPECS, AMBIENT, native_features, hourly_candidate


def load_candidate(directory):
    contract = json.loads((directory/'contract.json').read_text())
    if contract['model_layout'] != 'regional_seasonal_3x4' or contract['temporal_alignment'] != ALIGNMENT or contract['features'] != FEATURES:
        raise ValueError('Incompatible candidate feature or alignment contract.')
    if contract['status'] != 'CANDIDATE_NOT_FOR_RELEASE':
        raise ValueError('This command accepts candidate contracts only, not release approvals.')
    for package in ['numpy', 'pandas', 'pvlib', 'sklearn']:
        if contract['versions'][package] != __import__(package).__version__:
            raise ValueError(f'Candidate runtime version changed: {package}')
    fingerprints = {}
    for name, digest in contract['code'].items():
        fingerprints[ROOT/name] = digest
    for item in contract['inputs'].values():
        fingerprints[ROOT/item['path']] = item['sha256']
    fingerprints[directory/'models_regional_seasonal.pkl'] = contract['models_sha256']
    for path, digest in fingerprints.items():
        if sha256(path) != digest:
            raise ValueError(f'Candidate input/code fingerprint changed: {path}')
    models = joblib.load(directory/'models_regional_seasonal.pkl')
    validate_models(models)
    return contract, models, fingerprints


def feature_parity(hourly, aligned, reference, day):
    ref = reference[reference.timestamp_sat.dt.strftime('%Y%m%d') == day]
    if ref.empty:
        return {'status': 'NO_TRAINING_REFERENCE_FOR_THIS_DAY', 'n_pairs': 0}
    ref = ref.rename(columns={'timestamp_sat': 'timestamp'})
    merged = ref.merge(hourly, on=['cell_id', 'timestamp'], suffixes=('_ref', '_native'), validate='one_to_one')
    if len(merged) != len(ref):
        raise ValueError(f'{day}: training reference contains missing native hourly features.')
    cols = ['reflectance', 'B0', 'cosZ']
    errors = {c: float((merged[c+'_ref']-merged[c+'_native']).abs().max()) for c in cols}
    ref_c = align_option_c(ref)
    joined = ref_c.merge(aligned, on=['cell_id', 'timestamp'], suffixes=('_ref', '_native'), validate='one_to_one')
    if len(joined) != len(ref_c):
        raise ValueError(f'{day}: missing option C training reference rows.')
    errors_c = {c: float((joined[c+'_ref']-joined[c+'_native']).abs().max()) for c in FEATURES}
    if any(v > 2e-5 for v in [*errors.values(), *errors_c.values()]):
        raise ValueError(f'{day}: native features differ from training: {errors}; C: {errors_c}')
    return {'status': 'MATCH', 'n_pairs': len(joined), 'max_abs_hourly_errors': errors,
            'max_abs_option_c_errors': errors_c}


def frame_metrics(frame):
    from sklearn.metrics import r2_score
    valid = frame[np.isfinite(frame.target_GHI) & np.isfinite(frame.ghi)]
    if len(valid) < 2:
        return {'n_pairs': len(valid), 'R2': None}
    delta = valid.ghi-valid.target_GHI
    return {'n_pairs': len(valid), 'n_stations': valid.station_id.nunique(),
            'R2': float(r2_score(valid.target_GHI, valid.ghi)),
            'RMSE_W_m2': float(np.sqrt(np.mean(delta**2))),
            'MAE_W_m2': float(np.mean(np.abs(delta))), 'MBE_W_m2': float(delta.mean())}


def compare_stations(contract, observations, output):
    davis = pd.read_parquet(ROOT/contract['inputs']['davis']['path'])
    if 'tiempo_utc' in davis.index.names:
        davis = davis.reset_index()
    davis['timestamp'] = pd.to_datetime(davis.tiempo_utc, utc=True)
    davis['station_id'] = davis.estacion.astype(str)
    davis['target_GHI'] = davis.solar_radiation
    pilar = pd.read_parquet(ROOT/contract['inputs']['pilar']['path'])
    pilar['station_id'] = 'PILAR_CMP21'
    pilar['target_GHI'] = pilar.GHI
    pilar['timestamp'] = pd.to_datetime(pilar.timestamp, utc=True)
    cols = ['station_id', 'timestamp', 'target_GHI']
    ground = pd.concat([davis[cols], pilar[cols]], ignore_index=True)
    ground = ground[ground.timestamp.dt.normalize().isin(observations.timestamp.dt.normalize().unique())]
    if ground.duplicated(['station_id', 'timestamp']).any():
        raise ValueError('Duplicate ground observation during sample validation.')
    merged = observations.merge(ground, on=['station_id', 'timestamp'], how='left', validate='many_to_one')
    merged['evaluation'] = np.where(merged.station_id == 'PILAR_CMP21', 'pilar_excluded_from_training',
                                    np.where(merged.timestamp.dt.year == contract['training_year'],
                                             'training_year_not_independent', 'temporal_holdout'))
    merged.to_parquet(output/'station_pairs.parquet', index=False)
    records = []
    valid = merged[merged.valid_daylight & (merged.target_GHI >= 0)]
    for (mode, evaluation, date), frame in valid.groupby(['mode', 'evaluation', 'date']):
        records.append({'mode': mode, 'evaluation': evaluation, 'date': date, **frame_metrics(frame)})
    pd.DataFrame(records).to_csv(output/'station_metrics.csv', index=False)
    return records


def run(candidate, source, output, days):
    if output.exists():
        raise FileExistsError(f'Use a new sample output directory: {output}')
    if len(days) != len(set(days)):
        raise ValueError('Duplicate sample dates.')
    contract, models, fingerprints = load_candidate(candidate)
    grid = pd.read_parquet(ROOT/contract['inputs']['grid']['path']).sort_values('cell_id').reset_index(drop=True)
    if not grid.cell_id.is_unique:
        raise ValueError('Canonical grid IDs are not unique.')
    coefficients = json.loads((ROOT/contract['inputs']['b0']['path']).read_text())
    reference = pd.read_parquet(ROOT/contract['inputs']['features']['path'])
    mapping = pd.read_parquet(ROOT/contract['inputs']['mapping']['path'])[['station_id', 'cell_id_2km']].rename(columns={'cell_id_2km': 'cell_id'})
    solar = GridSolar(grid)
    for path in solar.source_files:
        fingerprints[path] = sha256(path)
    source_paths = [source/f'reflectancia_{day}.parquet' for day in days]
    for path in source_paths:
        fingerprints[path] = sha256(path)  # fails before creating outputs if a requested day is absent
    output.mkdir(parents=True)
    manifest = {'status': 'RUNNING_CANDIDATE', 'created_at': datetime.now(timezone.utc).isoformat(),
                'candidate_contract_sha256': sha256(candidate/'contract.json'),
                'candidate_contract': str(candidate/'contract.json'), 'days': days,
                'hourly_interval_hours': 1., 'sampling_minutes': 30,
                'native_preprocessing': 'floor_UTC_mean_ref_zenith_azimuth_then_B0_cosZ_then_shared_C',
                'coverage': 'expected independently from sun at interval start/middle/end; low sun unsupported stays missing',
                'low_sun_domain': 'GHI_cs > 20 W/m2 and apparent zenith < 87 deg',
                'pv_specs': PV_SPECS, 'ambient_temperature_celsius_by_season': AMBIENT,
                'validation_scope': 'Sample days only; no annual extrapolation or release approval.',
                'limitations': [*contract['release_blockers'],
                                'Physical/PV model parameters fixed, ambient temperature is seasonal approximation.',
                                'Legacy native reflectance provenance and DQF filtering are not certified.',
                                'Both endpoints need two half-hour scenes; missing/low-sun intervals are not zero.'],
                'code': {name: sha256(ROOT/name) for name in ['models/atlas_native.py', 'scripts/run_atlas_sample.py']},
                'input_hashes': {str(p): digest for p, digest in fingerprints.items()}, 'diagnostics': []}
    def save_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    save_manifest()
    observations = []
    try:
        for day, path in zip(days, source_paths):
            raw = pd.read_parquet(path)
            actual_day = pd.to_datetime(raw.timestamp).dt.strftime('%Y%m%d')
            if raw.empty or not (actual_day == day).all():
                raise ValueError(f'{path}: timestamps do not match requested day.')
            hourly, aligned = native_features(raw, coefficients, grid)
            parity = feature_parity(hourly, aligned, reference, day)
            accumulators = {mode: DailyEnergy(grid.cell_id.to_numpy()) for mode in ('hard', 'blended')}
            writers = {}
            qas = {mode: {'radiative_closure_max_W_m2': 0., 'negative_count': 0} for mode in accumulators}
            try:
                for mode, frame in hourly_candidate(grid, models, aligned, solar, day):
                    values = frame[list(ENERGY_COLUMNS)].to_numpy()
                    qas[mode]['negative_count'] += int(np.count_nonzero(values < -1e-8))
                    valid = frame.valid_daylight.to_numpy()
                    if valid.any():
                        closure = np.abs(frame.ghi[valid] - frame.dni[valid]*np.cos(np.radians(frame.zenith_deg[valid])) - frame.dhi[valid])
                        qas[mode]['radiative_closure_max_W_m2'] = max(qas[mode]['radiative_closure_max_W_m2'], float(closure.max()))
                    accumulators[mode].add(frame)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if mode not in writers:
                        metadata = dict(table.schema.metadata or {})
                        metadata[b'atlas_candidate_run'] = json.dumps({'candidate_contract_sha256': manifest['candidate_contract_sha256'],
                                                                      'source_sha256': fingerprints[path], 'mode': mode,
                                                                      'status': 'CANDIDATE_NOT_FOR_RELEASE'}).encode()
                        writers[mode] = pq.ParquetWriter(output/f'hourly_{day}_{mode}.parquet.tmp',
                                                        table.schema.with_metadata(metadata), compression='zstd')
                    writers[mode].write_table(table)
                    station_frame = mapping.merge(frame, on='cell_id', validate='many_to_one')
                    station_frame['mode'], station_frame['date'] = mode, day
                    observations.append(station_frame)
            finally:
                for writer in writers.values():
                    writer.close()
            daily = {}
            for mode, accumulator in accumulators.items():
                if qas[mode]['negative_count'] or qas[mode]['radiative_closure_max_W_m2'] > 1e-5:
                    raise ValueError(f'Physical QA failed: {mode}, {qas[mode]}')
                frame = grid[['cell_id', 'lat', 'lon', 'region', 'in_province', 'weight_R0', 'weight_R1', 'weight_R2']].merge(
                    accumulator.finish(), on='cell_id', validate='one_to_one')
                frame['date'] = day
                frame['candidate_contract_sha256'] = manifest['candidate_contract_sha256']
                frame['run_status'] = 'CANDIDATE_NOT_FOR_RELEASE'
                frame.to_parquet(output/f'daily_{day}_{mode}.parquet', index=False)
                daily[mode] = frame
                qas[mode]['coverage_mean_pct'] = float(frame.coverage_pct.mean())
                qas[mode]['coverage_min_pct'] = float(frame.coverage_pct.min())
                qas[mode]['complete_cells'] = int(frame.complete_day.sum())
                (output/f'hourly_{day}_{mode}.parquet.tmp').rename(output/f'hourly_{day}_{mode}.parquet')
            difference = daily['blended'].observed_ghi_kwh_m2-daily['hard'].observed_ghi_kwh_m2
            diag = {'date': day, 'feature_parity': parity, 'modes': qas,
                    'blended_minus_hard_observed_ghi_kwh_m2_mean': float(difference.mean()),
                    'blended_minus_hard_observed_ghi_kwh_m2_max_abs': float(difference.abs().max())}
            manifest['diagnostics'].append(diag)
            save_manifest()
            print(f'{day}: parity={parity["status"]}; coverage={qas["blended"]["coverage_mean_pct"]:.1f}%; '
                  f'closure={qas["blended"]["radiative_closure_max_W_m2"]:.2e} W/m²', flush=True)
        manifest['station_metrics'] = compare_stations(contract, pd.concat(observations, ignore_index=True), output)
        for path, digest in fingerprints.items():
            if sha256(path) != digest:
                raise ValueError(f'Input changed during sample inference: {path}')
        manifest['artifacts'] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.name != 'manifest.json'}
        manifest['status'] = 'SAMPLE_EXECUTED_NOT_FOR_RELEASE'
        manifest['completed_at'] = datetime.now(timezone.utc).isoformat()
        save_manifest()
    except Exception as exc:
        manifest['status'] = 'FAILED_CANDIDATE_RUN'
        manifest['error'] = str(exc)
        save_manifest()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, default=ROOT/'data/ready/atlas_candidate_20261005_v2')
    parser.add_argument('--source', type=Path, default=ROOT/'data/processed_2km')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--days', nargs='+', required=True, help='UTC dates YYYYMMDD')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.candidate, args.source, args.output, args.days)
