#!/usr/bin/env python3
"""Reproducible 3 × 4 candidate, never an automatic definitive approval.

Train on 2020, grouped spatial CV within 2020, temporal holdout 2021 and
independent Pilar holdout. Historical native features lack raw-scene lineage;
that limitation is explicitly carried into the candidate contract.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pvlib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

from models.atlas_contract import ALIGNMENT, FEATURES, SEASONS, align_option_c, season, sha256, validate_models
from models.bdjpt import HybridBDJPT_HGB

INPUTS = {
    'features': 'data/ready/b0_model/station_features_B0_2km_dev.parquet',
    'b0': 'data/ready/b0_model/B0_params_grid2km_dev.json',
    'grid': 'data/ready/cordoba_grid_2km_regions.parquet',
    'mapping': 'data/ready/station_to_grid_2km.parquet',
    'clusters': 'data/processed/stations/davis_solar_clusters.gpkg',
    'davis': 'data/processed/stations/davis_calibrated_hourly_final.parquet',
    'pilar': 'data/processed/stations/pilar_cmp21_hourly.parquet',
}


def metrics(actual, predicted):
    if len(actual) < 2:
        return {'n_pairs': int(len(actual)), 'R2': None, 'RMSE': None, 'MAE': None, 'MBE': None}
    delta = np.asarray(predicted) - np.asarray(actual)
    return {'n_pairs': len(delta), 'R2': float(r2_score(actual, predicted)),
            'RMSE': float(np.sqrt(np.mean(delta**2))), 'MAE': float(np.abs(delta).mean()),
            'MBE': float(delta.mean())}


def prepare():
    grid = pd.read_parquet(ROOT / INPUTS['grid']).set_index('cell_id')
    mapping = pd.read_parquet(ROOT / INPUTS['mapping']).set_index('station_id')
    if not mapping.index.is_unique:
        raise ValueError('Duplicate station mapping.')
    clusters = gpd.read_file(ROOT / INPUTS['clusters']).set_index('estacion')['cluster']
    if not clusters.index.is_unique or set(clusters.unique()) != {0, 1, 2}:
        raise ValueError('Expected unique stations in three current clusters.')
    # Old mapping has FOUR-region labels: use it only for canonical cell IDs.
    for _, row in mapping.iterrows():
        cell = grid.loc[row.cell_id_2km]
        if not np.allclose([cell.lat, cell.lon], [row.lat_cell, row.lon_cell], atol=1e-6, rtol=0):
            raise ValueError('Station mapping does not match canonical grid coordinates.')
    sat = pd.read_parquet(ROOT / INPUTS['features'])
    b0 = json.loads((ROOT / INPUTS['b0']).read_text())
    zen = np.radians(sat.zenith_deg.to_numpy())
    az = np.cos(np.radians(sat.azimuth_deg.to_numpy()))
    expected_b0 = np.maximum(b0['A'] + b0['B']*np.cos(zen) + b0['C']*np.sin(zen)*az
                             + b0['D']*np.sin(zen)*az**2, 0.01)
    if not np.allclose(expected_b0, sat.B0, atol=2e-5, rtol=0):
        raise ValueError('Native features do not reproduce the supplied B0 coefficients.')
    aligned = align_option_c(sat, 'timestamp_sat')
    davis = pd.read_parquet(ROOT / INPUTS['davis'])
    if 'tiempo_utc' in davis.index.names:
        davis = davis.reset_index()
    davis['station_id'] = davis.estacion.astype(str)
    davis['timestamp'] = pd.to_datetime(davis.tiempo_utc, utc=True)
    davis['target_GHI'] = davis.solar_radiation
    pilar = pd.read_parquet(ROOT / INPUTS['pilar'])
    pilar['station_id'] = 'PILAR_CMP21'
    pilar['target_GHI'] = pilar.GHI
    pilar['timestamp'] = pd.to_datetime(pilar.timestamp, utc=True)
    ground = pd.concat([davis[['station_id', 'timestamp', 'target_GHI']],
                        pilar[['station_id', 'timestamp', 'target_GHI']]], ignore_index=True)
    ground = ground[ground.timestamp.dt.year.isin([2020, 2021])].copy()
    ground['cell_id'] = ground.station_id.map(mapping.cell_id_2km)
    ground['region'] = ground.station_id.map(clusters)
    is_pilar = ground.station_id == 'PILAR_CMP21'
    ground.loc[is_pilar, 'region'] = int(grid.loc[mapping.loc['PILAR_CMP21', 'cell_id_2km'], 'region'])
    unmapped = ground.loc[ground.cell_id.isna() | ground.region.isna(), 'station_id'].unique()
    if len(unmapped):
        raise ValueError(f'Unmapped stations (no silent exclusion): {unmapped.tolist()}')
    ground = ground.dropna(subset=['target_GHI'])
    ground = ground[ground.target_GHI >= 0]
    if ground.duplicated(['station_id', 'timestamp']).any():
        raise ValueError('Duplicate ground station/hour.')
    data = ground.merge(aligned, on=['cell_id', 'timestamp'], validate='many_to_one')
    data['ghi_cs'] = np.nan
    for cell_id, indices in data.groupby('cell_id').groups.items():
        cell = grid.loc[cell_id]
        times = pd.DatetimeIndex(data.loc[indices, 'timestamp'])
        loc = pvlib.location.Location(float(cell.lat), float(cell.lon))
        data.loc[indices, 'ghi_cs'] = loc.get_clearsky(times, model='ineichen')['ghi'].to_numpy()
    data = data[np.isfinite(data.ghi_cs) & (data.ghi_cs > 20)].copy()
    data['y_kt'] = (data.target_GHI / data.ghi_cs).clip(0, 1.15)
    data['season'] = data.timestamp.dt.month.map(season)
    data['region'] = data.region.astype(int)
    return data


def fit_model(data):
    return HybridBDJPT_HGB(max_iter=100, max_depth=12, random_state=42).fit(
        data[FEATURES].to_numpy(), data.y_kt.to_numpy(),
        data.target_GHI.to_numpy(), data.ghi_cs.to_numpy())


def predict(model, data):
    return model.predict(data[FEATURES].to_numpy(), data.ghi_cs.to_numpy()) * data.ghi_cs.to_numpy()


def run(output):
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    inputs = {key: {'path': value, 'sha256': sha256(ROOT / value)} for key, value in INPUTS.items()}
    data = prepare()
    print(f'Aligned pairs: {len(data):,}', flush=True)
    models = {r: {} for r in range(3)}
    scores, predictions = [], []
    # Preflight all twelve configurations before any training.
    training = data[(data.timestamp.dt.year == 2020) & (data.station_id != 'PILAR_CMP21')]
    for r in range(3):
        for s in SEASONS:
            part = training[(training.region == r) & (training.season == s)]
            if len(part) < 100 or part.station_id.nunique() < 3:
                raise ValueError(f'Insufficient independent training groups: {r}/{s}')
    for r in range(3):
        for s in SEASONS:
            train = training[(training.region == r) & (training.season == s)].reset_index(drop=True)
            cv = GroupKFold(n_splits=3)
            oof = np.full(len(train), np.nan)
            for fold, (tr, te) in enumerate(cv.split(train, groups=train.station_id)):
                model = fit_model(train.iloc[tr])
                oof[te] = predict(model, train.iloc[te])
                pred = train.iloc[te][['station_id', 'timestamp', 'region', 'season', 'target_GHI']].copy()
                pred['predicted_GHI'] = oof[te]
                pred['fold'] = fold
                pred['evaluation'] = 'spatial_oof_2020'
                predictions.append(pred)
            model = fit_model(train)
            models[r][s] = {'hybrid': model}
            scores.append({'region': r, 'season': s, 'evaluation': 'spatial_oof_2020',
                           'n_stations': train.station_id.nunique(), **metrics(train.target_GHI, oof)})
            for label, mask in [('temporal_holdout_2021', (data.timestamp.dt.year == 2021) & (data.station_id != 'PILAR_CMP21')),
                                ('pilar_independent', data.station_id == 'PILAR_CMP21')]:
                test = data[mask & (data.region == r) & (data.season == s)]
                values = predict(model, test) if len(test) else np.array([])
                scores.append({'region': r, 'season': s, 'evaluation': label,
                               'n_stations': test.station_id.nunique(), **metrics(test.target_GHI, values)})
                if len(test):
                    pred = test[['station_id', 'timestamp', 'region', 'season', 'target_GHI']].copy()
                    pred['predicted_GHI'] = values
                    pred['fold'] = -1
                    pred['evaluation'] = label
                    predictions.append(pred)
            print(f'R{r}/{s}: n={len(train):,}, spatial R²={scores[-3]["R2"]:.4f}', flush=True)
    validate_models(models)
    for item in inputs.values():
        if sha256(ROOT / item['path']) != item['sha256']:
            raise RuntimeError(f'Input changed during training: {item["path"]}')
    output.mkdir(parents=True)
    model_file = output / 'models_regional_seasonal.pkl'
    joblib.dump(models, model_file)
    pd.DataFrame(scores).to_csv(output / 'metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(output / 'predictions.parquet', index=False)
    contract = {
        'status': 'CANDIDATE_NOT_FOR_RELEASE', 'created_at': datetime.now(timezone.utc).isoformat(),
        'model_layout': 'regional_seasonal_3x4', 'features': FEATURES,
        'temporal_alignment': ALIGNMENT, 'models_sha256': sha256(model_file),
        'inputs': inputs, 'training_year': 2020, 'temporal_holdout_year': 2021,
        'clear_sky': 'pvlib Location(grid lat/lon).get_clearsky: Ineichen defaults',
        'spatial_cv': 'GroupKFold(3), grouped by station, within 2020',
        'pilar': 'excluded from every fit; evaluated independently in its mapped grid region',
        'target': 'GHI/GHI_cs, clipped [0,1.15] for fitting; GHI metrics use unclipped ground truth',
        'release_blockers': ['Raw scene/projection provenance missing for historical native features.',
                             'B0 calibration not nested inside validation folds; metrics conditional on fixed B0.',
                             'Regionalization itself is precomputed, not nested inside validation folds.',
                             'No agreed, passed independent release QA; annual production not certified.'],
        'code': {str(p.relative_to(ROOT)): sha256(p) for p in
                 [Path(__file__).resolve(), ROOT/'models/atlas_contract.py', ROOT/'models/bdjpt.py']},
        'versions': {'python': sys.version, 'numpy': np.__version__, 'pandas': pd.__version__,
                     'pvlib': pvlib.__version__, 'sklearn': __import__('sklearn').__version__},
        'artifacts': {name: sha256(output/name) for name in ['metrics.csv', 'predictions.parquet']},
    }
    (output/'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    print(f'Candidate only, no production files overwritten: {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output)
