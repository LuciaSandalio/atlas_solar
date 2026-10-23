"""Shared, strict conventions for the regional Atlas pipeline.

Legacy/global models and untraceable products must not be promoted to definitive.
Candidate experiments are intentionally separate from release approval.
"""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from models.bdjpt import HybridBDJPT_HGB

FEATURES = ['reflectance', 'B0', 'Fr', 'cosZ', 'day_of_year']
SEASONS = ('Verano', 'Otono', 'Invierno', 'Primavera')
ALIGNMENT = 'C_hourly_round_mean_consecutive_same_utc_day_v1'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def season(month):
    return SEASONS[((month % 12) // 3)]


def align_option_c(frame, timestamp='timestamp'):
    """Round scenes to UTC hour, then mean(t,t+1) for ref/B0/cosZ.

    Missing endpoints, nonfinite rows and midnight pairs produce no interval.
    Fr is computed AFTER averaging. Output timestamp is the first endpoint.
    No rounding or endpoint fallback may differ between training and inference.
    """
    cols = ['reflectance', 'B0', 'cosZ']
    data = frame[['cell_id', timestamp, *cols]].copy()
    times = pd.to_datetime(data[timestamp])
    if times.dt.tz is None:
        raise ValueError('Scene timestamps must be timezone-aware (UTC).')
    if data.duplicated(['cell_id', timestamp]).any():
        raise ValueError('Duplicate cell/timestamp observations.')
    data['timestamp'] = times.dt.tz_convert('UTC').dt.round('h')
    # A missing observation must not disappear through pandas mean(skipna=True).
    data[cols] = data[cols].replace([np.inf, -np.inf], np.nan)
    grouped = data.groupby(['cell_id', 'timestamp'])[cols]
    hourly = grouped.mean().where(grouped.count().eq(grouped.size(), axis=0))
    hourly = hourly.reset_index().sort_values(['cell_id', 'timestamp'])
    following = hourly.groupby('cell_id', sort=False).shift(-1)
    valid = ((following.timestamp - hourly.timestamp) == pd.Timedelta(hours=1))
    valid &= following.timestamp.dt.normalize() == hourly.timestamp.dt.normalize()
    out = hourly.loc[valid].copy()
    out[cols] = (hourly.loc[valid, cols] + following.loc[valid, cols]) / 2
    out = out.dropna(subset=cols)
    out['Fr'] = np.maximum(out.reflectance**2 - out.B0**2, 0)
    out['day_of_year'] = out.timestamp.dt.dayofyear
    return out.reset_index(drop=True)


def validate_models(models):
    if not isinstance(models, dict) or set(models) != {0, 1, 2}:
        raise ValueError('Required: dictionary of 3 regions × 4 seasons; global/4-region model rejected.')
    for region in range(3):
        if set(models[region]) != set(SEASONS):
            raise ValueError(f'Region {region}: four complete seasons required.')
        for name in SEASONS:
            entry = models[region][name]
            if not isinstance(entry, dict) or not isinstance(entry.get('hybrid'), HybridBDJPT_HGB):
                raise ValueError(f'Region {region}/{name}: HybridBDJPT_HGB required.')
            if getattr(entry['hybrid'].hgb, 'n_features_in_', None) != 6:
                raise ValueError(f'Region {region}/{name}: model not fitted with five inputs + baseline.')


def predict_regional(models, name, features, clear_sky, weights, mode='blended'):
    validate_models(models)
    x = np.asarray(features)
    cs = np.asarray(clear_sky)
    w = np.asarray(weights, dtype=float)
    if mode not in ('blended', 'hard'):
        raise ValueError('Mode must be blended or hard.')
    if x.ndim != 2 or x.shape[1] != 5 or cs.shape != (len(x),) or w.shape != (len(x), 3):
        raise ValueError('Incompatible model input dimensions.')
    if not np.isfinite(w).all() or (w < 0).any() or not np.allclose(w.sum(1), 1, atol=1e-3):
        raise ValueError('Regional weights must be finite, nonnegative and normalized.')
    # Account for the four-decimal rounding in the stored 3-NN grid.
    w = w / w.sum(1, keepdims=True)
    if mode == 'hard':
        w = np.eye(3)[w.argmax(1)]
    valid = np.isfinite(x).all(1) & np.isfinite(cs) & (cs > 0)
    result = np.full(len(x), np.nan)
    result[valid] = 0
    for region in range(3):
        mask = valid & (w[:, region] > 0)
        if mask.any():
            result[mask] += w[mask, region] * models[region][name]['hybrid'].predict(x[mask], cs[mask])
    return np.clip(result, 0, 1.15)


def reject_legacy(stage):
    """Legacy stages cannot satisfy annual, native-grid, traceable release rules."""
    raise RuntimeError(
        f'{stage}: definitive production blocked. The legacy pipeline does not '
        'implement the verified native 2 km / option C / annual-product contract. '
        'Run python scripts/audit_atlas_release.py for blockers. '
        'Existing products are preserved, not certified.'
    )
