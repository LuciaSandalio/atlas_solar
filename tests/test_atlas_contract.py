"""Regression tests for the incompatibilities discovered in the release audit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from models.atlas_contract import align_option_c, predict_regional, validate_models, SEASONS
from models.bdjpt import HybridBDJPT_HGB
from scripts.fase_1_b_mass_grid2km import scan_coordinates, is_current_extraction


def scenes(times, reflectance=None):
    n = len(times)
    return pd.DataFrame({'cell_id': [17]*n, 'timestamp': pd.to_datetime(times, utc=True),
                         'reflectance': reflectance if reflectance is not None else np.arange(n)*0.2,
                         'B0': [0.1]*n, 'cosZ': [0.8]*n})


def test_option_c_averages_endpoints_and_then_computes_contrast():
    frame = scenes(['2020-01-01 12:00', '2020-01-01 13:00'], [0.2, 0.6])
    out = align_option_c(frame)
    assert len(out) == 1
    assert out.timestamp.iloc[0] == pd.Timestamp('2020-01-01 12:00Z')
    assert out.reflectance.iloc[0] == pytest.approx(0.4)
    assert out.Fr.iloc[0] == pytest.approx(0.15)  # not mean(Fr(t), Fr(t+1))


@pytest.mark.parametrize('times', [
    ['2020-01-01 12:00', '2020-01-01 14:00'],
    ['2020-01-01 23:00', '2020-01-02 00:00'],
])
def test_option_c_rejects_gaps_and_day_boundaries(times):
    assert align_option_c(scenes(times)).empty


def test_missing_endpoint_cannot_become_zero_or_single_scene_fallback():
    assert align_option_c(scenes(['2020-01-01 12:00', '2020-01-01 13:00'], [0.3, np.nan])).empty


def test_scene_rounding_allows_distinct_scenes_but_rejects_duplicates():
    frame = scenes(['2020-01-01 12:00', '2020-01-01 12:10', '2020-01-01 13:00'])
    assert len(align_option_c(frame)) == 1
    with pytest.raises(ValueError, match='Duplicate'):
        align_option_c(pd.concat([frame, frame.iloc[:1]]))


def test_naive_time_is_not_silently_localized():
    frame = scenes(['2020-01-01 12:00', '2020-01-01 13:00'])
    frame['timestamp'] = frame.timestamp.dt.tz_localize(None)
    with pytest.raises(ValueError, match='timezone-aware'):
        align_option_c(frame)


@pytest.mark.parametrize('model', [HybridBDJPT_HGB(), {r: {} for r in range(4)}, {r: {} for r in range(3)}])
def test_global_old_regional_and_incomplete_models_are_rejected(model):
    with pytest.raises(ValueError):
        validate_models(model)


class ConstantHybrid(HybridBDJPT_HGB):
    def __init__(self, value):
        super().__init__()
        self.hgb.n_features_in_ = 6
        self.value = value

    def predict(self, features, cs):
        return np.full(len(features), self.value)


def test_blending_uses_every_weight_and_preserves_missing_data():
    models = {r: {s: {'hybrid': ConstantHybrid(v)} for s in SEASONS}
              for r, v in enumerate([0.2, 0.6, 1.0])}
    x = np.ones((2, 5)); x[1, 0] = np.nan
    weights = np.array([[0.25, 0.5, 0.25]]*2)
    pred = predict_regional(models, 'Verano', x, np.array([800, 800]), weights)
    assert pred[0] == pytest.approx(0.6)
    assert np.isnan(pred[1])
    with pytest.raises(ValueError, match='weights'):
        predict_regional(models, 'Verano', x, np.ones(2), weights*0)


def projection_scene(longitude):
    return xr.Dataset({'Rad': xr.DataArray(np.zeros((2, 2)), dims=['y', 'x'],
                                          attrs={'grid_mapping': 'projection'}),
                       'projection': xr.DataArray(0, attrs={
                           'grid_mapping_name': 'geostationary', 'perspective_point_height': 35786023.0,
                           'longitude_of_projection_origin': longitude, 'latitude_of_projection_origin': 0.0,
                           'semi_major_axis': 6378137.0, 'semi_minor_axis': 6356752.31414,
                           'sweep_angle_axis': 'x'})},
                      coords={'x': xr.DataArray([0, 0.01], dims='x', attrs={'units': 'rad'}),
                              'y': xr.DataArray([0, 0.01], dims='y', attrs={'units': 'rad'})})


def test_projection_comes_from_each_netcdf():
    x1, y1, _ = scan_coordinates(projection_scene(-75), [-75.0], [0.0])
    x2, _, _ = scan_coordinates(projection_scene(-89.5), [-75.0], [0.0])
    assert x1[0] == pytest.approx(0, abs=1e-10)
    assert y1[0] == pytest.approx(0, abs=1e-10)
    assert abs(x2[0]) > 0.01


def test_missing_projection_fails_instead_of_assuming_goes16():
    ds = projection_scene(-75)
    del ds.projection.attrs['sweep_angle_axis']
    with pytest.raises(ValueError, match='sweep_angle_axis'):
        scan_coordinates(ds, [-64], [-32])


def test_old_parquet_is_never_treated_as_current_extraction(tmp_path):
    path = tmp_path/'old.parquet'
    pd.DataFrame({'cell_id': [1]}).to_parquet(path)
    assert not is_current_extraction(path, 'any-grid', 30)


def test_extraction_captures_scene_identity_and_projection(tmp_path, monkeypatch):
    from scripts import fase_1_b_mass_grid2km as extraction
    ds = projection_scene(-75).reindex(x=np.linspace(-0.2, 0.2, 21),
                                      y=np.linspace(-0.2, 0.2, 21), fill_value=10)
    ds.x.attrs['units'] = 'rad'
    ds.y.attrs['units'] = 'rad'
    ds['Rad'] = xr.DataArray(np.full((21, 21), 10.), dims=['y', 'x'],
                             attrs={'grid_mapping': 'projection'})
    ds['band_id'] = xr.DataArray(2)
    ds.attrs['platform_ID'] = 'G19'
    source = tmp_path/'scene.nc'
    ds.to_netcdf(source, engine='h5netcdf')
    monkeypatch.setattr(extraction, '_WORKER_GRID_DF',
                        pd.DataFrame({'cell_id': [7], 'lon': [-64.], 'lat': [-32.]}))
    result = extraction.extract_scene_from_local_netcdf(source, pd.Timestamp('2025-05-01 16:00Z'))
    assert result.satellite.tolist() == ['G19']
    assert result.source_file.tolist() == ['scene.nc']
    assert np.isfinite(result.reflectance).all()
    assert result.attrs['scene_provenance']['sha256'] == extraction.file_sha256(source)
    assert result.attrs['scene_provenance']['projection']['longitude_of_projection_origin'] == -75


def test_published_day_has_verifiable_metadata_and_cannot_be_reused_after_tampering(tmp_path, monkeypatch):
    from scripts import fase_1_b_mass_grid2km as extraction
    day = '20200101'
    timestamp = pd.Timestamp('2020-01-01 16:00Z')
    monkeypatch.setattr(extraction, '_WORKER_GRID_SHA256', 'grid-hash')
    monkeypatch.setattr(extraction, '_WORKER_GRID_DF', pd.DataFrame({'cell_id': [1]}))
    monkeypatch.setattr(extraction, 'get_daylight_scenes_for_day', lambda *a, **k:
                        pd.DataFrame({'url': ['s3://noaa-goes16/scene.nc'], 'ts': [timestamp]}))
    def download(bucket, key, dest):
        dest.write_bytes(b'local-test-scene')
        return True
    monkeypatch.setattr(extraction, 'download_with_retry_and_verify', download)
    def extract(path, time):
        frame = pd.DataFrame({'cell_id': [1], 'lon': [-64.], 'lat': [-32.],
                              'radiance': [10.], 'zenith_deg': [30.], 'azimuth_deg': [45.],
                              'reflectance': [0.2], 'timestamp': [time],
                              'satellite': ['G16'], 'source_file': ['scene.nc']})
        frame.attrs['scene_provenance'] = {'source_file': path.name, 'satellite': 'G16',
                                           'sha256': extraction.file_sha256(path)}
        return frame
    monkeypatch.setattr(extraction, 'extract_scene_from_local_netcdf', extract)
    result = extraction.process_single_day_worker(day, tmp_path, tmp_path/'download')
    assert result['status'] == 'OK'
    path = tmp_path/f'reflectancia_{day}.parquet'
    assert extraction.is_current_extraction(path, 'grid-hash', 30)
    assert not extraction.is_current_extraction(path, 'other-grid', 30)
    path.write_bytes(b'broken-parquet')
    assert not extraction.is_current_extraction(path, 'grid-hash', 30)


@pytest.mark.parametrize('module,function', [
    ('scripts.fase_5b_00_audit', 'run_audit'),
    ('scripts.fase_5b_02_run_atlas', 'run_streaming_atlas'),
    ('scripts.fase_5b_03_aggregate', 'run_aggregation'),
    ('scripts.fase_5b_04_export_maps', 'run_export_and_mapping'),
    ('scripts.prepare_viewer', 'main'),
    ('scripts.fase_4_bdjpt', 'main'),
])
def test_legacy_entry_points_stop_before_reading_or_overwriting_artifacts(module, function):
    import importlib
    with pytest.raises(RuntimeError, match='definitive production blocked'):
        getattr(importlib.import_module(module), function)()
