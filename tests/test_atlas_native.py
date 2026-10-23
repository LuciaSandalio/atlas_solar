import numpy as np
import pandas as pd
import pytest
import pvlib

from models.atlas_native import (GridSolar, native_features, irradiance_and_pv,
                                  DailyEnergy, ENERGY_COLUMNS)


@pytest.fixture(scope='module')
def solar_grid():
    grid = pd.DataFrame({'cell_id': [41, 9, 300], 'lat': [-32., -31.667, -34.],
                          'lon': [-64., -63.883, -65.]})
    return GridSolar(grid)


@pytest.mark.parametrize('time', ['2020-01-16 16:00Z', '2020-02-29 17:00Z',
                                  '2021-04-16 16:00Z', '2021-07-16 16:00Z', '2021-10-16 16:00Z'])
def test_vector_clear_sky_matches_training_location(solar_grid, time):
    actual = solar_grid.at(pd.Timestamp(time))
    for i, cell in solar_grid.grid.iterrows():
        location = pvlib.location.Location(cell.lat, cell.lon)
        times = pd.DatetimeIndex([pd.Timestamp(time)])
        expected = location.get_clearsky(times, model='ineichen').ghi.iloc[0]
        assert actual['ghi_cs'][i] == pytest.approx(expected, rel=1e-11, abs=1e-8)


def test_native_preprocessing_uses_floor_before_b0_and_c():
    raw = pd.DataFrame({'cell_id': [42]*4, 'lat': [-32.]*4, 'lon': [-64.]*4,
                         'timestamp': pd.to_datetime(['2020-01-01 12:00:20Z', '2020-01-01 12:30:20Z',
                                                      '2020-01-01 13:00:20Z', '2020-01-01 13:30:20Z']),
                         'reflectance': np.array([.2, .4, .6, .8], dtype='float32'),
                         'zenith_deg': [40.]*4, 'azimuth_deg': [10.]*4})
    grid = raw[['cell_id', 'lat', 'lon']].drop_duplicates()
    hourly, aligned = native_features(raw, {'A': .1, 'B': 0., 'C': 0., 'D': 0.}, grid)
    assert hourly.reflectance.to_list() == pytest.approx([.3, .7])
    assert aligned.reflectance.iloc[0] == pytest.approx(.5)
    assert aligned.Fr.iloc[0] == pytest.approx(.24)
    assert aligned.complete_pair.iloc[0]
    _, incomplete = native_features(raw.iloc[[0, 2, 3]], {'A': .1, 'B': 0., 'C': 0., 'D': 0.}, grid)
    assert not incomplete.complete_pair.iloc[0]
    with pytest.raises(ValueError, match='coordinates'):
        native_features(raw.assign(cell_id=10), {'A': .1, 'B': 0., 'C': 0., 'D': 0.}, grid)


def test_erbs_closure_and_pv_units(solar_grid):
    time = pd.Timestamp('2020-01-16 16:00Z')
    solar = solar_grid.at(time)
    values = irradiance_and_pv(np.array([500., 800., np.nan]), solar, time)
    closure = values['dni']*np.cos(np.radians(solar['zenith']))+values['dhi']
    np.testing.assert_allclose(values['ghi'][:2], closure[:2], atol=1e-9)
    for name in ENERGY_COLUMNS:
        assert np.isfinite(values[name][:2]).all()
        assert (values[name][:2] >= 0).all()
        assert np.isnan(values[name][2])
    assert (values['pac_fixed'][:2] <= 1000*.96*.90).all()


def synthetic_hour(hour, missing=False):
    expected = 10 <= hour < 18
    valid = expected and not missing
    value = 500. if valid else np.nan if expected else 0.
    return pd.DataFrame({'cell_id': [99], 'timestamp': [pd.Timestamp('2020-01-01', tz='UTC')+pd.Timedelta(hours=hour)],
                         'expected_daylight': [expected], 'valid_daylight': [valid],
                         **{k: [value] for k in ENERGY_COLUMNS}})


def test_hourly_integration_recovers_kwh_and_tracks_missing_instead_of_zero():
    full, partial = DailyEnergy([99]), DailyEnergy([99])
    for h in range(24):
        full.add(synthetic_hour(h))
        partial.add(synthetic_hour(h, missing=h == 12))
    good, bad = full.finish().iloc[0], partial.finish().iloc[0]
    assert good.ghi_kwh_m2 == 4.
    assert good.yf_fixed_kwh_kwp == 4.
    assert good.coverage_pct == 100.
    assert bad.observed_ghi_kwh_m2 == 3.5
    assert np.isnan(bad.ghi_kwh_m2)
    assert bad.missing_daylight_hours == 1
    assert bad.coverage_pct == 87.5


def test_missing_entire_hour_and_double_counting_are_rejected():
    accumulator = DailyEnergy([99])
    accumulator.add(synthetic_hour(12))
    with pytest.raises(ValueError, match='Duplicate'):
        accumulator.add(synthetic_hour(12))
    with pytest.raises(ValueError, match='24'):
        accumulator.finish()


def test_missing_daylight_zero_is_rejected():
    frame = synthetic_hour(12, missing=True)
    frame[list(ENERGY_COLUMNS)] = 0.
    with pytest.raises(ValueError, match='remain NaN'):
        DailyEnergy([99]).add(frame)
