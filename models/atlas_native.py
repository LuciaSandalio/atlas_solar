"""Native 2 km candidate inference. No release approval or annual extrapolation.

Hourly bins reproduce the historical training inputs: floor UTC, mean reflectance
and angles, THEN B0/cosZ, THEN shared option C. Sampling completeness is a separate
mask: an absent half-hour cannot be hidden by a mean of the remaining scene.
"""
import calendar
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pvlib

from models.atlas_contract import align_option_c, FEATURES, predict_regional, season

ENERGY_COLUMNS = ('ghi', 'dni', 'dhi', 'poa_fixed', 'poa_tracker', 'pac_fixed', 'pac_tracker')
PV_SPECS = dict(pdc0_w=1000., gamma_pdc=-0.0035, noct_c=45., eta_inv_nom=0.96,
                bos_factor=0.90, tilt=30., azimuth=0., tracker_max_angle=60., tracker_gcr=0.35)
AMBIENT = {'Verano': 24., 'Otono': 18., 'Invierno': 11., 'Primavera': 19.}


def native_features(raw, coefficients, grid):
    required = ['reflectance', 'zenith_deg', 'azimuth_deg']
    data = raw[['cell_id', 'timestamp', 'lat', 'lon', *required]].copy()
    times = pd.to_datetime(data.timestamp)
    if times.dt.tz is None:
        raise ValueError('Native scenes must have timezone-aware UTC timestamps.')
    data['timestamp'] = times.dt.tz_convert('UTC')
    if data.duplicated(['cell_id', 'timestamp']).any():
        raise ValueError('Duplicate native scene/cell.')
    if data.timestamp.dt.normalize().nunique() != 1:
        raise ValueError('Exactly one UTC day per native input.')
    coordinates = grid.set_index('cell_id')[['lat', 'lon']].reindex(data.cell_id)
    if not np.allclose(data[['lat', 'lon']].to_numpy(), coordinates.to_numpy(), atol=5e-6, rtol=0):
        raise ValueError('Native cell IDs/coordinates do not match the canonical grid.')
    finite = np.isfinite(data[required]).all(axis=1)
    finite &= data.reflectance.between(0, 1.2 + 1e-6)
    finite &= data.zenith_deg.between(0, 180) & data.azimuth_deg.between(0, 360)
    data.loc[~finite, required] = np.nan
    data['hour'] = data.timestamp.dt.floor('h')
    slot = data.timestamp.dt.round('30min')
    data['slot'] = slot
    data['slot_valid'] = ((data.timestamp-slot).abs() <= pd.Timedelta(minutes=7)) & (slot.dt.floor('h') == data.hour)
    # Exact hourly values are retained for parity checks even for incomplete bins.
    group = data.groupby(['cell_id', 'hour'])
    hourly = group[required].mean().where(group[required].count().eq(group.size(), axis=0))
    quality = group.agg(n_scenes=('timestamp', 'size'), n_slots=('slot', 'nunique'),
                        slots_valid=('slot_valid', 'all'))
    quality['complete_bin'] = (quality.n_scenes == 2) & (quality.n_slots == 2) & quality.slots_valid
    hourly = hourly.join(quality).reset_index().rename(columns={'hour': 'timestamp'})
    # Match the float32 arithmetic of the stored native training features.
    zen = np.radians(hourly.zenith_deg.to_numpy())
    cg = np.cos(np.radians(hourly.azimuth_deg.to_numpy()))
    hourly['cosZ'] = np.cos(zen)
    hourly['B0'] = np.maximum(coefficients['A'] + coefficients['B']*np.cos(zen)
                              + coefficients['C']*np.sin(zen)*cg
                              + coefficients['D']*np.sin(zen)*cg**2, 0.01)
    aligned = align_option_c(hourly)
    ordered = hourly.sort_values(['cell_id', 'timestamp']).copy()
    next_complete = ordered.groupby('cell_id').complete_bin.shift(-1).fillna(False).astype(bool)
    ordered['complete_pair'] = ordered.complete_bin & next_complete
    aligned = aligned.merge(ordered[['cell_id', 'timestamp', 'complete_pair']],
                            on=['cell_id', 'timestamp'], validate='one_to_one')
    return hourly, aligned


class GridSolar:
    """Vectorized equivalent of pvlib Location(grid lat/lon).get_clearsky.

    Read the installed pvlib climatology files directly, rather than trusting
    caches with no grid identity. Daily Linke interpolation includes leap years.
    """
    def __init__(self, grid):
        self.grid = grid.reset_index(drop=True)
        self.lat = self.grid.lat.to_numpy(dtype=float)
        self.lon = self.grid.lon.to_numpy(dtype=float)
        lat_index = np.array([pvlib.tools._degrees_to_index(x, 'latitude') for x in self.lat])
        lon_index = np.array([pvlib.tools._degrees_to_index(x, 'longitude') for x in self.lon])
        folder = Path(pvlib.__file__).parent/'data'
        self.source_files = [folder/'Altitude.h5', folder/'LinkeTurbidities.h5']
        with h5py.File(self.source_files[0]) as source:
            raw_alt = source['Altitude'][:][lat_index, lon_index]
        self.altitude = np.where(raw_alt == 255, 0., raw_alt.astype(float)*28 - 450)
        with h5py.File(self.source_files[1]) as source:
            self.linke = source['LinkeTurbidity'][:][lat_index, lon_index].astype(float)/20
        self.pressure = pvlib.atmosphere.alt2pres(self.altitude)

    def at(self, timestamp):
        time = pd.Timestamp(timestamp)
        if time.tzinfo is None:
            raise ValueError('Solar timestamps must have a timezone.')
        time = time.tz_convert('UTC')
        times = pd.DatetimeIndex([time]*len(self.grid))
        sp = pvlib.solarposition.get_solarposition(times, self.lat, self.lon,
                                                  altitude=self.altitude, pressure=self.pressure,
                                                  temperature=12.)
        mdays = np.array([calendar.monthrange(time.year, m)[1] for m in range(1, 13)])
        knots = np.r_[-15.5, np.cumsum(mdays)-mdays/2, mdays.sum()+15.5]
        index = np.searchsorted(knots, time.dayofyear)-1
        fraction = (time.dayofyear-knots[index])/(knots[index+1]-knots[index])
        values = np.column_stack([self.linke[:, -1], self.linke, self.linke[:, 0]])
        lt = values[:, index]*(1-fraction) + values[:, index+1]*fraction
        apparent = sp.apparent_zenith.to_numpy()
        am = pvlib.atmosphere.get_absolute_airmass(
            pvlib.atmosphere.get_relative_airmass(apparent), self.pressure)
        extra = float(pvlib.irradiance.get_extra_radiation(time))
        cs = pvlib.clearsky.ineichen(apparent, am, lt, altitude=self.altitude, dni_extra=extra)
        return dict(zenith=sp.zenith.to_numpy(), apparent_zenith=apparent,
                    azimuth=sp.azimuth.to_numpy(), ghi_cs=np.asarray(cs['ghi']), dni_extra=extra)


def irradiance_and_pv(ghi, solar, timestamp):
    """W/m² irradiances and W AC power for 1 kWp; NaN propagates."""
    ghi = np.asarray(ghi, dtype=float)
    n = len(ghi)
    result = {name: np.full(n, np.nan) for name in ENERGY_COLUMNS}
    result['ghi'] = ghi.copy()
    valid = np.isfinite(ghi) & (ghi >= 0) & (solar['apparent_zenith'] < 87)
    if not valid.any():
        return result
    z = solar['zenith'][valid]
    app = solar['apparent_zenith'][valid]
    az = solar['azimuth'][valid]
    times = pd.DatetimeIndex([pd.Timestamp(timestamp)]*int(valid.sum()))
    erbs = pvlib.irradiance.erbs(ghi[valid], z, times)
    dni, dhi = np.asarray(erbs['dni']), np.asarray(erbs['dhi'])
    result['dni'][valid], result['dhi'][valid] = dni, dhi
    extra = solar['dni_extra']
    tracker = pvlib.tracking.singleaxis(app, az, axis_tilt=0., axis_azimuth=0.,
                                       max_angle=60., backtrack=True, gcr=.35)
    for name, tilt, surface_azimuth in [('fixed', 30., 0.),
                                       ('tracker', tracker['surface_tilt'], tracker['surface_azimuth'])]:
        poa = pvlib.irradiance.get_total_irradiance(tilt, surface_azimuth, app, az, dni,
                                                  ghi[valid], dhi, dni_extra=extra, model='perez')
        power = np.maximum(np.asarray(poa['poa_global']), 0.)
        temperature = AMBIENT[season(pd.Timestamp(timestamp).month)] + (45.-20.)/800.*power
        dc = pvlib.pvsystem.pvwatts_dc(power, temperature, 1000., -.0035)
        ac = pvlib.inverter.pvwatts(dc, 1000., eta_inv_nom=.96)*.90
        result['poa_'+name][valid] = power
        result['pac_'+name][valid] = np.maximum(ac, 0.)
    return result


def hourly_candidate(grid, models, aligned, solar_engine, date):
    """Yield 24 UTC intervals for both modes, including missing daylight hours.

    Geometry at target time reproduces training clear sky. Any sunlight at an
    interval's start/middle/end marks it expected, independently of scene presence.
    Low sun outside training/Erbs support remains missing, never physical night.
    """
    day = pd.Timestamp(date)
    day = day.tz_localize('UTC') if day.tzinfo is None else day.tz_convert('UTC')
    day = day.normalize()
    ids = grid.cell_id.to_numpy()
    weights = grid[['weight_R0', 'weight_R1', 'weight_R2']].to_numpy()
    start = solar_engine.at(day)
    by_time = {t: group.set_index('cell_id') for t, group in aligned.groupby('timestamp')}
    for h in range(24):
        time = day + pd.Timedelta(hours=h)
        middle = solar_engine.at(time+pd.Timedelta(minutes=30))
        end = solar_engine.at(time+pd.Timedelta(hours=1))
        expected = np.minimum.reduce([start['apparent_zenith'], middle['apparent_zenith'],
                                      end['apparent_zenith']]) < 90
        model_domain = (start['ghi_cs'] > 20) & (start['apparent_zenith'] < 87)
        part = by_time.get(time, pd.DataFrame(columns=[*FEATURES, 'complete_pair'])).reindex(ids)
        features = part[FEATURES].to_numpy(dtype=float)
        complete = part.complete_pair.fillna(False).astype(bool).to_numpy()
        eligible = model_domain & complete
        features[~eligible] = np.nan
        for mode in ('blended', 'hard'):
            kt = predict_regional(models, season(time.month), features, start['ghi_cs'], weights, mode)
            values = irradiance_and_pv(kt*start['ghi_cs'], start, time)
            valid = expected & np.logical_and.reduce([np.isfinite(x) for x in values.values()])
            # Only wholly dark intervals contribute a physical zero.
            for v in values.values():
                v[~expected] = 0.
                v[expected & ~valid] = np.nan
            frame = pd.DataFrame({'cell_id': ids, 'timestamp': time, 'expected_daylight': expected,
                                  'valid_daylight': valid, 'complete_scene_pair': complete,
                                  'within_model_domain': model_domain, 'ghi_cs': start['ghi_cs'],
                                  'zenith_deg': start['zenith'], **values})
            yield mode, frame
        start = end


class DailyEnergy:
    """Streaming hourly integration with explicit observed and complete totals."""
    def __init__(self, ids):
        self.ids = np.asarray(ids)
        self.expected = np.zeros(len(ids), dtype=np.int32)
        self.valid = np.zeros(len(ids), dtype=np.int32)
        self.sums = np.zeros((len(ids), len(ENERGY_COLUMNS)), dtype=float)
        self.timestamps = set()

    def add(self, frame):
        if not np.array_equal(frame.cell_id.to_numpy(), self.ids):
            raise ValueError('Accumulator cell order mismatch.')
        time = pd.Timestamp(frame.timestamp.iloc[0])
        if frame.timestamp.nunique() != 1 or time in self.timestamps:
            raise ValueError('Duplicate or mixed hourly interval.')
        if time.minute or time.second or time.microsecond or time.nanosecond:
            raise ValueError('Integration requires exact hourly interval labels.')
        if self.timestamps and time.normalize() != next(iter(self.timestamps)).normalize():
            raise ValueError('Cannot combine different UTC days in one daily accumulator.')
        expected, valid = frame.expected_daylight.to_numpy(), frame.valid_daylight.to_numpy()
        values = frame[list(ENERGY_COLUMNS)].to_numpy()
        if (valid & ~expected).any() or not np.isfinite(values[valid]).all() or (values[valid] < 0).any():
            raise ValueError('Invalid daylight validity flags or energy values.')
        if not np.all(values[~expected] == 0):
            raise ValueError('Physical night must be zero.')
        if not np.isnan(values[expected & ~valid]).all():
            raise ValueError('Missing daylight must remain NaN.')
        self.timestamps.add(time)
        self.expected += expected
        self.valid += valid
        self.sums += np.where(valid[:, None], values, 0.)/1000.  # W × 1 h -> kWh

    def finish(self):
        if len(self.timestamps) != 24:
            raise ValueError('A daily product needs all 24 expected UTC intervals, including absent scenes.')
        complete = self.valid == self.expected
        out = pd.DataFrame({'cell_id': self.ids, 'n_expected_hours': self.expected,
                            'n_valid_hours': self.valid, 'missing_daylight_hours': self.expected-self.valid,
                            'coverage_pct': np.divide(100.*self.valid, self.expected,
                                                      out=np.full(len(self.ids), 100.), where=self.expected > 0),
                            'complete_day': complete})
        for i, name in enumerate(ENERGY_COLUMNS):
            unit = 'kwh_kwp' if name.startswith('pac') else 'kwh_m2'
            output_name = name.replace('pac', 'yf')+'_'+unit
            out['observed_'+output_name] = self.sums[:, i]
            out[output_name] = np.where(complete, self.sums[:, i], np.nan)
        return out
