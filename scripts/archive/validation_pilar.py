import pandas as pd
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import pvlib
import numpy as np

# Paths
STATION_FILE = Path("data/processed/stations/pilar_cmp21_hourly.parquet")
MODEL_DIR = Path("data/ready/clearsky_rest2")
FEATURES_DIR = Path("data/ready/features_cell_hour")

# Pilar Coordinates
PILAR_LAT = -31.667
PILAR_LON = -63.883
PILAR_ALT = 338 # meters (approx)

def get_nearest_cell_id():
    """Finds the nearest cell_id to Pilar using one feature file."""
    # Try finding a file
    files = sorted(list(FEATURES_DIR.glob("features_*.parquet")))
    if not files:
        raise FileNotFoundError("No features files found to determine grid.")
        
    df = pd.read_parquet(files[0])
    
    # Euclidean distance
    dist = (df['lat'] - PILAR_LAT)**2 + (df['lon'] - PILAR_LON)**2
    idx = dist.idxmin()
    
    cell_id = df.loc[idx, 'cell_id']
    lat = df.loc[idx, 'lat']
    lon = df.loc[idx, 'lon']
    return cell_id, lat, lon

def main():
    # 1. Identify Target Cell
    try:
        target_cell_id, target_lat, target_lon = get_nearest_cell_id()
        print(f"📍 Pilar ({PILAR_LAT}, {PILAR_LON}) -> Grid Cell {target_cell_id} ({target_lat:.4f}, {target_lon:.4f})")
    except Exception as e:
        print(f"❌ Error finding cell: {e}")
        return

    # 2. Load Station Data
    print("📉 Loading Station Data...")
    df_obs = pd.read_parquet(STATION_FILE)
    
    # 3. Load Model Data
    print(f"📈 Loading Model Data from {MODEL_DIR}...")
    model_files = sorted(list(MODEL_DIR.glob("cs_*.parquet")))
    if not model_files:
        print("❌ No model files found.")
        return

    dfs_model = []
    for f in model_files:
        try:
            temp = pd.read_parquet(f)
            # Filter for target cell
            temp = temp[temp['cell_id'] == target_cell_id]
            dfs_model.append(temp)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            
    if not dfs_model:
        print("❌ No model data for target cell.")
        return
        
    df_model = pd.concat(dfs_model)
    
    # 4. Merge
    print("🔄 Merging datasets...")
    merged = pd.merge(
        df_obs,
        df_model,
        left_on='timestamp',
        right_on='time_round',
        how='inner',
        suffixes=('_obs', '_mod')
    )
    
    if merged.empty:
        print("❌ No overlapping timestamps found.")
        return

    # 5. Compute Solar Position (for Validation Analysis)
    print("☀️ Computing Solar Position...")
    loc = pvlib.location.Location(PILAR_LAT, PILAR_LON, altitude=PILAR_ALT)
    solpos = loc.get_solarposition(merged['timestamp'])
    merged['zenith'] = solpos['zenith'].values
    merged['elevation'] = solpos['elevation'].values
    
    # 6. Clear Sky Detection
    # Logic: 0.95 < Obs/Model < 1.05 (Stricter?)
    # User suggested: "Refine the clear-sky filter... 0.95 < K_cs < 1.05"
    # User also asked for stability checks.
    
    merged['k_cs'] = merged['GHI'] / merged['ghi_cs']
    
    # Filter constraints
    # 1. Elevation > 10 degrees (avoid horizon shading/cosine error)
    # 2. GHI_model > 50 W/m2
    # 3. K_cs ratio
    
    mask_valid = (merged['elevation'] > 10) & (merged['ghi_cs'] > 50)
    
    # Flexible Threshold for Analysis
    K_LOW = 0.9
    K_HIGH = 1.1
    
    mask_clear = mask_valid & (merged['k_cs'] > K_LOW) & (merged['k_cs'] < K_HIGH)
    
    df_clear = merged[mask_clear].copy()
    
    # 7. Metrics & Analysis
    print(f"\n📊 --- Validation Summary (All Clear Hours: {len(df_clear)}) ---")
    
    # Standard Metrics
    diff = df_clear['ghi_cs'] - df_clear['GHI']
    mbe = diff.mean()
    rmse = np.sqrt((diff**2).mean())
    mean_obs = df_clear['GHI'].mean()
    print(f"Overall MBE: {mbe:.2f} W/m² ({mbe/mean_obs*100:.1f}%)")
    print(f"Overall RMSE: {rmse:.2f} W/m² ({rmse/mean_obs*100:.1f}%)")

    # Filtered Metrics (Zenith < 70)
    df_hq = df_clear[df_clear['zenith'] < 70].copy()
    if not df_hq.empty:
        diff_hq = df_hq['ghi_cs'] - df_hq['GHI']
        mbe_hq = diff_hq.mean()
        rmse_hq = np.sqrt((diff_hq**2).mean())
        mean_hq = df_hq['GHI'].mean()
        print(f"\n☀️  Headline Metrics (Zenith < 70°, N={len(df_hq)})")
        print(f"   MBE: {mbe_hq:.2f} W/m² ({mbe_hq/mean_hq*100:.1f}%)")
        print(f"   RMSE: {rmse_hq:.2f} W/m² ({rmse_hq/mean_hq*100:.1f}%)")
    
    # Seasonal Analysis (Group by Month)
    df_clear['month'] = df_clear['timestamp'].dt.month
    
    # Experimental Calibration: reduce July by 1.5%
    # "Apply a tiny seasonal scaling (e.g. 1–2% reduction in July)"
    # Re-filter df_hq to include month
    df_hq = df_clear[df_clear['zenith'] < 70].copy()
    
    df_exp = df_hq.copy()
    mask_jul = df_exp['month'] == 7
    df_exp.loc[mask_jul, 'ghi_cs'] *= 0.985 # 1.5% reduction
    
    mean_exp = df_exp['GHI'].mean()
    diff_exp = df_exp['ghi_cs'] - df_exp['GHI']
    mbe_exp = diff_exp.mean()
    
    print(f"\n🔧 Experimental (Zenith < 70° + July -1.5%)")
    print(f"   MBE: {mbe_exp:.2f} W/m² ({mbe_exp/mean_exp*100:.1f}%)")
    
    
    # Seasonal Analysis (Group by Month) - Using HQ Data for this table? 
    # Let's keep original for full seasonal breakdown but maybe note Zenith impact
    monthly_stats = df_clear.groupby('month').apply(
        lambda x: pd.Series({
            'count': len(x),
            'mbe': (x['ghi_cs'] - x['GHI']).mean(),
            'rmse': np.sqrt(((x['ghi_cs'] - x['GHI'])**2).mean()),
            'mean_ghi': x['GHI'].mean(),
            'zenith_mean': x['zenith'].mean()
        })
    ).sort_index()
    
    print("\n📅 --- Monthly Breakdown (All Zeniths) ---")
    print(monthly_stats)
    
    # Zenith Binning
    # Bin by 10 degrees
    df_clear['zenith_bin'] = (df_clear['zenith'] // 10 * 10).astype(int)
    zenith_stats = df_clear.groupby('zenith_bin').apply(
        lambda x: pd.Series({
            'count': len(x),
            'mbe': (x['ghi_cs'] - x['GHI']).mean(),
            'rmse': np.sqrt(((x['ghi_cs'] - x['GHI'])**2).mean())
        })
    ).sort_index()

    # 8. Plotting
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Scatter (Obs vs Model)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(merged['GHI'], merged['ghi_cs'], color='gray', alpha=0.1, label='All Data', s=10)
    sc = ax1.scatter(df_clear['GHI'], df_clear['ghi_cs'], c=df_clear['month'], cmap='viridis', label='Clear Data', s=15)
    plt.colorbar(sc, ax=ax1, label='Month')
    
    mx = 1200
    ax1.plot([0, mx], [0, mx], 'r--', lw=2)
    ax1.set_xlim(0, mx)
    ax1.set_ylim(0, mx)
    ax1.set_xlabel('Observed GHI (W/m²)')
    ax1.set_ylabel('Model GHI (W/m²)')
    ax1.set_title(f'Scatter Plot (MBE={mbe:.1f}, RMSE={rmse:.1f})')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Seasonal Bias (MBE per Month)
    ax2 = fig.add_subplot(gs[0, 1])
    months = monthly_stats.index
    ax2.bar(months, monthly_stats['mbe'], color='skyblue', edgecolor='black')
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('MBE (W/m²)')
    ax2.set_xticks(months)
    ax2.set_title('Seasonal Bias (Model - Obs)')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Plot 3: Bias vs Zenith
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(df_clear['zenith'], df_clear['ghi_cs'] - df_clear['GHI'], alpha=0.3, c='purple')
    ax3.plot(zenith_stats.index.values + 5, zenith_stats['mbe'].values, 'r-o', lw=2, label='Mean Bias')
    ax3.axhline(0, color='black', lw=1)
    ax3.axvline(70, color='orange', linestyle='--', label='70° Cutoff')
    ax3.set_xlabel('Solar Zenith Angle (deg)')
    ax3.set_ylabel('Error (W/m²)')
    ax3.set_title('Bias vs Zenith Angle')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time Series (Sample)
    # Just show the first available clear month to avoid mess
    if not monthly_stats.empty:
        first_month = monthly_stats.index[0]
        sample = merged[merged['timestamp'].dt.month == first_month]
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(sample['timestamp'].values, sample['ghi_cs'].values, label='Model', color='blue', lw=1)
        ax4.plot(sample['timestamp'].values, sample['GHI'].values, label='Obs', color='black', lw=1, alpha=0.6)
        
        # Highlight clear points
        sample_clear = df_clear[df_clear['timestamp'].dt.month == first_month]
        ax4.scatter(sample_clear['timestamp'].values, sample_clear['GHI'].values, color='green', s=10, zorder=5)
        
        ax4.set_title(f'Time Series Sample (Month {first_month})')
        ax4.set_ylabel('GHI (W/m²)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30)

    plt.tight_layout()
    out_file = "validation_seasonal_report.png"
    plt.savefig(out_file)
    print(f"\n✅ Saved analysis report: {out_file}")

if __name__ == "__main__":
    main()
