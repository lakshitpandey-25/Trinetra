"""
TRINETRA — AI Powered Geospatial Disaster Intelligence
Module: Data Ingestion from Satellite & Sensor Sources
Covers: Sentinel-2, MODIS VIIRS, GPM IMERG, SRTM DEM, SMAP, Landsat 8/9
"""

import ee
import geemap
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from datetime import datetime, timedelta
import requests
import json
import os
from pathlib import Path

# ── Initialize Google Earth Engine ────────────────────────────────────────────
ee.Initialize()

# ── Uttarakhand bounding box ───────────────────────────────────────────────────
UTTARAKHAND_BBOX = {
    'lon_min': 77.5, 'lon_max': 81.1,
    'lat_min': 28.7, 'lat_max': 31.5
}

UTTARAKHAND_CRS = 'EPSG:32644'   # WGS84 / UTM zone 44N
OUTPUT_SCALE    = 10             # metres (Sentinel-2 native)


def get_study_region() -> ee.Geometry:
    """Return Uttarakhand study area as a GEE geometry."""
    return ee.Geometry.Rectangle([
        UTTARAKHAND_BBOX['lon_min'], UTTARAKHAND_BBOX['lat_min'],
        UTTARAKHAND_BBOX['lon_max'], UTTARAKHAND_BBOX['lat_max']
    ])


# ── Sentinel-2 ────────────────────────────────────────────────────────────────
class Sentinel2Ingester:
    """
    Ingest Sentinel-2 Level-2A (Surface Reflectance) multispectral imagery.
    Bands used:
      B2  Blue  (490 nm)
      B3  Green (560 nm)
      B4  Red   (665 nm)
      B5  Red-edge 1 (705 nm)
      B6  Red-edge 2 (740 nm)
      B7  Red-edge 3 (783 nm)
      B8  NIR broad  (842 nm)
      B8A NIR narrow (865 nm)
      B11 SWIR-1 (1610 nm)
      B12 SWIR-2 (2190 nm)
      SCL Scene Classification Layer (cloud masking)
    """

    BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7',
             'B8', 'B8A', 'B11', 'B12', 'SCL']

    def __init__(self, start_date: str, end_date: str, cloud_cover: float = 20):
        self.start_date  = start_date
        self.end_date    = end_date
        self.cloud_cover = cloud_cover
        self.region      = get_study_region()

    def fetch_collection(self) -> ee.ImageCollection:
        """Fetch and filter Sentinel-2 collection by date, bounds, cloud cover."""
        collection = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(self.region)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', self.cloud_cover))
            .select(self.BANDS)
        )
        count = collection.size().getInfo()
        print(f"[S2] Images found: {count} (cloud < {self.cloud_cover}%)")
        return collection

    def apply_cloud_mask(self, image: ee.Image) -> ee.Image:
        """
        Apply Scene Classification Layer (SCL) cloud masking.
        Mask classes: 3=Cloud shadow, 8=Cloud medium, 9=Cloud high, 10=Cirrus
        """
        scl   = image.select('SCL')
        clear = (scl.neq(3).And(scl.neq(8))
                           .And(scl.neq(9))
                           .And(scl.neq(10)))
        return image.updateMask(clear).divide(10000)  # scale to [0,1]

    def get_median_composite(self) -> ee.Image:
        """Create cloud-free median mosaic over the period."""
        collection = self.fetch_collection()
        masked     = collection.map(self.apply_cloud_mask)
        composite  = masked.median().clip(self.region)
        print("[S2] Median composite created.")
        return composite

    def get_temporal_stack(self, freq: str = '16D') -> list:
        """
        Build list of bi-monthly composites for change detection.
        freq: '16D' matches Landsat revisit for consistency.
        """
        collection = self.fetch_collection()
        masked     = collection.map(self.apply_cloud_mask)
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end   = datetime.strptime(self.end_date,   '%Y-%m-%d')
        composites = []
        cur = start
        while cur < end:
            nxt = min(cur + timedelta(days=16), end)
            period = (masked
                      .filterDate(cur.strftime('%Y-%m-%d'), nxt.strftime('%Y-%m-%d'))
                      .median()
                      .clip(self.region))
            composites.append((cur.strftime('%Y-%m-%d'), period))
            cur = nxt
        print(f"[S2] Temporal stack: {len(composites)} composites")
        return composites

    def export_to_drive(self, image: ee.Image, description: str, scale: int = 10):
        """Export GeoTIFF to Google Drive (Google Earth Engine batch export)."""
        task = ee.batch.Export.image.toDrive(
            image       = image.toFloat(),
            description = description,
            folder      = 'TRINETRA_Data',
            region      = self.region,
            scale       = scale,
            crs         = UTTARAKHAND_CRS,
            maxPixels   = 1e10,
            fileFormat  = 'GeoTIFF'
        )
        task.start()
        print(f"[S2] Export started → Drive/{description}")
        return task

    def export_to_asset(self, image: ee.Image, asset_id: str):
        """Export directly to GEE asset for downstream processing."""
        task = ee.batch.Export.image.toAsset(
            image       = image.toFloat(),
            description = f"asset_{asset_id}",
            assetId     = f"projects/trinetra/{asset_id}",
            region      = self.region,
            scale       = 10,
            maxPixels   = 1e10
        )
        task.start()
        print(f"[S2] Asset export started → {asset_id}")
        return task


# ── Sentinel-1 SAR ────────────────────────────────────────────────────────────
class Sentinel1SARIngester:
    """
    Ingest Sentinel-1 C-band SAR for flood inundation mapping.
    Uses VV + VH polarisation in Interferometric Wide swath mode.
    SAR penetrates clouds — critical during monsoon season.
    """

    def fetch_sar_collection(self, start_date: str, end_date: str) -> ee.ImageCollection:
        collection = (
            ee.ImageCollection('COPERNICUS/S1_GRD')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .filter(ee.Filter.eq('instrumentMode', 'IW'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
            .select(['VV', 'VH'])
        )
        print(f"[S1-SAR] Images found: {collection.size().getInfo()}")
        return collection

    def compute_flood_mask(self, pre_image: ee.Image, post_image: ee.Image,
                           threshold_db: float = -15.0) -> ee.Image:
        """
        Detect inundation by comparing pre/post SAR backscatter.
        Low VV backscatter (<= threshold_db) = open water / flooding.
        """
        change = post_image.select('VV').subtract(pre_image.select('VV'))
        flood  = post_image.select('VV').lte(threshold_db).And(change.lte(-3))
        return flood.rename('flood_mask')


# ── MODIS/VIIRS Active Fire ────────────────────────────────────────────────────
class MODISFireIngester:
    """
    MODIS Terra/Aqua + VIIRS S-NPP active fire product ingestion.
    Provides daily fire pixel detections with confidence scores.
    """

    def fetch_active_fires(self, start_date: str, end_date: str,
                           min_confidence: int = 70) -> pd.DataFrame:
        """Fetch MODIS active fire pixels with confidence >= min_confidence."""
        collection = (
            ee.ImageCollection('FIRMS')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
        )
        # Extract as table via GEE getRegion
        raw = geemap.ee_to_pandas(
            collection.getRegion(get_study_region(), 1000)
        )
        fire_df = raw[raw['confidence'] >= min_confidence].copy()
        fire_df['date'] = pd.to_datetime(fire_df['time'], unit='ms')
        print(f"[MODIS-Fire] Active fire pixels: {len(fire_df)} "
              f"(confidence >= {min_confidence}%)")
        return fire_df

    def fetch_burned_area(self, year: int) -> ee.Image:
        """MODIS MCD64A1 monthly burned area product."""
        ba = (
            ee.ImageCollection('MODIS/006/MCD64A1')
            .filterBounds(get_study_region())
            .filter(ee.Filter.calendarRange(year, year, 'year'))
            .select('BurnDate')
            .max()
            .clip(get_study_region())
        )
        print(f"[MODIS-BA] Burned area composite for {year}")
        return ba

    def fetch_lst(self, start_date: str, end_date: str) -> ee.Image:
        """MODIS MOD11A2 8-day Land Surface Temperature."""
        lst = (
            ee.ImageCollection('MODIS/006/MOD11A2')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .select('LST_Day_1km')
            .mean()
            .multiply(0.02)           # scale factor → Kelvin
            .subtract(273.15)         # Kelvin → Celsius
            .clip(get_study_region())
        )
        print("[MODIS-LST] Land Surface Temperature computed.")
        return lst


# ── GPM IMERG Rainfall ────────────────────────────────────────────────────────
class GPMRainfallIngester:
    """
    GPM IMERG Final Run V06 precipitation ingestion.
    Spatial resolution: 0.1° (~11km)
    Temporal resolution: 30-minute → aggregated to daily/72hr
    """

    def fetch_rainfall(self, start_date: str, end_date: str) -> ee.Image:
        """Cumulative rainfall over the period (mm)."""
        gpm = (
            ee.ImageCollection('NASA/GPM_L3/IMERG_V06')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .select('precipitationCal')
        )
        cumulative = gpm.sum().multiply(0.5).clip(get_study_region())
        # .multiply(0.5): mm/hr * 0.5hr = mm per 30-min step
        print(f"[GPM] Cumulative rainfall computed: {start_date} → {end_date}")
        return cumulative

    def fetch_daily_series(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Extract daily rainfall timeseries for spatial analysis."""
        gpm = (
            ee.ImageCollection('NASA/GPM_L3/IMERG_V06')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .select('precipitationCal')
        )
        # Reduce to daily totals
        def daily_sum(img):
            date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
            val  = img.reduceRegion(
                reducer  = ee.Reducer.mean(),
                geometry = get_study_region(),
                scale    = 11000,
                maxPixels= 1e8
            )
            return ee.Feature(None, val.set('date', date))

        fc = gpm.map(daily_sum)
        df = geemap.ee_to_pandas(fc)
        return df

    def compute_extreme_events(self, start_date: str, end_date: str,
                               threshold_mm: float = 100) -> ee.Image:
        """
        Map pixels where daily rainfall exceeded extreme threshold (100mm/day).
        IMD classification: Very heavy rain > 64.5mm, Extremely heavy > 204.4mm/day
        """
        gpm = (
            ee.ImageCollection('NASA/GPM_L3/IMERG_V06')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .select('precipitationCal')
        )
        extreme_count = gpm.map(
            lambda img: img.multiply(0.5).gte(threshold_mm / 48)
        ).sum().clip(get_study_region())
        return extreme_count


# ── SRTM DEM / Terrain ────────────────────────────────────────────────────────
class SRTMTerrainIngester:
    """
    SRTM Digital Elevation Model (30m resolution).
    Derives: elevation, slope, aspect, hillshade.
    """

    def fetch_terrain(self) -> ee.Image:
        """Fetch terrain products: elevation, slope (degrees), aspect (degrees)."""
        srtm    = ee.Image('USGS/SRTMGL1_003')
        terrain = ee.Terrain.products(srtm)
        result  = terrain.select(['elevation', 'slope', 'aspect']).clip(get_study_region())
        print("[SRTM] Terrain products fetched (elevation, slope, aspect).")
        return result

    def fetch_flow_accumulation(self) -> ee.Image:
        """HydroSHEDS flow accumulation for TWI computation."""
        flow = (
            ee.Image('WWF/HydroSHEDS/15ACC')
            .clip(get_study_region())
        )
        print("[HydroSHEDS] Flow accumulation fetched.")
        return flow

    def fetch_geology(self) -> ee.Image:
        """
        GLiM Lithological map as static landslide susceptibility input.
        Returns categorical lithology class image.
        """
        # Using USGS dataset as proxy (replace with GSI data for production)
        glim = ee.Image('CSP/ERGo/1_0/Global/SRTM_landforms').clip(get_study_region())
        return glim


# ── NASA SMAP Soil Moisture ───────────────────────────────────────────────────
class SMAPSoilMoistureIngester:
    """
    NASA SMAP (Soil Moisture Active Passive) L4 10km soil moisture.
    Surface soil moisture (ssm) in m³/m³.
    """

    def fetch_soil_moisture(self, start_date: str, end_date: str) -> ee.Image:
        """Mean surface soil moisture over the period."""
        smap = (
            ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .select('ssm')
        )
        mean_ssm = smap.mean().clip(get_study_region())
        print("[SMAP] Soil moisture composite computed.")
        return mean_ssm

    def compute_ssm_anomaly(self, start_date: str, end_date: str,
                            baseline_years: int = 5) -> ee.Image:
        """
        Compute soil moisture anomaly vs multi-year baseline.
        Positive anomaly = wetter than normal = elevated flood/landslide risk.
        """
        current = self.fetch_soil_moisture(start_date, end_date)
        # Baseline: same calendar period over previous N years
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        baselines = []
        for y in range(1, baseline_years + 1):
            bs = self.fetch_soil_moisture(
                (start_dt - timedelta(days=365*y)).strftime('%Y-%m-%d'),
                (start_dt - timedelta(days=365*y - 30)).strftime('%Y-%m-%d')
            )
            baselines.append(bs)
        baseline_img = ee.ImageCollection(baselines).mean()
        anomaly = current.subtract(baseline_img).rename('ssm_anomaly')
        return anomaly


# ── Landsat 8/9 ───────────────────────────────────────────────────────────────
class LandsatIngester:
    """
    Landsat 8/9 Collection 2 Surface Reflectance ingestion.
    16-day revisit, 30m resolution — useful for long time-series NDVI change.
    """

    def fetch_collection(self, start_date: str, end_date: str,
                         cloud_cover: float = 20) -> ee.ImageCollection:
        l8 = (
            ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover))
            .select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
                     'ST_B10','QA_PIXEL'])
        )
        l9 = (
            ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt('CLOUD_COVER', cloud_cover))
            .select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',
                     'ST_B10','QA_PIXEL'])
        )
        merged = l8.merge(l9).sort('system:time_start')
        print(f"[Landsat 8/9] Images found: {merged.size().getInfo()}")
        return merged

    def apply_scale_factors(self, image: ee.Image) -> ee.Image:
        """Apply Collection 2 scale factors and offset."""
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        thermal = image.select('ST_B10').multiply(0.00341802).add(149.0)
        return image.addBands(optical, None, True).addBands(thermal, None, True)


# ── ERA5 Wind Data ────────────────────────────────────────────────────────────
class ERA5WindIngester:
    """
    ERA5 reanalysis wind speed and direction for fire propagation modeling.
    """

    def fetch_wind(self, start_date: str, end_date: str) -> ee.Image:
        era5 = (
            ee.ImageCollection('ECMWF/ERA5/DAILY')
            .filterBounds(get_study_region())
            .filterDate(start_date, end_date)
            .select(['u_component_of_wind_10m', 'v_component_of_wind_10m'])
        )
        mean_wind = era5.mean().clip(get_study_region())
        wind_speed = mean_wind.expression(
            'sqrt(u*u + v*v)',
            {'u': mean_wind.select('u_component_of_wind_10m'),
             'v': mean_wind.select('v_component_of_wind_10m')}
        ).rename('wind_speed_ms')
        wind_dir = mean_wind.expression(
            '(180 / 3.14159) * atan2(-u, -v)',
            {'u': mean_wind.select('u_component_of_wind_10m'),
             'v': mean_wind.select('v_component_of_wind_10m')}
        ).rename('wind_direction_deg')
        print("[ERA5] Wind speed and direction computed.")
        return wind_speed.addBands(wind_dir)


# ── IMD Ground Station Parser ─────────────────────────────────────────────────
class IMDGroundStationParser:
    """
    Parse India Meteorological Department CSV station data.
    Provides ground-truth for model validation.
    """

    def parse_stations(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        required = ['station_id', 'lat', 'lon', 'date', 'max_temp',
                    'min_temp', 'rainfall_mm', 'humidity_pct']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[IMD] Warning — missing columns: {missing}")
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['lat', 'lon', 'date'])
        print(f"[IMD] Parsed {len(df)} station records from {df['date'].min()} "
              f"to {df['date'].max()}")
        return df


# ── Master Ingestion Pipeline ─────────────────────────────────────────────────
def run_ingestion_pipeline(start_date: str, end_date: str,
                           output_dir: str = './data/raw'):
    """
    Orchestrate complete data ingestion:
    1. Sentinel-2 optical composite
    2. Sentinel-1 SAR (for floods during cloud cover)
    3. MODIS active fire pixels + burned area
    4. MODIS Land Surface Temperature
    5. GPM 72-hr cumulative rainfall
    6. SRTM terrain products
    7. HydroSHEDS flow accumulation
    8. SMAP soil moisture + anomaly
    9. ERA5 wind
    10. Export all to GDrive / GEE assets
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  TRINETRA Data Ingestion Pipeline")
    print(f"  Period: {start_date} → {end_date}")
    print(f"{'='*60}\n")

    # 1. Sentinel-2
    s2 = Sentinel2Ingester(start_date, end_date, cloud_cover=15)
    composite = s2.get_median_composite()
    s2.export_to_drive(composite, f"S2_median_{start_date}_{end_date}")

    # 2. MODIS fire
    modis = MODISFireIngester()
    fires  = modis.fetch_active_fires(start_date, end_date)
    fires.to_csv(f"{output_dir}/active_fires.csv", index=False)
    lst    = modis.fetch_lst(start_date, end_date)

    # 3. GPM rainfall
    gpm_data = GPMRainfallIngester().fetch_rainfall(start_date, end_date)

    # 4. SRTM terrain
    terrain     = SRTMTerrainIngester().fetch_terrain()
    flow_accum  = SRTMTerrainIngester().fetch_flow_accumulation()

    # 5. SMAP soil moisture
    smap_ingest = SMAPSoilMoistureIngester()
    soil_moist  = smap_ingest.fetch_soil_moisture(start_date, end_date)
    ssm_anomaly = smap_ingest.compute_ssm_anomaly(start_date, end_date)

    # 6. Wind
    wind = ERA5WindIngester().fetch_wind(start_date, end_date)

    # 7. Stack all for export
    all_bands = (composite
                 .addBands(gpm_data.rename('rainfall_72h'))
                 .addBands(terrain)
                 .addBands(flow_accum.rename('flow_accum'))
                 .addBands(soil_moist.rename('soil_moisture'))
                 .addBands(ssm_anomaly)
                 .addBands(lst.rename('lst_celsius'))
                 .addBands(wind))

    s2.export_to_drive(all_bands, f"TRINETRA_stack_{start_date}_{end_date}", scale=30)
    print(f"\n✓ Ingestion pipeline complete. Check GDrive/TRINETRA_Data/")
    return all_bands


if __name__ == '__main__':
    run_ingestion_pipeline('2024-05-01', '2024-06-14', './data/raw')
