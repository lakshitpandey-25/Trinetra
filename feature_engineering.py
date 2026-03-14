"""
TRINETRA — Feature Engineering Module
Computes all geospatial indices and hazard predictors from raw raster data.

Features computed:
  Spectral : NDVI, NDWI, NBR, NDMI, BAI, EVI, SAVI
  Terrain  : Slope, Aspect, Curvature, TWI, TPI
  Rainfall : Cumulative, Antecedent Rainfall Index (API), Extreme events
  Derived  : Temporal difference features (ΔNDVI, ΔNBR, ΔNDWI)
  Tabular  : Feature dataframe for RF / XGBoost input
"""

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import uniform_filter, gaussian_filter, label
from scipy.signal import convolve2d
import warnings
warnings.filterwarnings('ignore')


# ── Spectral Index Calculator ─────────────────────────────────────────────────
class SpectralIndexCalculator:
    """
    Computes vegetation, water, fire, and moisture indices
    from Sentinel-2 multispectral bands.

    Band mapping (Sentinel-2):
      B2=Blue, B3=Green, B4=Red, B5=RE1, B6=RE2, B7=RE3,
      B8=NIR-broad, B8A=NIR-narrow, B11=SWIR1, B12=SWIR2
    """

    @staticmethod
    def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Vegetation Index.
        NDVI = (NIR - Red) / (NIR + Red)
        S2:  NIR=B8, Red=B4
        Range [-1, 1]:  >0.6 dense forest | 0.2–0.6 shrub/grass | <0.1 bare/burned
        Low NDVI + high temperature = elevated fire risk.
        """
        ndvi = (nir - red) / (nir + red + 1e-8)
        return np.clip(ndvi, -1.0, 1.0)

    @staticmethod
    def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Water Index (McFeeters 1996).
        NDWI = (Green - NIR) / (Green + NIR)
        S2:  Green=B3, NIR=B8
        Positive values indicate open water bodies.
        Used for: flood extent mapping, water body expansion detection.
        """
        ndwi = (green - nir) / (green + nir + 1e-8)
        return np.clip(ndwi, -1.0, 1.0)

    @staticmethod
    def compute_ndwi_gao(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """
        Gao NDWI — leaf water content (vegetation moisture).
        NDWI_gao = (NIR - SWIR1) / (NIR + SWIR1)
        S2:  NIR=B8, SWIR1=B11
        Dry vegetation = low NDWI_gao = higher fire susceptibility.
        """
        ndwi = (nir - swir1) / (nir + swir1 + 1e-8)
        return np.clip(ndwi, -1.0, 1.0)

    @staticmethod
    def compute_nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        Normalized Burn Ratio.
        NBR = (NIR - SWIR2) / (NIR + SWIR2)
        S2:  NIR=B8A, SWIR2=B12
        dNBR = pre_NBR - post_NBR
          dNBR >  0.1 = burned area
          dNBR > 0.44 = high-severity burn
        """
        nbr = (nir - swir2) / (nir + swir2 + 1e-8)
        return np.clip(nbr, -1.0, 1.0)

    @staticmethod
    def compute_dnbr(pre_nbr: np.ndarray, post_nbr: np.ndarray) -> np.ndarray:
        """Delta NBR (dNBR) for burn severity classification."""
        return pre_nbr - post_nbr

    @staticmethod
    def compute_nbr2(swir1: np.ndarray, swir2: np.ndarray) -> np.ndarray:
        """
        NBR2 — short-wave infrared band ratio.
        NBR2 = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
        Useful for detecting post-fire moisture recovery.
        """
        nbr2 = (swir1 - swir2) / (swir1 + swir2 + 1e-8)
        return np.clip(nbr2, -1.0, 1.0)

    @staticmethod
    def compute_evi(nir: np.ndarray, red: np.ndarray, blue: np.ndarray) -> np.ndarray:
        """
        Enhanced Vegetation Index.
        EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
        Less soil-noise than NDVI; better in dense canopy.
        """
        evi = 2.5 * (nir - red) / (nir + 6.0 * red - 7.5 * blue + 1.0 + 1e-8)
        return np.clip(evi, -1.0, 1.0)

    @staticmethod
    def compute_savi(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
        """
        Soil-Adjusted Vegetation Index (Huete 1988).
        SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
        L=0.5 — optimal for sparse semi-arid Himalayan vegetation.
        """
        savi = ((nir - red) / (nir + red + L + 1e-8)) * (1.0 + L)
        return np.clip(savi, -1.0, 1.0)

    @staticmethod
    def compute_bai(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
        """
        Burn Area Index.
        BAI = 1 / ((Red - 0.1)^2 + (NIR - 0.06)^2)
        High values indicate active/recent burns.
        """
        return 1.0 / ((red - 0.1)**2 + (nir - 0.06)**2 + 1e-8)

    @staticmethod
    def compute_ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """
        Normalized Difference Moisture Index.
        NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        S2:  NIR=B8, SWIR1=B11
        Low NDMI → fuel moisture stress → elevated fire danger.
        """
        ndmi = (nir - swir1) / (nir + swir1 + 1e-8)
        return np.clip(ndmi, -1.0, 1.0)

    @staticmethod
    def compute_vci(ndvi: np.ndarray,
                    ndvi_min: np.ndarray, ndvi_max: np.ndarray) -> np.ndarray:
        """
        Vegetation Condition Index.
        VCI = (NDVI - NDVI_min) / (NDVI_max - NDVI_min) * 100
        Low VCI = vegetation stress / drought = fire-prone conditions.
        """
        vci = (ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-8) * 100
        return np.clip(vci, 0, 100)

    @staticmethod
    def compute_bsi(blue: np.ndarray, red: np.ndarray,
                    nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
        """
        Bare Soil Index — identifies eroded/unstable slopes.
        BSI = ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))
        High BSI on steep slopes = landslide susceptibility.
        """
        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-8)
        return np.clip(bsi, -1.0, 1.0)


# ── Terrain Feature Calculator ────────────────────────────────────────────────
class TerrainFeatureCalculator:
    """
    Derives geomorphological features from Digital Elevation Model (DEM).
    Input DEM should be SRTM 30m or finer.
    """

    def __init__(self, resolution_m: float = 30.0):
        self.res = resolution_m

    def compute_slope(self, dem: np.ndarray) -> np.ndarray:
        """
        Slope in degrees using 3×3 central difference (Horn's method).
        dz/dx and dz/dy computed via np.gradient.
        """
        dy, dx = np.gradient(dem, self.res)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        return np.degrees(slope_rad)

    def compute_aspect(self, dem: np.ndarray) -> np.ndarray:
        """
        Aspect (degrees from North, 0–360, clockwise).
        North = 0°, East = 90°, South = 180°, West = 270°.
        South-facing slopes drier → elevated fire risk.
        """
        dy, dx = np.gradient(dem, self.res)
        aspect = np.degrees(np.arctan2(dx, -dy))
        return (aspect + 360) % 360

    def compute_curvature(self, dem: np.ndarray) -> np.ndarray:
        """
        Profile curvature (second derivative of elevation).
        Negative = concave = water flow convergence = runoff accumulation.
        """
        kernel = np.array([[0, 1, 0],
                           [1,-4, 1],
                           [0, 1, 0]], dtype=float) / (self.res**2)
        return convolve2d(dem, kernel, mode='same', boundary='symm')

    def compute_tpi(self, dem: np.ndarray, window: int = 15) -> np.ndarray:
        """
        Topographic Position Index.
        TPI = elevation - mean_elevation_in_window
        Positive = ridges/peaks; Negative = valleys/channels.
        """
        mean_dem = uniform_filter(dem, size=window)
        return dem - mean_dem

    def compute_roughness(self, dem: np.ndarray, window: int = 3) -> np.ndarray:
        """
        Terrain Roughness Index = max - min within window.
        High roughness on steep slopes → mass movement susceptibility.
        """
        from scipy.ndimage import maximum_filter, minimum_filter
        return maximum_filter(dem, window) - minimum_filter(dem, window)

    def compute_twi(self, slope_deg: np.ndarray,
                    flow_accum: np.ndarray) -> np.ndarray:
        """
        Topographic Wetness Index (Beven & Kirkby 1979).
        TWI = ln(flow_accumulation / tan(slope_rad))
        High TWI (>10) = prone to waterlogging, saturation.
        Critical predictor for both flood and landslide.
        """
        slope_rad = np.radians(np.clip(slope_deg, 0.1, 89.0))
        flow_safe = np.maximum(flow_accum, 1.0)
        twi = np.log(flow_safe / np.tan(slope_rad))
        return np.clip(twi, 0, 25)

    def compute_stream_power_index(self, slope_deg: np.ndarray,
                                   flow_accum: np.ndarray) -> np.ndarray:
        """
        Stream Power Index = flow_accum * tan(slope)
        High SPI = high erosive potential = landslide/gully initiation zones.
        """
        slope_rad = np.radians(np.clip(slope_deg, 0.1, 89.0))
        return flow_accum * np.tan(slope_rad)

    def compute_ls_factor(self, slope_deg: np.ndarray,
                           flow_accum: np.ndarray) -> np.ndarray:
        """
        LS factor (slope-length steepness) from RUSLE erosion model.
        Used as proxy for soil erosion / landslide vulnerability.
        """
        slope_rad = np.radians(np.clip(slope_deg, 0.0, 89.0))
        m = 0.5  # slope exponent
        L = (flow_accum * self.res / 22.13) ** m
        S = np.where(
            slope_deg >= 9,
            16.8 * np.sin(slope_rad) - 0.5,
            10.8 * np.sin(slope_rad) + 0.03
        )
        return L * S


# ── Rainfall Feature Calculator ───────────────────────────────────────────────
class RainfallFeatureCalculator:
    """
    Derives hydrological hazard predictors from GPM IMERG rainfall data.
    """

    @staticmethod
    def antecedent_rainfall_index(rainfall_series: np.ndarray,
                                  decay: float = 0.85) -> np.ndarray:
        """
        Antecedent Rainfall Index (API).
        API_t = k * API_{t-1} + R_t
          k = 0.85 (recession coefficient — soil drainage rate)
        High API → saturated soil → elevated landslide + flood risk.
        rainfall_series: 1-D daily totals (mm/day)
        """
        api = np.zeros_like(rainfall_series, dtype=np.float32)
        api[0] = rainfall_series[0]
        for t in range(1, len(rainfall_series)):
            api[t] = decay * api[t-1] + rainfall_series[t]
        return api

    @staticmethod
    def compute_cumulative(daily_rain: np.ndarray,
                           window_days: int) -> np.ndarray:
        """
        Rolling cumulative rainfall over `window_days`.
        Works on 3-D array (days, H, W).
        """
        if daily_rain.ndim == 3:
            return np.sum(daily_rain[-window_days:, :, :], axis=0)
        return np.convolve(daily_rain, np.ones(window_days), mode='full')[:len(daily_rain)]

    @staticmethod
    def extreme_rainfall_mask(daily_rain_2d: np.ndarray,
                              threshold_mm: float = 100.0) -> np.ndarray:
        """
        Binary mask: 1 where rainfall exceeded extreme threshold.
        IMD thresholds:
          Heavy          : 64.5 – 115.5 mm/day
          Very heavy     : 115.5 – 204.4 mm/day
          Extremely heavy: > 204.4 mm/day
        Default 100mm/day = landslide warning threshold (NDMA guideline).
        """
        extreme = (daily_rain_2d > threshold_mm).astype(np.float32)
        return gaussian_filter(extreme, sigma=2)  # spatial smoothing

    @staticmethod
    def rainfall_intensity_duration_frequency(
            rainfall_series: np.ndarray,
            duration_hr: int = 24) -> float:
        """
        Estimate return period of a rainfall event.
        Returns: approximate recurrence interval in years.
        """
        window = duration_hr
        rolling_max = np.max([
            np.sum(rainfall_series[i:i+window])
            for i in range(len(rainfall_series) - window)
        ])
        # Empirical Gumbel estimate (placeholder — calibrate with IMD data)
        mean_annual_max = np.mean(rainfall_series) * 365
        cv = 0.3
        T = 1 / (1 - np.exp(-1 * np.exp(
            -(rolling_max - mean_annual_max) / (mean_annual_max * cv * np.sqrt(6) / np.pi)
        )))
        return float(T)

    @staticmethod
    def compute_spi(monthly_rain: np.ndarray, timescale: int = 3) -> np.ndarray:
        """
        Standardized Precipitation Index (McKee 1993).
        SPI < -1.0 = moderate drought (fire risk).
        SPI > +1.0 = wet conditions (flood/landslide risk).
        timescale: months of aggregation (1, 3, 6, 12).
        """
        from scipy.stats import norm
        rolling = np.convolve(monthly_rain, np.ones(timescale) / timescale, mode='valid')
        mean = np.mean(rolling)
        std  = np.std(rolling) + 1e-8
        return (rolling - mean) / std


# ── Temporal Change Features ──────────────────────────────────────────────────
class TemporalChangeCalculator:
    """
    Computes temporal difference features to detect anomalous change
    (fire onset, flood inundation, landslide scars).
    """

    @staticmethod
    def ndvi_anomaly(ndvi_current: np.ndarray,
                     ndvi_baseline: np.ndarray) -> np.ndarray:
        """
        ΔNDVI = NDVI_current - NDVI_baseline
        Negative values indicate vegetation loss / drought stress / fire damage.
        """
        return ndvi_current - ndvi_baseline

    @staticmethod
    def dnbr_severity_class(dnbr: np.ndarray) -> np.ndarray:
        """
        Classify burn severity from dNBR (Key & Benson 2006).
        Returns integer class:
          0 = Unburned     (dNBR < 0.1)
          1 = Low severity (0.1 – 0.27)
          2 = Moderate-low (0.27 – 0.44)
          3 = Moderate-high(0.44 – 0.66)
          4 = High severity(> 0.66)
        """
        classes = np.zeros_like(dnbr, dtype=np.int8)
        classes[dnbr >= 0.1]  = 1
        classes[dnbr >= 0.27] = 2
        classes[dnbr >= 0.44] = 3
        classes[dnbr >= 0.66] = 4
        return classes

    @staticmethod
    def detect_sar_inundation(pre_vv: np.ndarray, post_vv: np.ndarray,
                               threshold_db: float = -3.0) -> np.ndarray:
        """
        SAR-based flood inundation detection.
        Water surfaces have much lower VV backscatter than land.
        change_dB = post_VV - pre_VV; large decrease = new inundation.
        """
        change = post_vv - pre_vv
        inundated = ((change <= threshold_db) & (post_vv <= -15.0)).astype(np.float32)
        # Morphological cleanup
        from scipy.ndimage import binary_closing, binary_opening
        inundated = binary_closing(binary_opening(inundated.astype(bool),
                                                   np.ones((3,3))),
                                    np.ones((5,5))).astype(np.float32)
        return inundated


# ── Feature Stack Builder ─────────────────────────────────────────────────────
class FeatureStackBuilder:
    """
    Assembles all features into a unified multi-band raster stack
    ready for CNN patch extraction and tabular model training.
    """

    FEATURE_NAMES = [
        # Spectral (8)
        'ndvi', 'ndwi', 'nbr', 'ndmi', 'bai', 'evi', 'bsi', 'vci',
        # Terrain (7)
        'elevation', 'slope_deg', 'aspect_deg', 'curvature', 'twi',
        'tpi', 'roughness',
        # Hydrological (4)
        'rainfall_72h', 'rainfall_extreme', 'api_7d', 'soil_moisture',
        # Change (3)
        'ndvi_diff_30d', 'nbr_diff_30d', 'ssm_anomaly',
        # Ancillary (3)
        'lst_celsius', 'wind_speed', 'flow_accumulation',
    ]

    def build_feature_stack(self, bands: dict) -> np.ndarray:
        """
        Build (H, W, N_features) feature stack.

        Expected `bands` keys:
          's2'          : (H, W, 12) Sentinel-2 reflectance (0-1 scaled)
          'dem'         : (H, W)     DEM in metres
          'slope'       : (H, W)     Slope in degrees
          'flow_accum'  : (H, W)     D8 flow accumulation
          'rainfall_72h': (H, W)     72-hr cumulative rainfall (mm)
          'soil_moisture': (H, W)    SMAP SSM (m³/m³)
          'ndvi_prev'   : (H, W)     NDVI 30 days ago
          'nbr_prev'    : (H, W)     NBR 30 days ago
          'ssm_anomaly' : (H, W)     SMAP SSM anomaly
          'lst'         : (H, W)     Land Surface Temp (°C)
          'wind_speed'  : (H, W)     Wind speed (m/s)
        """
        s   = SpectralIndexCalculator()
        t   = TerrainFeatureCalculator(resolution_m=30.0)
        r   = RainfallFeatureCalculator()

        b   = bands['s2']
        # Unpack bands  [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]
        #                 0   1   2   3   4   5   6   7    8    9
        blue  = b[..., 0]; green = b[..., 1]; red   = b[..., 2]
        nir   = b[..., 6]; nirn  = b[..., 7]; swir1 = b[..., 8]; swir2 = b[..., 9]

        ndvi  = s.compute_ndvi(nir, red)
        ndwi  = s.compute_ndwi(green, nir)
        nbr   = s.compute_nbr(nirn, swir2)
        ndmi  = s.compute_ndmi(nir, swir1)
        bai   = s.compute_bai(red, nir)
        evi   = s.compute_evi(nir, red, blue)
        bsi   = s.compute_bsi(blue, red, nir, swir1)

        # VCI requires NDVI climatology — use 0.1 / 0.9 percentile as proxy
        ndvi_min = np.percentile(ndvi, 5)
        ndvi_max = np.percentile(ndvi, 95)
        vci   = s.compute_vci(ndvi, ndvi_min, ndvi_max)

        dem   = bands['dem']
        slope = bands.get('slope', t.compute_slope(dem))
        curv  = t.compute_curvature(dem)
        tpi   = t.compute_tpi(dem)
        rough = t.compute_roughness(dem)
        aspect= t.compute_aspect(dem)
        fa    = bands.get('flow_accum', np.ones_like(dem))
        twi   = t.compute_twi(slope, fa)

        rain72     = bands['rainfall_72h']
        rain_ext   = r.extreme_rainfall_mask(rain72, threshold_mm=100.0)
        api7       = np.full_like(rain72, float(np.mean(rain72)) * 0.85)
        soil_moist = bands['soil_moisture']

        ndvi_diff  = ndvi - bands.get('ndvi_prev', ndvi)
        nbr_diff   = nbr  - bands.get('nbr_prev',  nbr)
        ssm_anom   = bands.get('ssm_anomaly', np.zeros_like(soil_moist))
        lst        = bands.get('lst', np.full_like(dem, 25.0))
        wind       = bands.get('wind_speed', np.zeros_like(dem))

        stack = np.stack([
            ndvi, ndwi, nbr, ndmi, bai, evi, bsi, vci,
            dem, slope, aspect, curv, twi, tpi, rough,
            rain72, rain_ext, api7, soil_moist,
            ndvi_diff, nbr_diff, ssm_anom,
            lst, wind, fa
        ], axis=-1).astype(np.float32)

        print(f"[FeatureStack] Shape: {stack.shape} | "
              f"Features: {stack.shape[-1]} | "
              f"NaN pixels: {np.isnan(stack).sum()}")
        return stack

    def normalize_stack(self, stack: np.ndarray,
                        fit: bool = True,
                        stats_path: str = './models/feature_stats.npz'
                        ) -> np.ndarray:
        """
        Per-feature min-max normalization to [0, 1].
        Saves / loads statistics for consistent train/inference scaling.
        """
        H, W, F = stack.shape
        flat = stack.reshape(-1, F)
        if fit:
            mins = np.nanmin(flat, axis=0)
            maxs = np.nanmax(flat, axis=0)
            np.savez(stats_path, mins=mins, maxs=maxs,
                     feature_names=self.FEATURE_NAMES)
            print(f"[Normalize] Stats saved → {stats_path}")
        else:
            data = np.load(stats_path)
            mins = data['mins']; maxs = data['maxs']

        norm = (flat - mins) / (maxs - mins + 1e-8)
        norm = np.clip(norm, 0, 1)
        return norm.reshape(H, W, F)


# ── Patch Extractor (for CNN) ─────────────────────────────────────────────────
class PatchExtractor:
    """
    Extracts (patch_size × patch_size) tiles from the feature stack
    for CNN training and inference.
    """

    def __init__(self, patch_size: int = 256, stride: int = 128,
                 overlap_ratio: float = 0.5):
        self.patch_size   = patch_size
        self.stride       = stride if stride else int(patch_size * (1 - overlap_ratio))

    def extract_patches(self, stack: np.ndarray,
                         labels: np.ndarray = None):
        """
        Extract image patches from (H, W, F) stack.
        Returns:
          X_patches: (N, patch_size, patch_size, F)
          y_patches: (N, patch_size, patch_size)  if labels provided
          coords:    list of (row_start, col_start) for geo-referencing
        """
        H, W, F = stack.shape
        P = self.patch_size
        S = self.stride
        X_patches, y_patches, coords = [], [], []

        for r in range(0, H - P + 1, S):
            for c in range(0, W - P + 1, S):
                patch = stack[r:r+P, c:c+P, :]
                if np.isnan(patch).mean() > 0.3:
                    continue   # skip > 30% NaN patches
                X_patches.append(patch)
                coords.append((r, c))
                if labels is not None:
                    y_patches.append(labels[r:r+P, c:c+P])

        X = np.array(X_patches, dtype=np.float32)
        print(f"[PatchExtractor] Extracted {len(X)} patches of {P}×{P}")
        if labels is not None:
            return X, np.array(y_patches, dtype=np.int8), coords
        return X, coords

    def reconstruct_from_patches(self, patches: np.ndarray,
                                  coords: list, H: int, W: int,
                                  n_classes: int = 4) -> np.ndarray:
        """
        Reassemble overlapping prediction patches into full-scene probability map.
        Uses averaging for overlapping regions.
        """
        P = self.patch_size
        accum  = np.zeros((H, W, n_classes), dtype=np.float32)
        counts = np.zeros((H, W), dtype=np.float32)

        for patch, (r, c) in zip(patches, coords):
            accum[r:r+P, c:c+P, :] += patch
            counts[r:r+P, c:c+P]   += 1

        counts[counts == 0] = 1
        return accum / counts[..., np.newaxis]


# ── Tabular Feature Extractor (for RF / XGBoost) ─────────────────────────────
class TabularFeatureExtractor:
    """
    Converts raster stack to tabular dataframe for Random Forest / XGBoost.
    Each pixel = one row.
    """

    EXTRA_FEATURES = [
        'lithology_class', 'forest_type', 'dist_to_river_m',
        'dist_to_road_m', 'population_density'
    ]

    def stack_to_dataframe(self, stack: np.ndarray,
                            feature_names: list,
                            labels: np.ndarray = None,
                            mask: np.ndarray   = None) -> pd.DataFrame:
        """
        Flatten (H, W, F) stack to (N_pixels, F) DataFrame.
        mask: binary valid-pixel mask (e.g. non-cloud, non-water-body)
        """
        H, W, F = stack.shape
        flat = stack.reshape(-1, F)
        df   = pd.DataFrame(flat, columns=feature_names)

        if mask is not None:
            valid = mask.reshape(-1).astype(bool)
            df    = df[valid].reset_index(drop=True)

        if labels is not None:
            lbl_flat = labels.reshape(-1)
            if mask is not None:
                lbl_flat = lbl_flat[mask.reshape(-1).astype(bool)]
            df['label'] = lbl_flat.astype(np.int8)

        # Drop mostly-NaN features
        nan_frac = df.isnull().mean()
        drop_cols = nan_frac[nan_frac > 0.5].index.tolist()
        if drop_cols:
            print(f"[TabularExtractor] Dropping high-NaN cols: {drop_cols}")
            df = df.drop(columns=drop_cols)

        # Impute remaining NaN with column median
        df = df.fillna(df.median(numeric_only=True))
        print(f"[TabularExtractor] DataFrame shape: {df.shape}")
        return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Synthetic test data
    H, W = 512, 512
    print("Running feature engineering demo on synthetic data...\n")

    rng = np.random.default_rng(42)
    bands = {
        's2':           rng.uniform(0, 0.5, (H, W, 12)).astype(np.float32),
        'dem':          rng.uniform(500, 4000, (H, W)).astype(np.float32),
        'slope':        rng.uniform(0, 50, (H, W)).astype(np.float32),
        'flow_accum':   rng.exponential(100, (H, W)).astype(np.float32),
        'rainfall_72h': rng.exponential(50, (H, W)).astype(np.float32),
        'soil_moisture':rng.uniform(0.1, 0.5, (H, W)).astype(np.float32),
        'ndvi_prev':    rng.uniform(-0.1, 0.8, (H, W)).astype(np.float32),
        'nbr_prev':     rng.uniform(-0.2, 0.6, (H, W)).astype(np.float32),
        'ssm_anomaly':  rng.normal(0, 0.05, (H, W)).astype(np.float32),
        'lst':          rng.uniform(10, 45, (H, W)).astype(np.float32),
        'wind_speed':   rng.uniform(0, 15, (H, W)).astype(np.float32),
    }

    builder = FeatureStackBuilder()
    stack   = builder.build_feature_stack(bands)
    stack_n = builder.normalize_stack(stack, fit=True,
                                       stats_path='./feature_stats.npz')

    extractor = PatchExtractor(patch_size=256, stride=128)
    X, coords = extractor.extract_patches(stack_n)
    print(f"\nPatch array shape: {X.shape}")
    print("\n✓ Feature engineering complete.")
