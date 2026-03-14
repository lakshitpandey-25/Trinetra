"""
TRINETRA — FastAPI REST Server
Serves real-time hazard risk predictions and GeoJSON risk maps.

Endpoints:
  POST /v1/predict          — Single-point hazard prediction
  POST /v1/predict/batch    — Batch point predictions
  GET  /v1/riskmap/{district} — Latest GeoJSON risk map
  GET  /v1/districts        — All districts with current risk levels
  GET  /v1/alerts/active    — All active alerts above L2
  GET  /v1/history/{district} — 30-day risk history
  GET  /v1/status           — System health check
  GET  /v1/models/info      — Loaded model metadata
"""

from fastapi import (FastAPI, HTTPException, BackgroundTasks,
                     Query, Depends, Header)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import joblib
import json
import redis
import asyncio
import uvicorn
import logging
from datetime import datetime, timedelta
from pathlib import Path
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger('trinetra.api')

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = 'TRINETRA API',
    description = 'Real-time multi-hazard risk prediction for Uttarakhand, India',
    version     = '1.0.0',
    docs_url    = '/docs',
    redoc_url   = '/redoc'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ['*'],
    allow_credentials = True,
    allow_methods     = ['*'],
    allow_headers     = ['*'],
)

# ── Redis cache ───────────────────────────────────────────────────────────────
try:
    cache = redis.Redis(host='localhost', port=6379, db=0,
                        decode_responses=True, socket_timeout=2)
    cache.ping()
    REDIS_OK = True
    logger.info("Redis cache connected ✓")
except redis.exceptions.ConnectionError:
    cache   = None
    REDIS_OK = False
    logger.warning("Redis not available — caching disabled")

CACHE_TTL = 900  # 15 minutes

# ── Model registry ────────────────────────────────────────────────────────────
models: Dict[str, object] = {}
model_meta: Dict[str, dict] = {}

RISK_THRESHOLDS = {
    'fire':      {'L1': 0.30, 'L2': 0.50, 'L3': 0.70, 'L4': 0.85},
    'flood':     {'L1': 0.30, 'L2': 0.50, 'L3': 0.70, 'L4': 0.85},
    'landslide': {'L1': 0.25, 'L2': 0.45, 'L3': 0.65, 'L4': 0.80},
}

DISTRICT_LIST = [
    'chamoli', 'rudraprayag', 'pithoragarh', 'uttarkashi',
    'bageshwar', 'champawat', 'tehri', 'pauri', 'almora',
    'nainital', 'haridwar', 'dehradun'
]


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@app.on_event('startup')
async def load_models():
    """Load all ML models at application startup."""
    logger.info("Loading TRINETRA models...")
    model_paths = {
        'rf':       './models/rf_hazard.pkl',
        'xgb':      './models/xgb_hazard.json',
        'ensemble': './models/ensemble_meta.pkl',
    }
    for name, path in model_paths.items():
        if Path(path).exists():
            if name == 'xgb':
                clf = xgb.XGBClassifier()
                clf.load_model(path)
                models[name] = clf
            else:
                models[name] = joblib.load(path)
            model_meta[name] = {
                'path': path,
                'loaded_at': datetime.utcnow().isoformat(),
                'size_mb': round(Path(path).stat().st_size / 1e6, 2)
            }
            logger.info(f"  ✓ {name} loaded ({model_meta[name]['size_mb']} MB)")
        else:
            logger.warning(f"  ✗ {name} not found at {path}")

    # CNN (loaded separately due to TF overhead)
    cnn_path = './models/best_unet.keras'
    if Path(cnn_path).exists():
        models['cnn'] = keras.models.load_model(cnn_path)
        model_meta['cnn'] = {'path': cnn_path, 'loaded_at': datetime.utcnow().isoformat()}
        logger.info("  ✓ CNN loaded")
    logger.info(f"Models ready: {list(models.keys())}")


@app.on_event('shutdown')
async def shutdown():
    logger.info("TRINETRA API shutting down.")


# ── Request / Response Schemas ────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    district:          str   = Field(..., example='chamoli')
    latitude:          float = Field(..., ge=28.7, le=31.5, example=30.4)
    longitude:         float = Field(..., ge=77.5, le=81.1, example=79.3)
    ndvi:              float = Field(..., ge=-1.0, le=1.0,   example=0.42)
    ndwi:              float = Field(..., ge=-1.0, le=1.0,   example=-0.1)
    nbr:               float = Field(..., ge=-1.0, le=1.0,   example=0.35)
    ndmi:              float = Field(..., ge=-1.0, le=1.0,   example=0.2)
    slope_deg:         float = Field(..., ge=0.0,  le=90.0,  example=34.5)
    elevation_m:       float = Field(..., ge=0.0,  le=8848,  example=1800)
    rainfall_72h_mm:   float = Field(..., ge=0.0,             example=186.0)
    soil_moisture:     float = Field(..., ge=0.0,  le=1.0,   example=0.38)
    twi:               float = Field(..., ge=0.0,  le=25.0,  example=7.8)
    lst_celsius:       Optional[float] = Field(25.0, example=28.5)
    wind_speed_ms:     Optional[float] = Field(0.0,  example=4.2)
    bai:               Optional[float] = Field(0.0)
    vci:               Optional[float] = Field(50.0, ge=0, le=100)

    @validator('district')
    def validate_district(cls, v):
        if v.lower() not in DISTRICT_LIST:
            raise ValueError(f"Unknown district: {v}. "
                             f"Valid: {DISTRICT_LIST}")
        return v.lower()


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(..., max_items=1000)


class PredictionResponse(BaseModel):
    district:    str
    timestamp:   str
    latitude:    Optional[float]
    longitude:   Optional[float]
    hazard_probabilities: dict
    composite_risk:       float
    dominant_hazard:      str
    model_version:        str
    cached:               bool = False


# ── Feature builder ───────────────────────────────────────────────────────────
FEATURE_NAMES = [
    'ndvi', 'ndwi', 'nbr', 'ndmi', 'bai', 'evi', 'bsi', 'vci',
    'elevation', 'slope_deg', 'aspect_deg', 'curvature', 'twi',
    'tpi', 'roughness', 'rainfall_72h', 'rainfall_extreme', 'api_7d',
    'soil_moisture', 'ndvi_diff_30d', 'nbr_diff_30d', 'ssm_anomaly',
    'lst_celsius', 'wind_speed', 'flow_accumulation',
    'lithology_class', 'forest_type', 'dist_to_river_m',
    'dist_to_road_m', 'population_density',
]

def request_to_features(req: PredictionRequest) -> np.ndarray:
    """Convert PredictionRequest to feature vector."""
    rain_ext = float(req.rainfall_72h_mm > 100.0)
    api7     = req.rainfall_72h_mm * 0.85
    row = [
        req.ndvi, req.ndwi, req.nbr, req.ndmi, req.bai or 0.0,
        0.0,  # EVI placeholder
        0.0,  # BSI placeholder
        req.vci or 50.0,
        req.elevation_m, req.slope_deg,
        180.0,   # aspect placeholder
        0.0,     # curvature placeholder
        req.twi,
        0.0,     # TPI placeholder
        0.0,     # roughness placeholder
        req.rainfall_72h_mm, rain_ext, api7,
        req.soil_moisture,
        0.0, 0.0, 0.0,   # temporal diff placeholders
        req.lst_celsius, req.wind_speed_ms,
        100.0,   # flow accumulation placeholder
        1.0, 2.0,   # lithology, forest type
        500.0, 200.0, 150.0   # distances, pop density
    ]
    return np.array([row], dtype=np.float32)


def classify_alert(prob: float, hazard: str) -> str:
    t = RISK_THRESHOLDS[hazard]
    for lv in ['L4', 'L3', 'L2', 'L1']:
        if prob >= t[lv]:
            return lv
    return 'NOMINAL'


def run_inference(features: np.ndarray) -> dict:
    """Core inference using available models."""
    # RF
    rf_probs  = (models['rf']['calibrated'].predict_proba(features)
                 if 'rf' in models
                 else np.array([[0.5, 0.2, 0.15, 0.15]]))

    # XGB
    xgb_probs = (models['xgb'].predict_proba(features)
                 if 'xgb' in models
                 else np.array([[0.5, 0.2, 0.15, 0.15]]))

    # CNN placeholder (tabular-only context)
    cnn_probs = np.array([[0.6, 0.15, 0.15, 0.1]])

    # Ensemble
    if 'ensemble' in models:
        meta_X = np.concatenate([cnn_probs, rf_probs, xgb_probs], axis=1)
        meta_X_s = models['ensemble']['meta_scaler'].transform(meta_X)
        final_probs = models['ensemble']['meta_learner'].predict_proba(meta_X_s)[0]
    else:
        final_probs = (cnn_probs[0] + rf_probs[0] + xgb_probs[0]) / 3.0

    return {
        'fire':      float(final_probs[1]),
        'flood':     float(final_probs[2]),
        'landslide': float(final_probs[3]),
        'background':float(final_probs[0]),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post('/v1/predict', response_model=dict, tags=['Prediction'])
async def predict_hazard(request: PredictionRequest,
                          bg: BackgroundTasks):
    """
    Single-point hazard risk prediction.
    Accepts feature values for one geographic pixel / location.
    Returns fire, flood, and landslide probabilities with alert levels.
    """
    # Cache key
    cache_key = (f"pred:{request.district}:"
                 f"{request.latitude:.2f}:{request.longitude:.2f}:"
                 f"{int(request.rainfall_72h_mm)}")

    if REDIS_OK and cache:
        cached = cache.get(cache_key)
        if cached:
            resp = json.loads(cached)
            resp['cached'] = True
            return resp

    features    = request_to_features(request)
    probs       = run_inference(features)

    hazard_out  = {}
    for hz in ['fire', 'flood', 'landslide']:
        p = probs[hz]
        hazard_out[hz] = {
            'probability': round(p, 4),
            'alert_level': classify_alert(p, hz),
            'confidence':  round(min(p * 2, 1.0), 4)
        }

    dominant = max(['fire', 'flood', 'landslide'], key=lambda h: probs[h])
    composite = round(max(probs['fire'], probs['flood'], probs['landslide']), 4)

    result = {
        'district':             request.district,
        'timestamp':            datetime.utcnow().isoformat() + 'Z',
        'latitude':             request.latitude,
        'longitude':            request.longitude,
        'hazard_probabilities': hazard_out,
        'composite_risk':       composite,
        'dominant_hazard':      dominant,
        'model_version':        'TRINETRA-v1.0',
        'cached':               False
    }

    if REDIS_OK and cache:
        cache.setex(cache_key, CACHE_TTL, json.dumps(result))

    bg.add_task(_check_and_dispatch, result)
    logger.info(f"Prediction: {request.district} → "
                f"fire={probs['fire']:.2f} flood={probs['flood']:.2f} "
                f"landslide={probs['landslide']:.2f}")
    return result


@app.post('/v1/predict/batch', tags=['Prediction'])
async def predict_batch(payload: BatchPredictionRequest,
                         bg: BackgroundTasks):
    """
    Batch prediction for multiple locations (up to 1000 per request).
    Useful for district-wide gridded assessment.
    """
    results = []
    feature_matrix = np.vstack([
        request_to_features(req) for req in payload.predictions
    ])

    # Batch inference
    rf_probs  = (models['rf']['calibrated'].predict_proba(feature_matrix)
                 if 'rf' in models
                 else np.full((len(payload.predictions), 4), 0.25))
    xgb_probs = (models['xgb'].predict_proba(feature_matrix)
                 if 'xgb' in models
                 else np.full((len(payload.predictions), 4), 0.25))
    cnn_probs = np.full((len(payload.predictions), 4), [0.6, 0.15, 0.15, 0.1])

    if 'ensemble' in models:
        meta_X   = np.concatenate([cnn_probs, rf_probs, xgb_probs], axis=1)
        meta_X_s = models['ensemble']['meta_scaler'].transform(meta_X)
        all_probs = models['ensemble']['meta_learner'].predict_proba(meta_X_s)
    else:
        all_probs = (cnn_probs + rf_probs + xgb_probs) / 3.0

    for i, req in enumerate(payload.predictions):
        fp = all_probs[i]
        hazard_out = {
            hz: {'probability': round(float(fp[j+1]), 4),
                 'alert_level': classify_alert(float(fp[j+1]), hz)}
            for j, hz in enumerate(['fire', 'flood', 'landslide'])
        }
        results.append({
            'district':             req.district,
            'latitude':             req.latitude,
            'longitude':            req.longitude,
            'hazard_probabilities': hazard_out,
            'composite_risk':       round(float(max(fp[1], fp[2], fp[3])), 4)
        })

    logger.info(f"Batch prediction: {len(results)} locations")
    return {'predictions': results, 'count': len(results),
            'timestamp': datetime.utcnow().isoformat() + 'Z'}


@app.get('/v1/riskmap/{district}', tags=['Risk Maps'])
async def get_risk_map(district: str,
                        hazard: str = Query('all', enum=['all','fire','flood','landslide'])):
    """
    Return latest GeoJSON risk map for a district.
    Optional filter by hazard type.
    """
    district = district.lower()
    geojson_path = Path(f'./outputs/riskmap_{district}.geojson')
    if not geojson_path.exists():
        raise HTTPException(404, f"Risk map not found for '{district}'. "
                                 f"Trigger /v1/predict first.")
    with open(geojson_path) as f:
        geojson = json.load(f)

    if hazard != 'all':
        geojson['features'] = [
            feat for feat in geojson['features']
            if feat['properties'].get('hazard') == hazard
        ]
    return JSONResponse(content=geojson)


@app.get('/v1/districts', tags=['Districts'])
async def get_all_districts():
    """Return current risk levels for all Uttarakhand districts."""
    results = []
    for dist in DISTRICT_LIST:
        json_path = Path(f'./outputs/risk_{dist}_latest.json')
        if json_path.exists():
            with open(json_path) as f:
                results.append(json.load(f))
        else:
            results.append({'district': dist, 'status': 'no_data'})
    return {'districts': results, 'count': len(results),
            'timestamp': datetime.utcnow().isoformat() + 'Z'}


@app.get('/v1/alerts/active', tags=['Alerts'])
async def get_active_alerts(min_level: str = Query('L2',
                            enum=['L1','L2','L3','L4'])):
    """Return all active alerts at or above the specified level."""
    level_order = {'L1': 1, 'L2': 2, 'L3': 3, 'L4': 4}
    min_ord = level_order[min_level]
    alerts  = []
    for dist in DISTRICT_LIST:
        json_path = Path(f'./outputs/risk_{dist}_latest.json')
        if not json_path.exists():
            continue
        with open(json_path) as f:
            risk = json.load(f)
        for hz, data in risk.get('hazard_probabilities', {}).items():
            level = data.get('alert_level', 'NOMINAL')
            if level in level_order and level_order[level] >= min_ord:
                alerts.append({
                    'district':    dist,
                    'hazard':      hz,
                    'probability': data['mean_probability'],
                    'alert_level': level,
                    'timestamp':   risk['timestamp']
                })
    alerts.sort(key=lambda x: x['probability'], reverse=True)
    return {'alerts': alerts, 'count': len(alerts),
            'min_level': min_level,
            'timestamp': datetime.utcnow().isoformat() + 'Z'}


@app.get('/v1/history/{district}', tags=['History'])
async def get_risk_history(district: str,
                            days: int = Query(30, ge=1, le=90),
                            hazard: str = Query('all')):
    """
    Return 30-day risk history for a district.
    Data sourced from alert_log.jsonl.
    """
    log_path = Path('./logs/alert_log.jsonl')
    if not log_path.exists():
        return {'district': district, 'history': [], 'message': 'No history yet'}

    cutoff = datetime.utcnow() - timedelta(days=days)
    history = []
    with open(log_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                if (record.get('district', '').lower() == district.lower() and
                        datetime.fromisoformat(record['ts']) >= cutoff):
                    if hazard == 'all' or record.get('hazard') == hazard:
                        history.append(record)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    return {'district': district, 'history': history,
            'count': len(history), 'days': days}


@app.get('/v1/status', tags=['System'])
async def system_status():
    """System health check — model and cache status."""
    return {
        'status':         'operational',
        'timestamp':      datetime.utcnow().isoformat() + 'Z',
        'models_loaded':  list(models.keys()),
        'models_missing': [m for m in ['rf','xgb','ensemble','cnn']
                           if m not in models],
        'cache_status':   'connected' if REDIS_OK else 'disconnected',
        'version':        '1.0.0',
        'coverage':       'Uttarakhand, India',
        'hazards':        ['forest_fire', 'flood', 'landslide']
    }


@app.get('/v1/models/info', tags=['System'])
async def models_info():
    """Return metadata about loaded models."""
    return {'models': model_meta,
            'timestamp': datetime.utcnow().isoformat() + 'Z'}


# ── Background tasks ──────────────────────────────────────────────────────────
async def _check_and_dispatch(result: dict):
    """Dispatch alerts if risk thresholds are exceeded."""
    try:
        from alert_system import AlertDispatcher
        dispatcher = AlertDispatcher()
        hazards = result.get('hazard_probabilities', {})
        for hz, data in hazards.items():
            level = data.get('alert_level', 'NOMINAL')
            if level in ('L3', 'L4'):
                await dispatcher.dispatch(
                    district    = result['district'],
                    hazard_type = hz,
                    level       = level,
                    probability = data['probability']
                )
    except ImportError:
        logger.debug("Alert dispatcher not available — skipping dispatch")
    except Exception as e:
        logger.error(f"Alert dispatch error: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    uvicorn.run(
        'api_server:app',
        host    = '0.0.0.0',
        port    = 8000,
        reload  = False,
        workers = 4,
        log_level = 'info'
    )
