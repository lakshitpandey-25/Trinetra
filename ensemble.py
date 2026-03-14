"""
TRINETRA — Ensemble Fusion & Risk Probability Generation

Stacking meta-learner:
  Level-0 models : CNN (spatial) + Random Forest (tabular) + XGBoost (tabular)
  Level-1 model  : Logistic Regression meta-learner on concatenated L0 probs

Risk output:
  Three-class probability output per pixel (fire, flood, landslide)
  Alert level classification (L1 Monitor → L4 Emergency)
  GeoJSON / GeoTIFF risk map export
"""

import numpy as np
import pandas as pd
import json
import joblib
import rasterio
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import shapes as rio_shapes
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    classification_report, roc_auc_score, f1_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path
from typing  import Tuple, Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
HAZARD_CLASSES = {0: 'background', 1: 'fire', 2: 'flood', 3: 'landslide'}
LABEL_NAMES    = list(HAZARD_CLASSES.values())

RISK_THRESHOLDS: Dict[str, Dict[str, float]] = {
    'fire':      {'L1': 0.30, 'L2': 0.50, 'L3': 0.70, 'L4': 0.85},
    'flood':     {'L1': 0.30, 'L2': 0.50, 'L3': 0.70, 'L4': 0.85},
    'landslide': {'L1': 0.25, 'L2': 0.45, 'L3': 0.65, 'L4': 0.80},
}

ALERT_ACTIONS = {
    'L1': 'Internal monitoring only — data collection continues.',
    'L2': 'Alert district officials — pre-position response resources.',
    'L3': 'Issue public SMS/IVR warning — NDRF/SDRF deployment — evacuation advisory.',
    'L4': 'CRITICAL — forced evacuation — close mountain highways — full emergency response.',
}


# ── Ensemble Stacking Classifier ──────────────────────────────────────────────
class EnsembleStackingClassifier:
    """
    Stacking meta-learner that fuses:
      cnn_probs   : (N, 4) — spatial CNN pixel-level predictions
      rf_probs    : (N, 4) — Random Forest tabular predictions
      xgb_probs   : (N, 4) — XGBoost tabular predictions

    Input to meta-learner: 12-dimensional concatenated probability vector
    Meta-learner output:   (N, 4) final class probabilities
    """

    def __init__(self, meta_C: float = 1.0):
        self.meta_learner  = LogisticRegression(
            C           = meta_C,
            max_iter    = 1000,
            multi_class = 'multinomial',
            solver      = 'lbfgs',
            class_weight= 'balanced',
            n_jobs      = -1,
            random_state= 42
        )
        self.meta_scaler   = StandardScaler()
        self.model_weights = None    # learned confidence per L0 model

    def build_meta_features(self,
                             cnn_probs: np.ndarray,
                             rf_probs:  np.ndarray,
                             xgb_probs: np.ndarray,
                             extra_features: Optional[np.ndarray] = None
                             ) -> np.ndarray:
        """
        Concatenate L0 probability vectors.
        Optionally append extra tabular features for context-aware fusion.
        Returns: (N, 12 [+ extra]) meta-feature matrix.
        """
        parts = [cnn_probs, rf_probs, xgb_probs]
        if extra_features is not None:
            parts.append(extra_features)
        return np.concatenate(parts, axis=1)

    def fit(self, cnn_probs: np.ndarray, rf_probs: np.ndarray,
            xgb_probs: np.ndarray, y_true: np.ndarray,
            extra_features: Optional[np.ndarray] = None):
        """
        Train meta-learner on held-out Level-0 predictions
        (these must come from a cross-validation hold-out — not in-sample).
        """
        meta_X = self.build_meta_features(cnn_probs, rf_probs, xgb_probs,
                                           extra_features)
        meta_X_s = self.meta_scaler.fit_transform(meta_X)
        self.meta_learner.fit(meta_X_s, y_true)

        # Evaluate meta-learner on training data
        preds = self.meta_learner.predict(meta_X_s)
        probs = self.meta_learner.predict_proba(meta_X_s)
        print("\n[Ensemble] Meta-Learner Training Report:")
        print(classification_report(y_true, preds,
                                     target_names=LABEL_NAMES, zero_division=0))
        print(f"[Ensemble] AUC-ROC: {roc_auc_score(y_true, probs, multi_class='ovr', average='weighted'):.4f}")
        print(f"[Ensemble] Log-loss: {log_loss(y_true, probs):.4f}")

        # Compute per-model confidence weights (how much does removing it hurt?)
        self._estimate_model_weights(cnn_probs, rf_probs, xgb_probs, y_true)

    def predict(self,
                cnn_probs: np.ndarray,
                rf_probs:  np.ndarray,
                xgb_probs: np.ndarray,
                extra_features: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate final predictions.
        Returns:
          final_class  : (N,)   argmax class labels
          final_probs  : (N, 4) softmax probabilities
        """
        meta_X   = self.build_meta_features(cnn_probs, rf_probs, xgb_probs,
                                             extra_features)
        meta_X_s = self.meta_scaler.transform(meta_X)
        final_probs = self.meta_learner.predict_proba(meta_X_s)
        final_class = np.argmax(final_probs, axis=1)
        return final_class, final_probs

    def predict_with_uncertainty(self,
                                  cnn_probs: np.ndarray,
                                  rf_probs:  np.ndarray,
                                  xgb_probs: np.ndarray
                                  ) -> Dict[str, np.ndarray]:
        """
        Return predictions + aleatoric/epistemic uncertainty estimates.
        Disagreement between L0 models = epistemic uncertainty.
        """
        _, final_probs = self.predict(cnn_probs, rf_probs, xgb_probs)

        # Entropy of ensemble = aleatoric uncertainty
        eps      = 1e-8
        entropy  = -np.sum(final_probs * np.log(final_probs + eps), axis=1)

        # Disagreement between models = epistemic uncertainty
        stacked  = np.stack([cnn_probs, rf_probs, xgb_probs], axis=0)
        mean_p   = stacked.mean(axis=0)
        variance = stacked.var(axis=0).sum(axis=1)

        return {
            'final_probs': final_probs,
            'final_class': np.argmax(final_probs, axis=1),
            'entropy':     entropy,
            'model_variance': variance,
            'confidence':  final_probs.max(axis=1)
        }

    def _estimate_model_weights(self, cnn_probs, rf_probs, xgb_probs, y_true):
        """Ablation: estimate contribution of each L0 model."""
        pairs = [('CNN', cnn_probs), ('RF', rf_probs), ('XGB', xgb_probs)]
        print("\n[Ensemble] Per-model contribution (ablation AUC drop):")
        baseline = roc_auc_score(y_true,
                                  (cnn_probs + rf_probs + xgb_probs) / 3,
                                  multi_class='ovr', average='weighted')
        for name, probs in pairs:
            remaining = [p for n, p in pairs if n != name]
            avg = sum(remaining) / len(remaining)
            score = roc_auc_score(y_true, avg,
                                   multi_class='ovr', average='weighted')
            print(f"  Without {name:5s}: AUC={score:.4f} (drop={baseline-score:+.4f})")

    def evaluate(self, cnn_probs: np.ndarray, rf_probs: np.ndarray,
                  xgb_probs: np.ndarray, y_true: np.ndarray) -> dict:
        """Full evaluation on test set."""
        final_class, final_probs = self.predict(cnn_probs, rf_probs, xgb_probs)
        print("\n[Ensemble] Final Test Report:")
        print(classification_report(y_true, final_class,
                                     target_names=LABEL_NAMES, zero_division=0))
        auc = roc_auc_score(y_true, final_probs, multi_class='ovr', average='weighted')
        f1  = f1_score(y_true, final_class, average='weighted', zero_division=0)
        ll  = log_loss(y_true, final_probs)
        print(f"[Ensemble] AUC-ROC   : {auc:.4f}")
        print(f"[Ensemble] F1 (wtd)  : {f1:.4f}")
        print(f"[Ensemble] Log-loss  : {ll:.4f}")
        return {'auc': auc, 'f1': f1, 'log_loss': ll}

    def save(self, path: str = './models/ensemble_meta.pkl'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'meta_learner': self.meta_learner,
            'meta_scaler':  self.meta_scaler,
        }, path)
        print(f"[Ensemble] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'EnsembleStackingClassifier':
        data = joblib.load(path)
        obj  = cls()
        obj.meta_learner = data['meta_learner']
        obj.meta_scaler  = data['meta_scaler']
        print(f"[Ensemble] Loaded from {path}")
        return obj


# ── Risk Probability Mapper ───────────────────────────────────────────────────
class RiskProbabilityMapper:
    """
    Converts pixel-level probability arrays to:
      - Structured JSON risk output per district
      - Alert level classifications
      - GeoTIFF raster exports
      - GeoJSON vector exports
    """

    @staticmethod
    def classify_alert_level(probability: float,
                              hazard_type: str) -> str:
        """Classify probability into 4-tier alert level."""
        thresholds = RISK_THRESHOLDS[hazard_type]
        for level in ['L4', 'L3', 'L2', 'L1']:
            if probability >= thresholds[level]:
                return level
        return 'NOMINAL'

    def generate_district_risk_output(self,
                                       final_probs: np.ndarray,
                                       district_name: str,
                                       bbox: dict = None) -> dict:
        """
        Generate structured risk probability output for a district.
        final_probs: (N, 4) — all pixels in the district
        """
        fire_p      = float(np.mean(final_probs[:, 1]))
        flood_p     = float(np.mean(final_probs[:, 2]))
        landslide_p = float(np.mean(final_probs[:, 3]))

        # Hotspot percentile (95th) for localised extreme risk
        fire_hot      = float(np.percentile(final_probs[:, 1], 95))
        flood_hot     = float(np.percentile(final_probs[:, 2], 95))
        landslide_hot = float(np.percentile(final_probs[:, 3], 95))

        output = {
            'district'   : district_name,
            'timestamp'  : datetime.utcnow().isoformat() + 'Z',
            'model_version': 'TRINETRA-v1.0',
            'pixel_count': int(final_probs.shape[0]),
            'bbox'       : bbox,
            'hazard_probabilities': {
                'fire': {
                    'mean_probability' : round(fire_p, 4),
                    'hotspot_95th_pct' : round(fire_hot, 4),
                    'alert_level'      : self.classify_alert_level(fire_p, 'fire'),
                    'hotspot_level'    : self.classify_alert_level(fire_hot, 'fire'),
                    'recommended_action': ALERT_ACTIONS[
                        self.classify_alert_level(fire_p, 'fire')
                        if self.classify_alert_level(fire_p, 'fire') != 'NOMINAL'
                        else 'L1'
                    ]
                },
                'flood': {
                    'mean_probability' : round(flood_p, 4),
                    'hotspot_95th_pct' : round(flood_hot, 4),
                    'alert_level'      : self.classify_alert_level(flood_p, 'flood'),
                    'hotspot_level'    : self.classify_alert_level(flood_hot, 'flood'),
                    'recommended_action': ALERT_ACTIONS.get(
                        self.classify_alert_level(flood_p, 'flood'), '')
                },
                'landslide': {
                    'mean_probability' : round(landslide_p, 4),
                    'hotspot_95th_pct' : round(landslide_hot, 4),
                    'alert_level'      : self.classify_alert_level(landslide_p, 'landslide'),
                    'hotspot_level'    : self.classify_alert_level(landslide_hot, 'landslide'),
                    'recommended_action': ALERT_ACTIONS.get(
                        self.classify_alert_level(landslide_p, 'landslide'), '')
                }
            },
            'composite_risk': round(max(fire_p, flood_p, landslide_p), 4),
            'dominant_hazard': ['background', 'fire', 'flood', 'landslide'][
                int(np.argmax([0, fire_p, flood_p, landslide_p]))
            ]
        }
        return output

    def export_risk_geotiff(self,
                             prob_array: np.ndarray,
                             transform,
                             crs,
                             output_path: str,
                             description: str = 'TRINETRA Risk Map'):
        """
        Export 3-band risk probability GeoTIFF.
        Band 1 = fire probability
        Band 2 = flood probability
        Band 3 = landslide probability
        Values: float32 [0.0, 1.0]
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        H, W = prob_array.shape[1], prob_array.shape[2]
        with rasterio.open(
            output_path, 'w',
            driver    = 'GTiff',
            height    = H,
            width     = W,
            count     = 3,
            dtype     = rasterio.float32,
            crs       = crs,
            transform = transform,
            compress  = 'lzw',
            predictor = 3,       # float predictor for better compression
            tiled     = True,
            blockxsize= 256,
            blockysize= 256,
        ) as dst:
            dst.write(prob_array[1].astype(np.float32), 1)  # fire
            dst.write(prob_array[2].astype(np.float32), 2)  # flood
            dst.write(prob_array[3].astype(np.float32), 3)  # landslide
            dst.update_tags(
                model       = 'TRINETRA-v1.0',
                bands       = '1=fire,2=flood,3=landslide',
                created_at  = datetime.utcnow().isoformat(),
                description = description
            )
        print(f"[RiskMapper] GeoTIFF saved → {output_path}")

    def export_risk_geojson(self,
                             risk_map: np.ndarray,
                             transform,
                             crs,
                             threshold: float = 0.5,
                             output_path: str = './outputs/riskmap.geojson'):
        """
        Vectorise risk probability raster to GeoJSON polygon features.
        Polygons where risk > threshold are exported.
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        hazard_names = {1: 'fire', 2: 'flood', 3: 'landslide'}
        all_features = []

        for band_idx, hazard in hazard_names.items():
            band = risk_map[band_idx]
            mask = (band > threshold).astype(np.uint8)
            if mask.max() == 0:
                continue
            for geom, val in rio_shapes(
                    band.astype(np.float32), mask=mask, transform=transform):
                all_features.append({
                    'type'      : 'Feature',
                    'geometry'  : geom,
                    'properties': {
                        'hazard'     : hazard,
                        'probability': round(float(val), 4),
                        'alert_level': self.classify_alert_level(float(val), hazard),
                        'timestamp'  : datetime.utcnow().isoformat()
                    }
                })

        geojson = {'type': 'FeatureCollection', 'features': all_features}
        with open(output_path, 'w') as f:
            json.dump(geojson, f)
        print(f"[RiskMapper] GeoJSON saved → {output_path} "
              f"({len(all_features)} features)")
        return geojson

    def generate_risk_summary_table(self,
                                     district_outputs: List[dict]) -> pd.DataFrame:
        """Build summary DataFrame from list of district risk outputs."""
        rows = []
        for out in district_outputs:
            row = {'district': out['district'],
                   'timestamp': out['timestamp'],
                   'composite_risk': out['composite_risk'],
                   'dominant_hazard': out['dominant_hazard']}
            for hz, data in out['hazard_probabilities'].items():
                row[f'{hz}_prob']  = data['mean_probability']
                row[f'{hz}_level'] = data['alert_level']
            rows.append(row)
        return pd.DataFrame(rows).sort_values('composite_risk', ascending=False)


# ── Full Inference Pipeline ───────────────────────────────────────────────────
class TRINETRAInferencePipeline:
    """
    End-to-end inference pipeline wrapping all three models + ensemble.
    Accepts pre-processed feature arrays and returns risk map.
    """

    def __init__(self,
                 cnn_model_path:      str = './models/best_unet.keras',
                 rf_model_path:       str = './models/rf_hazard.pkl',
                 xgb_model_path:      str = './models/xgb_hazard.json',
                 ensemble_model_path: str = './models/ensemble_meta.pkl'):
        import tensorflow as tf
        from tensorflow import keras
        import xgboost as xgb

        print("[Pipeline] Loading models...")
        self.cnn     = keras.models.load_model(cnn_model_path)
        self.rf_data = joblib.load(rf_model_path)
        self.xgb     = xgb.XGBClassifier()
        self.xgb.load_model(xgb_model_path)
        self.ensemble = EnsembleStackingClassifier.load(ensemble_model_path)
        self.mapper   = RiskProbabilityMapper()
        print("[Pipeline] All models loaded ✓")

    def run(self,
            feature_stack: np.ndarray,
            patches:       np.ndarray,
            patch_coords:  list,
            image_shape:   tuple,
            district_name: str,
            transform,
            crs,
            output_dir: str = './outputs') -> dict:
        """
        Full inference:
          1. CNN → patch-level probs → reassemble full scene
          2. RF  → pixel-level tabular probs
          3. XGB → pixel-level tabular probs
          4. Ensemble fusion
          5. Risk map export
          6. District risk output JSON
        """
        from feature_engineering import PatchExtractor
        H, W, _ = feature_stack.shape
        extractor = PatchExtractor(256, 128)

        # 1. CNN inference
        print("[Pipeline] CNN inference...")
        cnn_patch_probs = self.cnn.predict(patches, batch_size=8, verbose=1)
        cnn_full = extractor.reconstruct_from_patches(
            cnn_patch_probs, patch_coords, H, W, n_classes=4)
        cnn_flat = cnn_full.reshape(-1, 4)

        # 2. RF inference
        print("[Pipeline] RF inference...")
        X_tab = feature_stack.reshape(-1, feature_stack.shape[-1])
        rf_probs = self.rf_data['calibrated'].predict_proba(X_tab)

        # 3. XGB inference
        print("[Pipeline] XGBoost inference...")
        xgb_probs = self.xgb.predict_proba(X_tab)

        # 4. Ensemble fusion
        print("[Pipeline] Ensemble fusion...")
        _, final_probs = self.ensemble.predict(cnn_flat, rf_probs, xgb_probs)
        final_map = final_probs.reshape(H, W, 4).transpose(2, 0, 1)

        # 5. Export
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S')
        self.mapper.export_risk_geotiff(
            final_map, transform, crs,
            f'{output_dir}/{district_name}_risk_{ts}.tif'
        )
        self.mapper.export_risk_geojson(
            final_map, transform, crs,
            output_path=f'{output_dir}/riskmap_{district_name}.geojson'
        )

        # 6. District JSON output
        risk_output = self.mapper.generate_district_risk_output(
            final_probs, district_name)
        out_json = f'{output_dir}/risk_{district_name}_{ts}.json'
        with open(out_json, 'w') as f:
            json.dump(risk_output, f, indent=2)
        print(f"[Pipeline] Risk JSON → {out_json}")
        print(json.dumps(risk_output, indent=2))
        return risk_output


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("TRINETRA — Ensemble & Risk Mapper")
    print("=" * 55)

    # Synthetic demo
    rng = np.random.default_rng(42)
    N   = 10_000

    cnn_probs = rng.dirichlet(np.ones(4), N).astype(np.float32)
    rf_probs  = rng.dirichlet(np.ones(4), N).astype(np.float32)
    xgb_probs = rng.dirichlet(np.ones(4), N).astype(np.float32)
    y_true    = rng.choice([0,1,2,3], N, p=[0.82, 0.08, 0.04, 0.06])

    ensemble = EnsembleStackingClassifier()
    ensemble.fit(cnn_probs, rf_probs, xgb_probs, y_true)
    ensemble.evaluate(cnn_probs, rf_probs, xgb_probs, y_true)
    ensemble.save()

    # Risk mapper demo
    mapper = RiskProbabilityMapper()
    risk_out = mapper.generate_district_risk_output(
        np.column_stack([rng.dirichlet(np.ones(4), 500)]),
        district_name='Chamoli'
    )
    print("\nSample risk output:")
    print(json.dumps(risk_out, indent=2))
    print("\n✓ Ensemble pipeline complete.")
