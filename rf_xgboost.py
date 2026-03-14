"""
TRINETRA — Random Forest & XGBoost Hazard Classifiers
Tabular geospatial features → multi-class hazard risk probability.

Classes:
  0 = No hazard (background)
  1 = Forest fire
  2 = Flood
  3 = Landslide

Workflow:
  1. Load feature dataframe
  2. SMOTE oversampling for rare event classes
  3. Spatial k-fold cross-validation
  4. Train RF + XGBoost with Bayesian HPO
  5. Platt scaling calibration
  6. SHAP explainability
  7. Save models
"""

import numpy as np
import pandas as pd
import joblib
import json
import shap
import optuna
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import StratifiedKFold, cross_validate
from sklearn.preprocessing    import StandardScaler, LabelEncoder
from sklearn.calibration      import CalibratedClassifierCV
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (
    classification_report, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection       import permutation_importance
from imblearn.over_sampling   import SMOTE, ADASYN
from imblearn.pipeline        import Pipeline as ImbPipeline
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Feature schema ────────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # Spectral
    'ndvi', 'ndwi', 'nbr', 'ndmi', 'bai', 'evi', 'bsi', 'vci',
    # Terrain
    'elevation', 'slope_deg', 'aspect_deg', 'curvature', 'twi',
    'tpi', 'roughness',
    # Hydrological
    'rainfall_72h', 'rainfall_extreme', 'api_7d', 'soil_moisture',
    # Change detection
    'ndvi_diff_30d', 'nbr_diff_30d', 'ssm_anomaly',
    # Ancillary
    'lst_celsius', 'wind_speed', 'flow_accumulation',
    # Static (from GIS layers)
    'lithology_class', 'forest_type',
    'dist_to_river_m', 'dist_to_road_m', 'population_density',
]

LABEL_NAMES  = ['background', 'fire', 'flood', 'landslide']
N_CLASSES    = 4
RANDOM_STATE = 42


# ── Spatial Cross-Validator ───────────────────────────────────────────────────
class SpatialKFoldSplitter:
    """
    Creates spatially-blocked k-fold splits to prevent data leakage.
    Pixels within the same geographic block are kept together in
    either train OR test — never split across.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def create_folds(self, df: pd.DataFrame,
                     lat_col: str = 'lat', lon_col: str = 'lon') -> list:
        """
        Divide study area into n_splits×n_splits geographic grid.
        Assign each pixel to a block, then build k folds from blocks.
        Returns list of (train_idx, test_idx) tuples.
        """
        lat_bins = pd.cut(df[lat_col], bins=self.n_splits, labels=False)
        lon_bins = pd.cut(df[lon_col], bins=self.n_splits, labels=False)
        df = df.copy()
        df['block'] = lat_bins.astype(str) + '_' + lon_bins.astype(str)
        blocks  = df['block'].unique()
        blocks_per_fold = max(1, len(blocks) // self.n_splits)

        folds   = []
        all_idx = np.arange(len(df))
        for fold in range(self.n_splits):
            test_blocks = blocks[fold * blocks_per_fold:(fold + 1) * blocks_per_fold]
            test_mask   = df['block'].isin(test_blocks)
            train_idx   = all_idx[~test_mask]
            test_idx    = all_idx[ test_mask]
            folds.append((train_idx, test_idx))
            print(f"  Fold {fold+1}: train={len(train_idx):,} | test={len(test_idx):,} "
                  f"| test_blocks={len(test_blocks)}")
        return folds

    def create_simple_folds(self, n_samples: int) -> list:
        """Fallback spatial folds without coordinates — geographic block approximation."""
        block_size = n_samples // self.n_splits
        folds = []
        for fold in range(self.n_splits):
            test_start = fold * block_size
            test_end   = test_start + block_size if fold < self.n_splits - 1 else n_samples
            test_idx   = np.arange(test_start, test_end)
            train_idx  = np.concatenate([np.arange(0, test_start),
                                          np.arange(test_end, n_samples)])
            folds.append((train_idx, test_idx))
        return folds


# ── Random Forest Classifier ──────────────────────────────────────────────────
class RandomForestHazardClassifier:
    """
    Random Forest with SMOTE, spatial CV, calibration, and SHAP.
    """

    def __init__(self, n_estimators: int = 500):
        self.rf = RandomForestClassifier(
            n_estimators      = n_estimators,
            max_depth         = 25,
            min_samples_split = 5,
            min_samples_leaf  = 2,
            max_features      = 'sqrt',
            class_weight      = 'balanced_subsample',
            oob_score         = True,
            random_state      = RANDOM_STATE,
            n_jobs            = -1,
            warm_start        = False,
        )
        self.scaler       = StandardScaler()
        self.calibrated   = None
        self.feature_imp  = None

    def preprocess(self, X: pd.DataFrame,
                   fit: bool = True) -> np.ndarray:
        """Scale features; impute NaN with median."""
        X_filled = X[FEATURE_NAMES].copy().fillna(X[FEATURE_NAMES].median())
        if fit:
            return self.scaler.fit_transform(X_filled)
        return self.scaler.transform(X_filled)

    def apply_smote(self, X: np.ndarray,
                    y: np.ndarray) -> tuple:
        """
        SMOTE (Synthetic Minority Over-sampling Technique).
        Generates synthetic minority-class samples via k-NN interpolation.
        Applied only to training fold — never test/validation.
        """
        label_counts = dict(zip(*np.unique(y, return_counts=True)))
        print(f"  [SMOTE] Before: {label_counts}")
        sm = SMOTE(
            sampling_strategy = 'not majority',
            random_state      = RANDOM_STATE,
            k_neighbors       = 5
        )
        X_res, y_res = sm.fit_resample(X, y)
        label_counts_after = dict(zip(*np.unique(y_res, return_counts=True)))
        print(f"  [SMOTE] After : {label_counts_after}")
        return X_res, y_res

    def spatial_cross_validate(self, X: np.ndarray, y: np.ndarray,
                                 folds: list) -> dict:
        """
        Run spatial k-fold CV.
        Returns mean ± std for F1, AUC-ROC, precision, recall.
        """
        results = {'fold':[], 'f1':[], 'auc':[], 'precision':[], 'recall':[]}
        for i, (train_idx, test_idx) in enumerate(folds):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            X_tr_s, y_tr_s = self.apply_smote(X_tr, y_tr)
            self.rf.fit(X_tr_s, y_tr_s)
            probs = self.rf.predict_proba(X_te)
            preds = np.argmax(probs, axis=1)
            f1  = f1_score(y_te, preds, average='weighted', zero_division=0)
            auc = roc_auc_score(y_te, probs, multi_class='ovr',
                                average='weighted')
            pre = precision_score(y_te, preds, average='weighted', zero_division=0)
            rec = recall_score(y_te, preds, average='weighted', zero_division=0)
            results['fold'].append(i+1)
            results['f1'].append(f1)
            results['auc'].append(auc)
            results['precision'].append(pre)
            results['recall'].append(rec)
            print(f"  Fold {i+1}: F1={f1:.3f} | AUC={auc:.3f} | "
                  f"P={pre:.3f} | R={rec:.3f}")
        summary = {k: (float(np.mean(v)), float(np.std(v)))
                   for k, v in results.items() if k != 'fold'}
        print(f"\n  CV Summary (mean ± std):")
        for k, (m, s) in summary.items():
            print(f"    {k:15s}: {m:.3f} ± {s:.3f}")
        return summary

    def train_and_calibrate(self, X_train: np.ndarray,
                             y_train: np.ndarray):
        """Final training on full train split with calibration."""
        X_s, y_s = self.apply_smote(X_train, y_train)
        self.rf.fit(X_s, y_s)
        print(f"\n[RF] OOB Accuracy: {self.rf.oob_score_:.4f}")

        # Platt scaling calibration for probability reliability
        self.calibrated = CalibratedClassifierCV(
            estimator  = self.rf,
            method     = 'isotonic',
            cv         = 'prefit',
            n_jobs     = -1
        )
        self.calibrated.fit(X_train, y_train)
        print("[RF] Probability calibration complete (isotonic).")

    def get_feature_importance(self) -> pd.DataFrame:
        """Mean decrease in impurity (MDI) feature importance."""
        imp_df = pd.DataFrame({
            'feature':    FEATURE_NAMES[:len(self.rf.feature_importances_)],
            'importance': self.rf.feature_importances_
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        self.feature_imp = imp_df
        return imp_df

    def permutation_importance(self, X_val: np.ndarray,
                                y_val: np.ndarray, n_repeats: int = 10) -> pd.DataFrame:
        """
        Permutation importance (model-agnostic, unbiased vs MDI).
        Shuffles each feature and measures accuracy drop.
        """
        result = permutation_importance(
            self.rf, X_val, y_val,
            n_repeats    = n_repeats,
            random_state = RANDOM_STATE,
            n_jobs       = -1,
            scoring      = 'f1_weighted'
        )
        imp_df = pd.DataFrame({
            'feature': FEATURE_NAMES[:X_val.shape[1]],
            'importance_mean': result.importances_mean,
            'importance_std':  result.importances_std
        }).sort_values('importance_mean', ascending=False)
        return imp_df

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrated is not None:
            return self.calibrated.predict_proba(X)
        return self.rf.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray):
        preds = self.rf.predict(X_test)
        probs = self.predict_proba(X_test)
        print("\n[RF] Classification Report:")
        print(classification_report(y_test, preds,
                                     target_names=LABEL_NAMES, zero_division=0))
        auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
        print(f"[RF] AUC-ROC (OvR weighted): {auc:.4f}")
        return {'auc': auc, 'f1': f1_score(y_test, preds,
                                             average='weighted', zero_division=0)}

    def save(self, path: str = './models/rf_hazard.pkl'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'rf':         self.rf,
            'scaler':     self.scaler,
            'calibrated': self.calibrated,
            'feature_names': FEATURE_NAMES
        }, path)
        print(f"[RF] Saved → {path}")

    @classmethod
    def load(cls, path: str) -> 'RandomForestHazardClassifier':
        data = joblib.load(path)
        obj  = cls()
        obj.rf         = data['rf']
        obj.scaler     = data['scaler']
        obj.calibrated = data['calibrated']
        print(f"[RF] Loaded from {path}")
        return obj


# ── XGBoost Classifier ────────────────────────────────────────────────────────
class XGBoostHazardClassifier:
    """
    XGBoost gradient boosting with Bayesian HPO (Optuna) and SHAP.
    """

    def __init__(self):
        self.model       = None
        self.best_params = None
        self.scaler      = StandardScaler()
        self.explainer   = None

    def _optuna_objective(self, trial, X_train: np.ndarray,
                           y_train: np.ndarray) -> float:
        """
        Optuna objective for Bayesian hyperparameter optimisation.
        Returns weighted OvR AUC-ROC (3-fold stratified CV).
        """
        params = {
            'n_estimators':     trial.suggest_int('n_estimators',     500,  2000),
            'max_depth':        trial.suggest_int('max_depth',         4,    12),
            'learning_rate':    trial.suggest_float('learning_rate',   0.01, 0.3,  log=True),
            'subsample':        trial.suggest_float('subsample',       0.6,  1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree',0.5,  1.0),
            'colsample_bylevel':trial.suggest_float('colsample_bylevel',0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha',       1e-5, 10.0, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda',      1e-5, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight',  1,    10),
            'gamma':            trial.suggest_float('gamma',           0.0,  5.0),
            'max_delta_step':   trial.suggest_int('max_delta_step',    0,    10),
            'objective':        'multi:softprob',
            'num_class':        N_CLASSES,
            'tree_method':      'hist',
            'device':           'cuda',
            'eval_metric':      'mlogloss',
            'use_label_encoder': False,
            'random_state':     RANDOM_STATE,
            'n_jobs':           -1,
        }
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr, va in skf.split(X_train, y_train):
            clf = xgb.XGBClassifier(**params)
            clf.fit(
                X_train[tr], y_train[tr],
                eval_set             = [(X_train[va], y_train[va])],
                verbose              = False,
                early_stopping_rounds= 50
            )
            probs = clf.predict_proba(X_train[va])
            scores.append(roc_auc_score(y_train[va], probs,
                                         multi_class='ovr', average='weighted'))
        return float(np.mean(scores))

    def optimise_hyperparameters(self, X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  n_trials: int = 100,
                                  timeout: int = 3600) -> dict:
        """
        Run Bayesian HPO with Optuna (Tree-structured Parzen Estimator).
        n_trials: number of trials (100 ≈ good coverage)
        timeout:  max wall-clock seconds (3600 = 1 hour)
        """
        print(f"\n[XGB] Starting HPO: {n_trials} trials, timeout={timeout}s")
        study = optuna.create_study(
            direction  = 'maximize',
            sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE),
            pruner     = optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        study.optimize(
            lambda t: self._optuna_objective(t, X_train, y_train),
            n_trials   = n_trials,
            timeout    = timeout,
            n_jobs     = 1,
            show_progress_bar = True
        )
        self.best_params = study.best_params
        print(f"\n[XGB] Best AUC-ROC: {study.best_value:.4f}")
        print(f"[XGB] Best params:\n{json.dumps(self.best_params, indent=2)}")
        return self.best_params

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray,
              use_smote: bool = True):
        """Train XGBoost on full training data with early stopping."""
        if use_smote:
            sm = SMOTE(sampling_strategy='not majority',
                       random_state=RANDOM_STATE, k_neighbors=5)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            print(f"[XGB] Post-SMOTE train size: {len(X_train):,}")

        params = {
            **(self.best_params or {}),
            'n_estimators':      self.best_params.get('n_estimators', 1200) if self.best_params else 1200,
            'objective':         'multi:softprob',
            'num_class':         N_CLASSES,
            'tree_method':       'hist',
            'eval_metric':       ['mlogloss', 'merror'],
            'use_label_encoder': False,
            'random_state':      RANDOM_STATE,
            'n_jobs':            -1,
        }

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set             = [(X_val, y_val)],
            verbose              = 100,
            early_stopping_rounds= 50
        )
        print(f"[XGB] Best iteration: {self.model.best_iteration}")

    def compute_shap(self, X_sample: np.ndarray,
                     save_path: str = './logs/shap_summary.png'):
        """
        SHAP TreeExplainer — feature attribution for each class.
        Produces global summary plot and per-class bar plots.
        X_sample: (N, F) — representative sample (500–1000 rows recommended)
        """
        print("[XGB] Computing SHAP values...")
        self.explainer = shap.TreeExplainer(self.model)
        shap_values    = self.explainer.shap_values(X_sample)

        # Summary plot (all classes combined)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            feature_names = FEATURE_NAMES[:X_sample.shape[1]],
            plot_type     = 'bar',
            class_names   = LABEL_NAMES,
            show          = False
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[XGB] SHAP summary saved → {save_path}")
        return shap_values

    def get_feature_importance_df(self) -> pd.DataFrame:
        """XGBoost built-in feature importance (weight / gain / cover)."""
        imp_dict = self.model.get_booster().get_fscore()
        imp_df   = pd.DataFrame({
            'feature':    list(imp_dict.keys()),
            'importance': list(imp_dict.values())
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        return imp_df

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        preds = self.model.predict(X_test)
        probs = self.predict_proba(X_test)
        print("\n[XGB] Classification Report:")
        print(classification_report(y_test, preds,
                                     target_names=LABEL_NAMES, zero_division=0))
        auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')
        f1  = f1_score(y_test, preds, average='weighted', zero_division=0)
        print(f"[XGB] AUC-ROC (OvR weighted): {auc:.4f}")
        print(f"[XGB] F1 (weighted):           {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(ax=ax, cmap='Blues')
        plt.title('XGBoost Confusion Matrix')
        plt.savefig('./logs/xgb_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        return {'auc': auc, 'f1': f1}

    def save(self, model_path: str = './models/xgb_hazard.json',
             scaler_path: str = './models/xgb_scaler.pkl'):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(model_path)
        joblib.dump({'scaler': self.scaler,
                     'best_params': self.best_params,
                     'feature_names': FEATURE_NAMES}, scaler_path)
        print(f"[XGB] Saved → {model_path} | {scaler_path}")

    @classmethod
    def load(cls, model_path: str,
             scaler_path: str = './models/xgb_scaler.pkl') -> 'XGBoostHazardClassifier':
        obj   = cls()
        obj.model = xgb.XGBClassifier()
        obj.model.load_model(model_path)
        data  = joblib.load(scaler_path)
        obj.scaler      = data['scaler']
        obj.best_params = data.get('best_params')
        print(f"[XGB] Loaded from {model_path}")
        return obj


# ── Calibration Evaluation ────────────────────────────────────────────────────
def reliability_diagram(y_true: np.ndarray, y_proba: np.ndarray,
                         class_id: int = 1, n_bins: int = 10,
                         class_name: str = 'Fire',
                         save_path: str = './logs/reliability.png'):
    """
    Reliability (calibration) diagram for a single hazard class.
    Well-calibrated model: points fall on the 45° diagonal.
    """
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(
        (y_true == class_id).astype(int),
        y_proba[:, class_id],
        n_bins = n_bins
    )
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.plot(prob_pred, prob_true, 's-', label=f'{class_name}')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Reliability Diagram — {class_name}')
    plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Calibration] Reliability diagram saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("TRINETRA — RF + XGBoost Classifiers")
    print("=" * 55)

    # Synthetic dataset
    rng = np.random.default_rng(42)
    N   = 50_000
    X   = rng.standard_normal((N, len(FEATURE_NAMES))).astype(np.float32)
    y   = rng.choice([0, 1, 2, 3], size=N, p=[0.82, 0.08, 0.04, 0.06]).astype(np.int8)
    print(f"Dataset: {X.shape}, label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Spatial fold splitter
    splitter = SpatialKFoldSplitter(n_splits=5)
    folds    = splitter.create_simple_folds(N)

    # Random Forest
    print("\n--- Random Forest ---")
    rf_clf = RandomForestHazardClassifier(n_estimators=100)
    X_scaled = rf_clf.preprocess(
        pd.DataFrame(X, columns=FEATURE_NAMES), fit=True
    )
    cv_summary = rf_clf.spatial_cross_validate(X_scaled, y, folds)

    split = int(N * 0.8)
    rf_clf.train_and_calibrate(X_scaled[:split], y[:split])
    rf_metrics = rf_clf.evaluate(X_scaled[split:], y[split:])
    rf_clf.save('./models/rf_hazard.pkl')

    # XGBoost (quick demo — HPO disabled for speed)
    print("\n--- XGBoost ---")
    xgb_clf = XGBoostHazardClassifier()
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    xgb_clf.train(X_tr, y_tr, X_te, y_te, use_smote=True)
    xgb_metrics = xgb_clf.evaluate(X_te, y_te)
    xgb_clf.save('./models/xgb_hazard.json')

    print("\n✓ RF and XGBoost training complete.")
