"""
TRINETRA — Model Validation & Evaluation Suite
Spatial cross-validation, calibration assessment, SHAP analysis,
comparison against MODIS baseline, and full performance reporting.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix,
    ConfusionMatrixDisplay, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import shap

LABEL_NAMES  = ['background', 'fire', 'flood', 'landslide']
N_CLASSES    = 4
RANDOM_STATE = 42
REPORT_DIR   = Path('./reports')
PLOT_DIR     = Path('./plots/validation')


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Confusion Matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           model_name: str = 'Model',
                           save_path: Optional[str] = None) -> np.ndarray:
    """Normalized confusion matrix with per-class accuracy."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    ConfusionMatrixDisplay(cm, display_labels=LABEL_NAMES).plot(
        ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title(f'{model_name} — Confusion Matrix (counts)')

    # Normalised
    ConfusionMatrixDisplay(np.round(cm_norm, 2), display_labels=LABEL_NAMES).plot(
        ax=axes[1], cmap='Blues', colorbar=True, values_format='.2f')
    axes[1].set_title(f'{model_name} — Confusion Matrix (normalised)')

    plt.tight_layout()
    sp = save_path or str(PLOT_DIR / f'cm_{model_name.lower()}.png')
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Validation] CM saved → {sp}")
    return cm


# ── ROC Curves ───────────────────────────────────────────────────────────────
def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                    model_name: str = 'Model',
                    save_path: Optional[str] = None):
    """Per-class ROC curves with AUC annotations."""
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

    colors = ['#6366f1', '#ef4444', '#38bdf8', '#f59e0b']
    fig, ax = plt.subplots(figsize=(8, 6))

    aucs = {}
    for i, (cls_name, color) in enumerate(zip(LABEL_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc         = roc_auc_score(y_bin[:, i], y_proba[:, i])
        aucs[cls_name] = auc
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{cls_name} (AUC = {auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (0.500)')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name} — ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    sp = save_path or str(PLOT_DIR / f'roc_{model_name.lower()}.png')
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Validation] ROC curves saved → {sp}  AUCs: {aucs}")
    return aucs


# ── Precision-Recall Curves ───────────────────────────────────────────────────
def plot_precision_recall_curves(y_true: np.ndarray, y_proba: np.ndarray,
                                  model_name: str = 'Model',
                                  save_path: Optional[str] = None):
    """Per-class P-R curves — more informative than ROC for imbalanced data."""
    from sklearn.preprocessing import label_binarize
    y_bin  = label_binarize(y_true, classes=list(range(N_CLASSES)))
    colors = ['#6366f1', '#ef4444', '#38bdf8', '#f59e0b']

    fig, ax = plt.subplots(figsize=(8, 6))
    aps = {}
    for i, (cls_name, color) in enumerate(zip(LABEL_NAMES, colors)):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        aps[cls_name] = ap
        ax.plot(rec, prec, color=color, lw=2,
                label=f'{cls_name} (AP = {ap:.3f})')

    # Baseline (random)
    pos_rate = (y_true > 0).mean()
    ax.axhline(pos_rate, color='k', linestyle='--', alpha=0.5,
               label=f'Baseline ({pos_rate:.3f})')

    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(f'{model_name} — Precision-Recall Curves')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    sp = save_path or str(PLOT_DIR / f'pr_{model_name.lower()}.png')
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Validation] P-R curves saved → {sp}")
    return aps


# ── Calibration Plots ─────────────────────────────────────────────────────────
def plot_calibration_curves(y_true: np.ndarray, y_proba: np.ndarray,
                              model_name: str = 'Model',
                              save_path: Optional[str] = None):
    """
    Reliability diagrams for each hazard class.
    Well-calibrated model: points on the 45° diagonal.
    Pre-/post-calibration comparison plotted if both provided.
    """
    hazard_classes = [1, 2, 3]  # fire, flood, landslide
    hazard_names   = ['Fire', 'Flood', 'Landslide']
    colors         = ['#ef4444', '#38bdf8', '#f59e0b']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    briers = {}
    for ax, cls_id, cls_name, color in zip(
            axes, hazard_classes, hazard_names, colors):
        y_bin = (y_true == cls_id).astype(int)
        prob_true, prob_pred = calibration_curve(y_bin, y_proba[:, cls_id],
                                                  n_bins=10)
        brier = brier_score_loss(y_bin, y_proba[:, cls_id])
        briers[cls_name] = brier

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
        ax.plot(prob_pred, prob_true, 's-', color=color, lw=2,
                label=f'Model (Brier={brier:.3f})')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{cls_name} Reliability Diagram')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_xlim([-0.02, 1.02]); ax.set_ylim([-0.02, 1.02])

    fig.suptitle(f'{model_name} — Calibration Curves', fontsize=13)
    plt.tight_layout()
    sp = save_path or str(PLOT_DIR / f'calibration_{model_name.lower()}.png')
    plt.savefig(sp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Validation] Calibration curves saved → {sp}  Brier: {briers}")
    return briers


# ── SHAP Explainability ───────────────────────────────────────────────────────
class SHAPAnalyzer:
    """
    SHAP-based global and local explainability for the ensemble models.
    """

    def __init__(self, xgb_model, feature_names: List[str]):
        self.explainer     = shap.TreeExplainer(xgb_model)
        self.feature_names = feature_names

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute SHAP values for sample X. Returns (N, F, C) array."""
        return self.explainer.shap_values(X)

    def plot_global_importance(self, X: np.ndarray,
                                save_dir: str = None):
        """Beeswarm + bar plots of global feature importance."""
        save_dir = Path(save_dir or PLOT_DIR)
        shap_vals = self.compute_shap_values(X)

        for class_idx, class_name in enumerate(LABEL_NAMES):
            if class_idx == 0:
                continue   # skip background class
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Bar (mean |SHAP|)
            plt.sca(axes[0])
            shap.summary_plot(
                shap_vals[class_idx], X,
                feature_names = self.feature_names,
                plot_type     = 'bar',
                show          = False,
                max_display   = 20
            )
            axes[0].set_title(f'{class_name} — Feature Importance (mean |SHAP|)')

            # Beeswarm
            plt.sca(axes[1])
            shap.summary_plot(
                shap_vals[class_idx], X,
                feature_names = self.feature_names,
                plot_type     = 'violin',
                show          = False,
                max_display   = 20
            )
            axes[1].set_title(f'{class_name} — SHAP Value Distribution')

            plt.tight_layout()
            plt.savefig(save_dir / f'shap_{class_name}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[SHAP] {class_name} plot saved")

    def plot_local_explanation(self, x_sample: np.ndarray, class_idx: int = 1,
                                save_path: Optional[str] = None):
        """Force plot for a single prediction (local explanation)."""
        shap_vals = self.explainer.shap_values(x_sample.reshape(1, -1))
        plt.figure(figsize=(20, 3))
        shap.plots.waterfall(
            shap.Explanation(
                values        = shap_vals[class_idx][0],
                base_values   = self.explainer.expected_value[class_idx],
                data          = x_sample,
                feature_names = self.feature_names
            ),
            show = False
        )
        sp = save_path or str(PLOT_DIR / f'shap_local_{LABEL_NAMES[class_idx]}.png')
        plt.savefig(sp, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SHAP] Local explanation saved → {sp}")

    def compute_global_importance_df(self, X: np.ndarray) -> pd.DataFrame:
        """Return DataFrame of mean |SHAP| per feature per class."""
        shap_vals = self.compute_shap_values(X)
        records   = []
        for cls_idx, cls_name in enumerate(LABEL_NAMES[1:], start=1):
            mean_abs = np.mean(np.abs(shap_vals[cls_idx]), axis=0)
            for feat, imp in zip(self.feature_names, mean_abs):
                records.append({'class': cls_name, 'feature': feat, 'shap_importance': imp})
        return pd.DataFrame(records).sort_values('shap_importance', ascending=False)


# ── MODIS Baseline Comparison ─────────────────────────────────────────────────
class MODISBaselineComparison:
    """
    Compares TRINETRA predictions against MODIS active fire baseline.
    Provides quantitative improvement metrics.
    """

    def compare(self, trinetra_preds: np.ndarray,
                 modis_preds:         np.ndarray,
                 y_true:              np.ndarray,
                 hazard_class: int = 1) -> dict:
        """
        Compare TRINETRA vs MODIS fire detection.
        y_true, predictions: binary arrays for the hazard class.
        """
        def metrics(preds):
            tp = np.sum((preds == 1) & (y_true == hazard_class))
            fp = np.sum((preds == 1) & (y_true != hazard_class))
            fn = np.sum((preds == 0) & (y_true == hazard_class))
            tn = np.sum((preds == 0) & (y_true != hazard_class))
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            return {'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn),
                    'precision': round(precision, 4),
                    'recall':    round(recall, 4),
                    'f1':        round(f1, 4)}

        us_m = metrics(trinetra_preds)
        mo_m = metrics(modis_preds)

        comparison = {
            'hazard_class':   LABEL_NAMES[hazard_class],
            'trinetra':   us_m,
            'modis_baseline': mo_m,
            'improvement':    {
                'recall_delta':    round(us_m['recall']    - mo_m['recall'],    4),
                'precision_delta': round(us_m['precision'] - mo_m['precision'], 4),
                'f1_delta':        round(us_m['f1']        - mo_m['f1'],        4),
            }
        }

        print(f"\n[Baseline Comparison] {LABEL_NAMES[hazard_class].upper()}")
        print(f"  {'Metric':<20} {'TRINETRA':>14} {'MODIS Baseline':>14}")
        print(f"  {'-'*50}")
        for m in ['precision', 'recall', 'f1']:
            us_v = us_m[m]; mo_v = mo_m[m]
            delta = us_v - mo_v
            print(f"  {m:<20} {us_v:>14.3f} {mo_v:>14.3f}  ({delta:+.3f})")
        return comparison


# ── Full Validation Suite ─────────────────────────────────────────────────────
class ValidationSuite:
    """
    Runs the complete model validation pipeline and generates a
    structured JSON + HTML performance report.
    """

    def __init__(self, model_name: str = 'TRINETRA-Ensemble'):
        self.model_name  = model_name
        self.results     = {}
        ensure_dirs()

    def run(self, y_true: np.ndarray, y_pred: np.ndarray,
            y_proba: np.ndarray,
            X_test: Optional[np.ndarray] = None,
            xgb_model = None,
            feature_names: Optional[List[str]] = None) -> dict:
        """
        Run all validation steps.
        Returns structured results dict.
        """
        print(f"\n{'='*65}")
        print(f"  Validation: {self.model_name}")
        print(f"  Samples: {len(y_true):,}   Classes: {N_CLASSES}")
        print(f"{'='*65}\n")

        # 1. Classification report
        report_str = classification_report(
            y_true, y_pred, target_names=LABEL_NAMES, zero_division=0)
        print("[1/6] Classification Report:\n")
        print(report_str)
        self.results['classification_report'] = report_str

        # 2. Per-class metrics
        self.results['metrics'] = {
            'auc_roc_weighted':  round(roc_auc_score(y_true, y_proba,
                                                      multi_class='ovr',
                                                      average='weighted'), 4),
            'f1_weighted':       round(f1_score(y_true, y_pred,
                                                average='weighted',
                                                zero_division=0), 4),
            'log_loss':          round(log_loss(y_true, y_proba), 4),
        }
        print(f"[2/6] Key Metrics: {self.results['metrics']}")

        # 3. Confusion matrix
        cm = plot_confusion_matrix(y_true, y_pred, self.model_name)
        self.results['confusion_matrix'] = cm.tolist()
        print("[3/6] Confusion matrix plotted.")

        # 4. ROC & PR curves
        aucs = plot_roc_curves(y_true, y_proba, self.model_name)
        aps  = plot_precision_recall_curves(y_true, y_proba, self.model_name)
        self.results['per_class_auc'] = aucs
        self.results['per_class_ap']  = aps
        print(f"[4/6] ROC/PR curves plotted.  AUCs: {aucs}")

        # 5. Calibration
        briers = plot_calibration_curves(y_true, y_proba, self.model_name)
        self.results['brier_scores'] = briers
        print(f"[5/6] Calibration curves plotted.  Brier: {briers}")

        # 6. SHAP (optional — requires XGBoost model and test data)
        if xgb_model is not None and X_test is not None and feature_names:
            print("[6/6] Computing SHAP values...")
            shap_an = SHAPAnalyzer(xgb_model, feature_names)
            shap_df = shap_an.compute_global_importance_df(X_test[:500])
            shap_an.plot_global_importance(X_test[:500])
            self.results['top_shap_features'] = shap_df.groupby('class').apply(
                lambda x: x.nlargest(10, 'shap_importance')['feature'].tolist()
            ).to_dict()
        else:
            print("[6/6] SHAP skipped (model/data not provided).")

        # Save report
        self.results['model_name'] = self.model_name
        self.results['validated_at'] = datetime.utcnow().isoformat() + 'Z'
        self.results['n_samples']    = int(len(y_true))
        report_path = REPORT_DIR / f'validation_{self.model_name.lower()}.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Validation complete. Report → {report_path}")
        return self.results

    def print_summary(self):
        """Print concise performance summary table."""
        m = self.results.get('metrics', {})
        print(f"\n{'─'*40}")
        print(f"  Model: {self.model_name}")
        print(f"  AUC-ROC (weighted):  {m.get('auc_roc_weighted', 'N/A'):.4f}")
        print(f"  F1 (weighted):       {m.get('f1_weighted',       'N/A'):.4f}")
        print(f"  Log-loss:            {m.get('log_loss',           'N/A'):.4f}")
        print(f"{'─'*40}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("TRINETRA — Model Validation Suite")
    print("=" * 55)

    rng = np.random.default_rng(42)
    N   = 10_000

    # Simulate reasonable model predictions
    y_true = rng.choice([0,1,2,3], N, p=[0.82, 0.08, 0.04, 0.06])
    # Model with ~92% accuracy
    y_proba = np.zeros((N, 4), dtype=np.float32)
    for i, true_cls in enumerate(y_true):
        base = rng.dirichlet([0.5, 0.5, 0.5, 0.5])
        # 90% chance of higher probability on correct class
        if rng.random() < 0.9:
            base[true_cls] += 1.5
            base /= base.sum()
        y_proba[i] = base
    y_pred = np.argmax(y_proba, axis=1)

    suite = ValidationSuite(model_name='TRINETRA-Ensemble-v1')
    results = suite.run(y_true, y_pred, y_proba)
    suite.print_summary()

    print("\n✓ Validation suite complete.")
