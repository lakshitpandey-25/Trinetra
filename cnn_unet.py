"""
TRINETRA — U-Net CNN for AI Powered Geospatial Disaster Intelligence Semantic Segmentation
Architecture : U-Net with optional ResNet encoder backbone
Input        : (B, 256, 256, 13) multi-band feature patches
Output       : (B, 256, 256, 4) softmax — [background, fire, flood, landslide]
Loss         : 0.5 * Focal + 0.5 * Dice  (handles class imbalance)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
PATCH_SIZE    = 256
N_CHANNELS    = 25    # full feature stack
N_CLASSES     = 4     # 0=background, 1=fire, 2=flood, 3=landslide
BATCH_SIZE    = 16
LEARNING_RATE = 1e-4
EPOCHS        = 100
CLASS_WEIGHTS = {0: 0.5, 1: 3.5, 2: 3.0, 3: 4.0}   # upweight rare hazard classes

LABEL_NAMES = ['background', 'fire', 'flood', 'landslide']


# ── Building Blocks ───────────────────────────────────────────────────────────
def conv_bn_relu(inputs, filters: int, kernel: int = 3,
                 dropout_rate: float = 0.1):
    """Standard Conv → BatchNorm → ReLU block."""
    x = layers.Conv2D(filters, (kernel, kernel), padding='same',
                      kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate)(x)
    return x


def conv_block(inputs, filters: int, dropout_rate: float = 0.1):
    """Double Conv-BN-ReLU block (U-Net building block)."""
    x = conv_bn_relu(inputs, filters, dropout_rate=dropout_rate)
    x = conv_bn_relu(x, filters, dropout_rate=dropout_rate)
    return x


def encoder_block(inputs, filters: int, dropout_rate: float = 0.1):
    """Encoder: conv_block + max-pool downsampling."""
    skip = conv_block(inputs, filters, dropout_rate)
    pool = layers.MaxPooling2D((2, 2))(skip)
    return skip, pool


def decoder_block(inputs, skip_features, filters: int):
    """
    Decoder: transposed-conv upsample → concat skip → conv_block.
    Skip connections preserve spatial detail from encoder path.
    """
    x = layers.Conv2DTranspose(
        filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, filters, dropout_rate=0.0)
    return x


def attention_gate(g, s, inter_channels: int):
    """
    Attention gate (Oktay et al. 2018).
    Focuses decoder on spatially relevant skip features.
    g = gating signal (decoder), s = skip features (encoder)
    """
    Wg  = layers.Conv2D(inter_channels, (1,1), padding='same')(g)
    Wg  = layers.BatchNormalization()(Wg)
    Ws  = layers.Conv2D(inter_channels, (1,1), padding='same')(s)
    Ws  = layers.BatchNormalization()(Ws)
    psi = layers.Activation('relu')(layers.Add()([Wg, Ws]))
    psi = layers.Conv2D(1, (1,1), padding='same')(psi)
    psi = layers.BatchNormalization()(psi)
    psi = layers.Activation('sigmoid')(psi)
    return layers.Multiply()([s, psi])


# ── U-Net (Vanilla) ──────────────────────────────────────────────────────────
def build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, N_CHANNELS),
               n_classes: int = N_CLASSES,
               use_attention: bool = True) -> Model:
    """
    Build U-Net for multi-hazard pixel segmentation.

    Encoder stages  : 64 → 128 → 256 → 512 filters
    Bottleneck      : 1024 filters, dropout=0.3
    Decoder stages  : 512 → 256 → 128 → 64 filters
    Output head     : 1×1 Conv → softmax (n_classes channels)

    use_attention   : adds attention gates at every skip connection
                      → improves precision for small hazard regions
    """
    inputs = layers.Input(shape=input_shape, name='input_stack')

    # ── Encoder ──
    s1, p1 = encoder_block(inputs, 64,  dropout_rate=0.1)
    s2, p2 = encoder_block(p1,    128,  dropout_rate=0.1)
    s3, p3 = encoder_block(p2,    256,  dropout_rate=0.15)
    s4, p4 = encoder_block(p3,    512,  dropout_rate=0.2)

    # ── Bottleneck ──
    b = conv_block(p4, 1024, dropout_rate=0.3)

    # ── Decoder ──
    if use_attention:
        a4 = attention_gate(b,  s4, 256)
        d4 = decoder_block(b,  a4, 512)
        a3 = attention_gate(d4, s3, 128)
        d3 = decoder_block(d4, a3, 256)
        a2 = attention_gate(d3, s2, 64)
        d2 = decoder_block(d3, a2, 128)
        a1 = attention_gate(d2, s1, 32)
        d1 = decoder_block(d2, a1, 64)
    else:
        d4 = decoder_block(b,  s4, 512)
        d3 = decoder_block(d4, s3, 256)
        d2 = decoder_block(d3, s2, 128)
        d1 = decoder_block(d2, s1, 64)

    # ── Output head ──
    outputs = layers.Conv2D(n_classes, (1, 1),
                            activation='softmax',
                            name='segmentation_head')(d1)

    model = Model(inputs, outputs, name='TRINETRA_UNet_Attention')
    total_params = model.count_params()
    print(f"[UNet] Parameters: {total_params:,}")
    return model


# ── U-Net with ResNet50 Encoder ───────────────────────────────────────────────
def build_resnet_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
                      n_classes: int = N_CLASSES) -> Model:
    """
    U-Net with ImageNet pre-trained ResNet50 encoder.
    NOTE: ResNet50 expects 3-channel input.
    Use 3 most informative bands (e.g. RGB) or compress with 1×1 conv.
    """
    # Pre-processing: compress N_CHANNELS → 3 via learned 1×1 projection
    raw_in    = layers.Input(shape=(PATCH_SIZE, PATCH_SIZE, N_CHANNELS))
    projected = layers.Conv2D(3, (1,1), padding='same',
                               activation='relu', name='band_projection')(raw_in)

    base  = ResNet50(include_top=False, weights='imagenet', input_tensor=projected)

    # ResNet50 skip connection layers
    s1 = base.get_layer('conv1_relu').output          # 128×128×64
    s2 = base.get_layer('conv2_block3_1_relu').output # 64×64×64
    s3 = base.get_layer('conv3_block4_1_relu').output # 32×32×128
    s4 = base.get_layer('conv4_block6_1_relu').output # 16×16×256
    b  = base.get_layer('conv5_block3_out').output    # 8×8×2048

    d4 = decoder_block(b,  s4, 512)
    d3 = decoder_block(d4, s3, 256)
    d2 = decoder_block(d3, s2, 128)
    d1 = decoder_block(d2, s1, 64)

    outputs = layers.Conv2D(n_classes, (1,1), activation='softmax')(d1)
    model = Model(raw_in, outputs, name='TRINETRA_ResNet50UNet')
    print(f"[ResNetUNet] Parameters: {model.count_params():,}")
    return model


# ── Loss Functions ────────────────────────────────────────────────────────────
class HazardSegmentationLoss:
    """Combined Focal + Dice loss for heavily imbalanced hazard segmentation."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.5):
        self.gamma = gamma
        self.alpha = alpha   # weighting between focal and dice

    def focal_loss(self, y_true, y_pred):
        """
        Categorical focal loss.
        gamma=2.0 focuses training on hard-to-classify hazard pixels.
        """
        loss = keras.losses.CategoricalFocalCrossentropy(gamma=self.gamma)
        return loss(y_true, y_pred)

    def dice_loss(self, y_true, y_pred):
        """
        Dice loss — directly optimises pixel overlap (IoU-like).
        Robust to spatial class imbalance.
        """
        smooth = 1e-6
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        union        = tf.reduce_sum(y_true, axis=[1, 2]) + \
                       tf.reduce_sum(y_pred, axis=[1, 2])
        dice_coeff   = (2.0 * intersection + smooth) / (union + smooth)
        return 1.0 - tf.reduce_mean(dice_coeff)

    def tversky_loss(self, y_true, y_pred,
                     alpha: float = 0.7, beta: float = 0.3):
        """
        Tversky loss — asymmetric Dice that penalises false negatives more.
        alpha=0.7, beta=0.3: prioritises recall (critical for life-safety alerts).
        """
        smooth = 1e-6
        TP = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
        FP = tf.reduce_sum((1 - y_true) * y_pred, axis=[1, 2])
        FN = tf.reduce_sum(y_true * (1 - y_pred), axis=[1, 2])
        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        return 1.0 - tf.reduce_mean(tversky)

    def combined_loss(self, y_true, y_pred):
        """Primary training loss: Focal + Dice (50/50 blend)."""
        return (self.alpha * self.focal_loss(y_true, y_pred) +
                (1 - self.alpha) * self.dice_loss(y_true, y_pred))

    def __call__(self, y_true, y_pred):
        return self.combined_loss(y_true, y_pred)


# ── Custom Metrics ────────────────────────────────────────────────────────────
class PerClassIoU(keras.metrics.Metric):
    """Per-class IoU for monitoring fire / flood / landslide separately."""

    def __init__(self, class_id: int, name: str = None, **kwargs):
        super().__init__(name=name or f'iou_class_{class_id}', **kwargs)
        self.class_id = class_id
        self.intersection = self.add_weight('intersection', initializer='zeros')
        self.union        = self.add_weight('union',        initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(tf.argmax(y_pred, axis=-1) == self.class_id, tf.float32)
        y_true_bin = y_true[..., self.class_id]
        inter = tf.reduce_sum(y_true_bin * y_pred_bin)
        union = tf.reduce_sum(y_true_bin) + tf.reduce_sum(y_pred_bin) - inter
        self.intersection.assign_add(inter)
        self.union.assign_add(union)

    def result(self):
        return self.intersection / (self.union + 1e-7)

    def reset_state(self):
        self.intersection.assign(0)
        self.union.assign(0)


# ── Data Generator ────────────────────────────────────────────────────────────
class HazardDataGenerator(keras.utils.Sequence):
    """
    Keras data generator for large-scale geospatial patch loading.
    Loads patches lazily from disk to manage memory.
    """

    def __init__(self, patch_dir: str, label_dir: str,
                 batch_size: int = BATCH_SIZE,
                 augment: bool = True,
                 n_classes: int = N_CLASSES):
        self.patch_files  = sorted(Path(patch_dir).glob('*.npy'))
        self.label_files  = sorted(Path(label_dir).glob('*.npy'))
        self.batch_size   = batch_size
        self.augment      = augment
        self.n_classes    = n_classes
        assert len(self.patch_files) == len(self.label_files), \
            "Mismatch between patch and label counts"
        print(f"[Generator] {len(self.patch_files)} patches | "
              f"batch={batch_size} | augment={augment}")

    def __len__(self):
        return max(1, len(self.patch_files) // self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        batch_files = self.patch_files[start: start + self.batch_size]
        label_files = self.label_files[start: start + self.batch_size]

        X_batch = np.stack([np.load(f) for f in batch_files])
        y_raw   = np.stack([np.load(f) for f in label_files])
        # One-hot encode labels
        y_batch = np.eye(self.n_classes, dtype=np.float32)[y_raw]

        if self.augment:
            X_batch, y_batch = self._augment(X_batch, y_batch)

        return X_batch, y_batch

    def _augment(self, X, y):
        """Random flip + 90° rotation augmentation (preserves spatial structure)."""
        for i in range(len(X)):
            if np.random.rand() > 0.5:
                X[i] = np.fliplr(X[i]); y[i] = np.fliplr(y[i])
            if np.random.rand() > 0.5:
                X[i] = np.flipud(X[i]); y[i] = np.flipud(y[i])
            k = np.random.randint(0, 4)
            X[i] = np.rot90(X[i], k); y[i] = np.rot90(y[i], k)
        return X, y

    def on_epoch_end(self):
        """Shuffle file list after each epoch."""
        idx = np.random.permutation(len(self.patch_files))
        self.patch_files = [self.patch_files[i] for i in idx]
        self.label_files = [self.label_files[i] for i in idx]


# ── Trainer ───────────────────────────────────────────────────────────────────
class HazardSegmentationTrainer:
    """Trains U-Net with spatial cross-validation and full callback suite."""

    def __init__(self, model: Model,
                 checkpoint_dir: str = './checkpoints'):
        self.model          = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.loss_fn        = HazardSegmentationLoss(gamma=2.0, alpha=0.5)
        self.history        = None

    def compile(self):
        self.model.compile(
            optimizer = keras.optimizers.Adam(LEARNING_RATE,
                                               clipnorm=1.0,
                                               weight_decay=1e-5),
            loss      = self.loss_fn,
            metrics   = [
                keras.metrics.CategoricalAccuracy(name='acc'),
                keras.metrics.MeanIoU(num_classes=N_CLASSES, name='mean_iou'),
                PerClassIoU(1, name='iou_fire'),
                PerClassIoU(2, name='iou_flood'),
                PerClassIoU(3, name='iou_landslide'),
            ]
        )
        print("[Trainer] Model compiled ✓")

    def get_callbacks(self) -> list:
        return [
            callbacks.ModelCheckpoint(
                filepath         = str(self.checkpoint_dir / 'best_unet.keras'),
                monitor          = 'val_mean_iou',
                mode             = 'max',
                save_best_only   = True,
                save_weights_only= False,
                verbose          = 1
            ),
            callbacks.EarlyStopping(
                monitor              = 'val_loss',
                patience             = 20,
                restore_best_weights = True,
                verbose              = 1
            ),
            callbacks.ReduceLROnPlateau(
                monitor  = 'val_loss',
                factor   = 0.5,
                patience = 8,
                min_lr   = 1e-7,
                verbose  = 1
            ),
            callbacks.TensorBoard(
                log_dir          = './logs/unet',
                histogram_freq   = 0,
                update_freq      = 'epoch'
            ),
            callbacks.CSVLogger('./logs/unet_training.csv'),
        ]

    def train(self, train_gen: HazardDataGenerator,
              val_gen: HazardDataGenerator) -> keras.callbacks.History:
        print(f"\n[Trainer] Starting training: {EPOCHS} epochs")
        self.history = self.model.fit(
            train_gen,
            validation_data = val_gen,
            epochs          = EPOCHS,
            callbacks       = self.get_callbacks(),
            verbose         = 1
        )
        return self.history

    def train_on_arrays(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray):
        """Train directly on numpy arrays (small datasets)."""
        y_train_oh = np.eye(N_CLASSES, dtype=np.float32)[y_train]
        y_val_oh   = np.eye(N_CLASSES, dtype=np.float32)[y_val]

        self.history = self.model.fit(
            X_train, y_train_oh,
            validation_data = (X_val, y_val_oh),
            epochs          = EPOCHS,
            batch_size      = BATCH_SIZE,
            callbacks       = self.get_callbacks(),
            verbose         = 1
        )
        return self.history

    def predict_scene(self, patches: np.ndarray) -> np.ndarray:
        """
        Run inference on patch array.
        patches: (N, H, W, F)
        Returns: (N, H, W, n_classes) probability maps
        """
        probs = self.model.predict(patches, batch_size=8, verbose=1)
        return probs  # shape: (N, 256, 256, 4)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate model and return per-class metrics."""
        y_test_oh = np.eye(N_CLASSES, dtype=np.float32)[y_test]
        results   = self.model.evaluate(X_test, y_test_oh, batch_size=8, verbose=1)
        metric_dict = dict(zip(self.model.metrics_names, results))
        print("\n[Evaluation Results]")
        for k, v in metric_dict.items():
            print(f"  {k:25s}: {v:.4f}")
        return metric_dict

    def save_model(self, path: str = None):
        path = path or str(self.checkpoint_dir / 'final_unet.keras')
        self.model.save(path)
        print(f"[Trainer] Model saved → {path}")

    def load_model(self, path: str):
        self.model = keras.models.load_model(
            path,
            custom_objects={
                'combined_loss': HazardSegmentationLoss(),
                'PerClassIoU':   PerClassIoU
            }
        )
        print(f"[Trainer] Model loaded from {path}")

    def plot_training(self, save_path: str = './logs/training_curves.png'):
        """Plot loss and IoU curves from training history."""
        if self.history is None:
            print("[Trainer] No history to plot.")
            return
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].plot(self.history.history['loss'],     label='train_loss')
        axes[0].plot(self.history.history['val_loss'], label='val_loss')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')

        axes[1].plot(self.history.history['mean_iou'],     label='train_mIoU')
        axes[1].plot(self.history.history['val_mean_iou'], label='val_mIoU')
        axes[1].set_title('Mean IoU'); axes[1].legend(); axes[1].set_xlabel('Epoch')

        axes[2].plot(self.history.history.get('iou_fire', []),      label='Fire')
        axes[2].plot(self.history.history.get('iou_flood', []),     label='Flood')
        axes[2].plot(self.history.history.get('iou_landslide', []), label='Landslide')
        axes[2].set_title('Per-Class IoU'); axes[2].legend(); axes[2].set_xlabel('Epoch')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Trainer] Training curves saved → {save_path}")
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("TRINETRA — U-Net CNN")
    print("=" * 50)

    model = build_unet(
        input_shape   = (PATCH_SIZE, PATCH_SIZE, N_CHANNELS),
        n_classes     = N_CLASSES,
        use_attention = True
    )
    model.summary(line_length=100)

    trainer = HazardSegmentationTrainer(model)
    trainer.compile()

    # Quick smoke test with random data
    print("\n[SmokTest] Running forward pass on random data...")
    x_dummy = np.random.rand(2, PATCH_SIZE, PATCH_SIZE, N_CHANNELS).astype(np.float32)
    y_dummy = model.predict(x_dummy, verbose=0)
    print(f"  Input:  {x_dummy.shape}")
    print(f"  Output: {y_dummy.shape}  (expected: (2, 256, 256, 4))")
    print(f"  Prob sum per pixel: {y_dummy[0, 100, 100, :].sum():.4f} (should ≈ 1.0)")
    print("\n✓ U-Net model ready.")
