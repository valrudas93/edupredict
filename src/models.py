"""
EduPredict — Definición y entrenamiento de modelos.

Pipeline dual:
  - Pipeline 1: CNN 1D sobre comentarios cualitativos (TensorFlow / Keras)
  - Pipeline 2: Random Forest sobre puntajes numéricos (scikit-learn)
  - Fusión: concatenación de representaciones + clasificador final

Técnicas anti-overfitting aplicadas:
  - Dropout en CNN (0.4)
  - EarlyStopping con restauración de mejores pesos
  - class_weight='balanced' en Random Forest
  - Validación cruzada estratificada 5-fold
  - max_depth limitado en RF para regularización implícita
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import label_binarize
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    Input,
)
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from tensorflow.keras.regularizers import l2  # type: ignore

from src.config import (
    CLASS_LABELS,
    CNN_BATCH_SIZE,
    CNN_DROPOUT,
    CNN_EMBED_DIM,
    CNN_EPOCHS,
    CNN_FILTERS,
    CNN_KERNELS,
    CNN_MAX_LEN,
    CNN_PATIENCE,
    CNN_VOCAB_SIZE,
    CV_FOLDS,
    RANDOM_STATE,
    RF_CLASS_WEIGHT,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_ESTIMATORS,
)


# ── CNN 1D ────────────────────────────────────────────────────────────────────
def build_cnn_model(num_classes: int = 3) -> Model:
    """
    CNN 1D con múltiples tamaños de kernel para captura de n-gramas.

    Arquitectura:
        Embedding → [Conv1D(k=2), Conv1D(k=3), Conv1D(k=4)]
        → GlobalMaxPooling1D por rama → Concatenate
        → Dense(64, relu) → Dropout(0.4) → Dense(num_classes, softmax)

    El multi-kernel permite capturar bigramas, trigramas y 4-gramas
    simultáneamente, similar a un clasificador de texto TextCNN clásico.
    """
    text_input = Input(shape=(CNN_MAX_LEN,), name="text_input")

    embedding = Embedding(
        input_dim=CNN_VOCAB_SIZE,
        output_dim=CNN_EMBED_DIM,
        input_length=CNN_MAX_LEN,
        name="embedding",
    )(text_input)

    # Ramas paralelas con diferentes tamaños de kernel
    branches = []
    for k in CNN_KERNELS:
        conv = Conv1D(
            filters=CNN_FILTERS,
            kernel_size=k,
            activation="relu",
            kernel_regularizer=l2(1e-4),
            name=f"conv1d_k{k}",
        )(embedding)
        pool = GlobalMaxPooling1D(name=f"pool_k{k}")(conv)
        branches.append(pool)

    merged = Concatenate(name="merge")(branches)

    x = Dense(64, activation="relu", kernel_regularizer=l2(1e-4), name="dense_repr")(
        merged
    )
    x = Dropout(CNN_DROPOUT, name="dropout")(x)
    output = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(inputs=text_input, outputs=output, name="EduPredict_CNN1D")
    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 3,
) -> tuple[Model, dict]:
    """
    Entrena la CNN 1D con early stopping y reduce-LR.

    Returns:
        (model, history.history)
    """
    model = build_cnn_model(num_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=CNN_PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=CNN_EPOCHS,
        batch_size=CNN_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    return model, history.history


def get_cnn_representation(
    model: Model,
    X: np.ndarray,
) -> np.ndarray:
    """Extrae el vector de representación de la capa dense_repr (antes de dropout)."""
    repr_model = Model(
        inputs=model.input,
        outputs=model.get_layer("dense_repr").output,
        name="repr_extractor",
    )
    return repr_model.predict(X, verbose=0)


# ── Random Forest ─────────────────────────────────────────────────────────────
def build_rf() -> RandomForestClassifier:
    """Random Forest con hiperparámetros controlados para evitar overfitting."""
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train_rf_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[RandomForestClassifier, dict]:
    """
    Entrena Random Forest con validación cruzada estratificada 5-fold.

    Returns:
        (model_fitted, cv_results_dict)
    """
    rf = build_rf()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_acc = cross_val_score(rf, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
    cv_f1 = cross_val_score(
        rf, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1
    )

    rf.fit(X_train, y_train)

    cv_results = {
        "cv_accuracy_mean": float(cv_acc.mean()),
        "cv_accuracy_std": float(cv_acc.std()),
        "cv_f1_macro_mean": float(cv_f1.mean()),
        "cv_f1_macro_std": float(cv_f1.std()),
        "cv_accuracy_per_fold": cv_acc.tolist(),
        "cv_f1_per_fold": cv_f1.tolist(),
    }
    return rf, cv_results


# ── Modelo fusionado ──────────────────────────────────────────────────────────
def build_fusion_model(repr_dim: int = 64, num_classes: int = 3) -> Model:
    """
    Clasificador de fusión: toma la concatenación [repr_cnn(64) + proba_rf(3)]
    y produce la predicción final.

    Input: vector de dimensión repr_dim + num_classes
    """
    fusion_input = Input(shape=(repr_dim + num_classes,), name="fusion_input")
    x = Dense(32, activation="relu", kernel_regularizer=l2(1e-4), name="fusion_dense")(
        fusion_input
    )
    x = Dropout(0.3, name="fusion_dropout")(x)
    output = Dense(num_classes, activation="softmax", name="fusion_output")(x)

    model = Model(inputs=fusion_input, outputs=output, name="EduPredict_Fusion")
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_fusion(
    X_repr_train: np.ndarray,
    X_proba_rf_train: np.ndarray,
    y_train: np.ndarray,
    X_repr_val: np.ndarray,
    X_proba_rf_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Model, dict]:
    """
    Entrena el clasificador de fusión sobre la concatenación de representaciones.
    """
    X_fusion_train = np.concatenate([X_repr_train, X_proba_rf_train], axis=1)
    X_fusion_val = np.concatenate([X_repr_val, X_proba_rf_val], axis=1)

    repr_dim = X_repr_train.shape[1]
    num_classes = X_proba_rf_train.shape[1]

    model = build_fusion_model(repr_dim, num_classes)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=0,
        ),
    ]

    history = model.fit(
        X_fusion_train,
        y_train,
        validation_data=(X_fusion_val, y_val),
        epochs=60,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )
    return model, history.history


# ── Métricas ──────────────────────────────────────────────────────────────────
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str = "Modelo",
) -> dict:
    """
    Calcula métricas completas: accuracy, F1-macro, AUC-ROC macro.

    Args:
        y_true: Etiquetas reales (encoded).
        y_pred: Predicciones de clase (encoded).
        y_proba: Probabilidades por clase (n_samples, n_classes).
        model_name: Nombre para el reporte.

    Returns:
        Diccionario con todas las métricas.
    """
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_LABELS,
        zero_division=0,
        output_dict=True,
    )

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "auc_roc_macro": round(auc, 4),
        "f1_per_class": {
            cls: round(float(f1_per_class[i]), 4)
            for i, cls in enumerate(CLASS_LABELS)
        },
        "classification_report": report,
    }
