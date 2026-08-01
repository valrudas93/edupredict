"""
EduPredict — Pipeline de entrenamiento completo.

Ejecuta:
  1. Preprocesamiento
  2. CNN 1D (Pipeline texto)
  3. Random Forest (Pipeline numérico) con CV 5-fold
  4. Modelo de fusión (Pipeline dual)
  5. Ablation study (comparación de los 3 modelos)
  6. Persistencia de artefactos

Uso:
    python -m src.train
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import tensorflow as tf

# Consola de Windows usa cp1252 por defecto y no puede imprimir los
# caracteres Unicode (─, ·, ✅) que usan los mensajes de progreso.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

from src.config import MODEL_DIR, OUTPUT_DIR, RANDOM_STATE
from src.models import (
    evaluate_model,
    get_cnn_representation,
    train_cnn,
    train_fusion,
    train_rf_with_cv,
)
from src.preprocessing import preprocess

# Reproducibilidad
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


def _print_metrics(metrics: dict) -> None:
    """Imprime métricas de forma legible."""
    name = metrics["model_name"]
    print(f"\n  ── {name} ──")
    print(f"     Accuracy:       {metrics['accuracy']:.4f}")
    print(f"     F1-macro:       {metrics['f1_macro']:.4f}")
    print(f"     AUC-ROC macro:  {metrics['auc_roc_macro']:.4f}")
    print("     F1 por clase:")
    for cls, f1 in metrics["f1_per_class"].items():
        print(f"       {cls:<12} {f1:.4f}")


def run_training(dataset_path: str | None = None) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Returns:
        Diccionario con métricas de los 3 modelos + ablation study.
    """
    print("=" * 60)
    print("  EduPredict — Entrenamiento de modelos")
    print("  Equipo Sin Convergencia · SIC 2025")
    print("=" * 60)

    # ── 1. Preprocesamiento ────────────────────────────────────────────────────
    print("\n[1/5] Preprocesando datos...")
    data = preprocess(dataset_path)
    print(f"      Train: {len(data.y_train)} | Test: {len(data.y_test)}")
    print(f"      Clases train: {dict(zip(*np.unique(data.y_train, return_counts=True), strict=False))}")

    # ── 2. CNN 1D ─────────────────────────────────────────────────────────────
    print("\n[2/5] Entrenando CNN 1D...")
    cnn_model, cnn_history = train_cnn(
        X_train=data.X_text_train,
        y_train=data.y_train,
        X_val=data.X_text_test,
        y_val=data.y_test,
    )
    cnn_model.predict(data.X_text_train, verbose=0)
    cnn_proba_test = cnn_model.predict(data.X_text_test, verbose=0)
    cnn_pred_test = np.argmax(cnn_proba_test, axis=1)

    metrics_cnn = evaluate_model(
        data.y_test, cnn_pred_test, cnn_proba_test, model_name="CNN 1D (solo texto)"
    )
    _print_metrics(metrics_cnn)

    # ── 3. Random Forest ──────────────────────────────────────────────────────
    print("\n[3/5] Entrenando Random Forest con validación cruzada...")
    rf_model, cv_results = train_rf_with_cv(data.X_num_train, data.y_train)

    print(f"      CV Accuracy: {cv_results['cv_accuracy_mean']:.4f}"
          f" ± {cv_results['cv_accuracy_std']:.4f}")
    print(f"      CV F1-macro: {cv_results['cv_f1_macro_mean']:.4f}"
          f" ± {cv_results['cv_f1_macro_std']:.4f}")

    rf_proba_train = rf_model.predict_proba(data.X_num_train)
    rf_proba_test = rf_model.predict_proba(data.X_num_test)
    rf_pred_test = rf_model.predict(data.X_num_test)

    metrics_rf = evaluate_model(
        data.y_test, rf_pred_test, rf_proba_test,
        model_name="Random Forest (solo numérico)"
    )
    _print_metrics(metrics_rf)

    # ── 4. Modelo de fusión ───────────────────────────────────────────────────
    print("\n[4/5] Entrenando modelo de fusión (CNN + RF)...")
    repr_train = get_cnn_representation(cnn_model, data.X_text_train)
    repr_test = get_cnn_representation(cnn_model, data.X_text_test)

    fusion_model, fusion_history = train_fusion(
        X_repr_train=repr_train,
        X_proba_rf_train=rf_proba_train,
        y_train=data.y_train,
        X_repr_val=repr_test,
        X_proba_rf_val=rf_proba_test,
        y_val=data.y_test,
    )

    X_fusion_test = np.concatenate([repr_test, rf_proba_test], axis=1)
    fusion_proba_test = fusion_model.predict(X_fusion_test, verbose=0)
    fusion_pred_test = np.argmax(fusion_proba_test, axis=1)

    metrics_fusion = evaluate_model(
        data.y_test, fusion_pred_test, fusion_proba_test,
        model_name="Fusión CNN + RF"
    )
    _print_metrics(metrics_fusion)

    # ── 5. Ablation study ────────────────────────────────────────────────────
    print("\n[5/5] Ablation study:")
    print(f"  {'Modelo':<30} {'Accuracy':>9} {'F1-macro':>9} {'AUC-ROC':>9}")
    print("  " + "-" * 60)
    for m in [metrics_cnn, metrics_rf, metrics_fusion]:
        print(
            f"  {m['model_name']:<30} "
            f"{m['accuracy']:>9.4f} "
            f"{m['f1_macro']:>9.4f} "
            f"{m['auc_roc_macro']:>9.4f}"
        )

    best = max(
        [metrics_cnn, metrics_rf, metrics_fusion], key=lambda x: x["f1_macro"]
    )
    print(f"\n  ✅ Mejor modelo: {best['model_name']} (F1-macro={best['f1_macro']:.4f})")

    # ── Persistencia ──────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cnn_model.save(MODEL_DIR / "cnn_model.keras")
    fusion_model.save(MODEL_DIR / "fusion_model.keras")

    with open(MODEL_DIR / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    with open(MODEL_DIR / "tokenizer.pkl", "wb") as f:
        pickle.dump(data.tokenizer, f)
    with open(MODEL_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(data.scaler, f)
    with open(MODEL_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(data.label_encoder, f)

    results = {
        "metrics_cnn": metrics_cnn,
        "metrics_rf": metrics_rf,
        "metrics_fusion": metrics_fusion,
        "cv_results_rf": cv_results,
        "cnn_history": {
            k: [float(v) for v in vals] for k, vals in cnn_history.items()
        },
        "fusion_history": {
            k: [float(v) for v in vals] for k, vals in fusion_history.items()
        },
    }

    results_path = OUTPUT_DIR / "training_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n  Artefactos guardados en: {MODEL_DIR}")
    print(f"  Resultados en:           {results_path}")
    print("\n" + "=" * 60)

    return results


if __name__ == "__main__":
    run_training()
