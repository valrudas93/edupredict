"""
EduPredict — Preprocesamiento de datos.

Carga, limpieza, feature engineering y split estratificado del dataset
de evaluaciones docentes de la Universidad del Rosario.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore

from src.config import (
    CLASS_LABELS,
    CNN_MAX_LEN,
    CNN_VOCAB_SIZE,
    DATASET_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


# ── Tipos ─────────────────────────────────────────────────────────────────────
class ProcessedData(NamedTuple):
    """Contenedor de todos los artefactos del preprocesamiento."""

    X_text_train: np.ndarray
    X_text_test: np.ndarray
    X_num_train: np.ndarray
    X_num_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    tokenizer: Tokenizer
    scaler: StandardScaler
    label_encoder: LabelEncoder
    df: pd.DataFrame


# ── Funciones ─────────────────────────────────────────────────────────────────
def _clean_text(text: str) -> str:
    """Limpieza mínima de texto en español para tokenización."""
    text = text.lower().strip()
    # Eliminar puntuación excepto acentos y ñ
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_validate(path: str | None = None) -> pd.DataFrame:
    """
    Carga el dataset y valida su integridad básica.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si hay columnas faltantes o valores fuera de rango.
    """
    dataset_path = path or str(DATASET_PATH)
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")

    required_cols = {
        "id_docente",
        "asignatura",
        "semestre",
        "numero_estudiantes",
        "puntaje_claridad",
        "puntaje_metodologia",
        "puntaje_evaluacion",
        "comentario",
        "tendencia_desempeno",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")

    if df.isnull().any().any():
        raise ValueError("El dataset contiene valores nulos.")

    score_cols = ["puntaje_claridad", "puntaje_metodologia", "puntaje_evaluacion"]
    for col in score_cols:
        if df[col].min() < 1.0 or df[col].max() > 5.0:
            raise ValueError(f"'{col}' contiene valores fuera del rango [1, 5].")

    unknown_classes = set(df["tendencia_desempeno"].unique()) - set(CLASS_LABELS)
    if unknown_classes:
        raise ValueError(f"Clases desconocidas en tendencia_desempeno: {unknown_classes}")

    return df.copy()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering sobre el DataFrame crudo."""
    df = df.copy()

    # Puntaje promedio de las 3 dimensiones
    df["puntaje_promedio"] = df[
        ["puntaje_claridad", "puntaje_metodologia", "puntaje_evaluacion"]
    ].mean(axis=1)

    # Codificar semestre como entero ordinal
    sem_order = sorted(df["semestre"].unique())
    sem_map = {s: i + 1 for i, s in enumerate(sem_order)}
    df["semestre_num"] = df["semestre"].map(sem_map)

    # Label encoding de asignatura
    le_asig = LabelEncoder()
    df["asignatura_enc"] = le_asig.fit_transform(df["asignatura"])

    # Texto limpio para CNN
    df["comentario_clean"] = df["comentario"].map(_clean_text)

    return df


def build_preprocessors(
    df_train: pd.DataFrame,
) -> tuple[Tokenizer, StandardScaler, LabelEncoder]:
    """
    Ajusta tokenizer, scaler y label encoder sobre el conjunto de entrenamiento.
    Solo se llama con datos de train para evitar data leakage.
    """
    # Tokenizer de texto
    tokenizer = Tokenizer(num_words=CNN_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train["comentario_clean"].tolist())

    # Scaler de features numéricos
    scaler = StandardScaler()
    scaler.fit(df_train[NUMERIC_FEATURES].values)

    # Encoder de la variable objetivo
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_LABELS)  # Orden fijo para reproducibilidad

    return tokenizer, scaler, label_encoder


def text_to_sequences(
    texts: list[str],
    tokenizer: Tokenizer,
    max_len: int = CNN_MAX_LEN,
) -> np.ndarray:
    """Convierte textos a secuencias padded listas para la CNN."""
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")


def preprocess(path: str | None = None) -> ProcessedData:
    """
    Pipeline completo de preprocesamiento.

    Returns:
        ProcessedData con todos los artefactos necesarios para el entrenamiento.
    """
    df = load_and_validate(path)
    df = engineer_features(df)

    # Split estratificado 80/20
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["tendencia_desempeno"],
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    # Ajustar preprocessors solo con train
    tokenizer, scaler, label_encoder = build_preprocessors(df_train)

    # Transformar texto
    X_text_train = text_to_sequences(
        df_train["comentario_clean"].tolist(), tokenizer
    )
    X_text_test = text_to_sequences(
        df_test["comentario_clean"].tolist(), tokenizer
    )

    # Transformar numéricos
    X_num_train = scaler.transform(df_train[NUMERIC_FEATURES].values)
    X_num_test = scaler.transform(df_test[NUMERIC_FEATURES].values)

    # Transformar target
    y_train = label_encoder.transform(df_train["tendencia_desempeno"].values)
    y_test = label_encoder.transform(df_test["tendencia_desempeno"].values)

    return ProcessedData(
        X_text_train=X_text_train,
        X_text_test=X_text_test,
        X_num_train=X_num_train,
        X_num_test=X_num_test,
        y_train=y_train,
        y_test=y_test,
        tokenizer=tokenizer,
        scaler=scaler,
        label_encoder=label_encoder,
        df=df,
    )
