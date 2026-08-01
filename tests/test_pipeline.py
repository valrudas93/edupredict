"""
EduPredict — Tests unitarios del pipeline.

Cubre: carga de datos, preprocesamiento, shapes, rangos de métricas
y consistencia del modelo de fusión.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CLASS_LABELS, CNN_MAX_LEN, NUMERIC_FEATURES
from src.preprocessing import (
    _clean_text,
    build_preprocessors,
    engineer_features,
    load_and_validate,
    text_to_sequences,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Dataset real cargado desde disco."""
    return load_and_validate()


@pytest.fixture()
def engineered_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(raw_df)


@pytest.fixture()
def preprocessors(engineered_df: pd.DataFrame):
    return build_preprocessors(engineered_df)


# ── Tests de carga ─────────────────────────────────────────────────────────────
class TestLoadAndValidate:
    def test_shape(self, raw_df: pd.DataFrame) -> None:
        assert raw_df.shape[0] == 3000
        assert raw_df.shape[1] == 9

    def test_no_nulls(self, raw_df: pd.DataFrame) -> None:
        assert not raw_df.isnull().any().any()

    def test_no_duplicates(self, raw_df: pd.DataFrame) -> None:
        assert raw_df.duplicated().sum() == 0

    def test_score_range(self, raw_df: pd.DataFrame) -> None:
        for col in ["puntaje_claridad", "puntaje_metodologia", "puntaje_evaluacion"]:
            assert raw_df[col].min() >= 1.0
            assert raw_df[col].max() <= 5.0

    def test_known_classes(self, raw_df: pd.DataFrame) -> None:
        assert set(raw_df["tendencia_desempeno"].unique()) == set(CLASS_LABELS)

    def test_docentes(self, raw_df: pd.DataFrame) -> None:
        assert raw_df["id_docente"].nunique() == 50

    def test_semestres(self, raw_df: pd.DataFrame) -> None:
        assert raw_df["semestre"].nunique() == 8


# ── Tests de feature engineering ──────────────────────────────────────────────
class TestFeatureEngineering:
    def test_puntaje_promedio_range(self, engineered_df: pd.DataFrame) -> None:
        assert engineered_df["puntaje_promedio"].between(1.0, 5.0).all()

    def test_semestre_num_range(self, engineered_df: pd.DataFrame) -> None:
        assert engineered_df["semestre_num"].between(1, 8).all()

    def test_asignatura_enc_range(self, engineered_df: pd.DataFrame) -> None:
        assert engineered_df["asignatura_enc"].between(0, 6).all()

    def test_comentario_clean_lowercase(self, engineered_df: pd.DataFrame) -> None:
        # Verifica que no hay letras mayúsculas A-Z en los comentarios limpios
        has_upper = engineered_df["comentario_clean"].str.contains(r"[A-Z]", regex=True)
        assert not has_upper.any()

    def test_numeric_features_present(self, engineered_df: pd.DataFrame) -> None:
        for feat in NUMERIC_FEATURES:
            assert feat in engineered_df.columns


# ── Tests de texto ────────────────────────────────────────────────────────────
class TestTextPreprocessing:
    def test_clean_text_lowercase(self) -> None:
        result = _clean_text("EXCELENTE Profesor")
        assert result == result.lower()

    def test_clean_text_removes_punctuation(self) -> None:
        result = _clean_text("¡Excelente! muy bueno.")
        assert "!" not in result
        assert "." not in result

    def test_clean_text_strips_whitespace(self) -> None:
        result = _clean_text("  hola  mundo  ")
        assert result == result.strip()
        assert "  " not in result

    def test_sequences_shape(self, preprocessors, engineered_df: pd.DataFrame) -> None:
        tokenizer, _, _ = preprocessors
        texts = engineered_df["comentario_clean"].tolist()[:10]
        seqs = text_to_sequences(texts, tokenizer)
        assert seqs.shape == (10, CNN_MAX_LEN)

    def test_sequences_no_nan(self, preprocessors, engineered_df: pd.DataFrame) -> None:
        tokenizer, _, _ = preprocessors
        texts = engineered_df["comentario_clean"].tolist()[:50]
        seqs = text_to_sequences(texts, tokenizer)
        assert not np.isnan(seqs).any()


# ── Tests de scaler ────────────────────────────────────────────────────────────
class TestScaler:
    def test_scaler_mean_approx_zero(
        self, preprocessors, engineered_df: pd.DataFrame
    ) -> None:
        _, scaler, _ = preprocessors
        X = scaler.transform(engineered_df[NUMERIC_FEATURES].values)
        # Media aproximadamente 0 (tolerancia amplia por train/full split)
        assert np.abs(X.mean()) < 1.0

    def test_scaler_output_shape(
        self, preprocessors, engineered_df: pd.DataFrame
    ) -> None:
        _, scaler, _ = preprocessors
        X = scaler.transform(engineered_df[NUMERIC_FEATURES].values)
        assert X.shape == (len(engineered_df), len(NUMERIC_FEATURES))


# ── Tests de label encoder ────────────────────────────────────────────────────
class TestLabelEncoder:
    def test_classes_correct(self, preprocessors) -> None:
        _, _, le = preprocessors
        assert set(le.classes_) == set(CLASS_LABELS)

    def test_transform_inverse(self, preprocessors) -> None:
        _, _, le = preprocessors
        encoded = le.transform(CLASS_LABELS)
        decoded = le.inverse_transform(encoded)
        assert list(decoded) == CLASS_LABELS

    def test_num_classes(self, preprocessors) -> None:
        _, _, le = preprocessors
        assert len(le.classes_) == 3
