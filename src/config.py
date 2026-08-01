"""
EduPredict — Configuración global del proyecto.

Paleta basada en el Manual de Marca oficial de la Universidad del Rosario 2020.
Rojo Mutisia Clematis (#DA0921) como color principal institucional.
"""

from pathlib import Path

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

DATASET_PATH = DATA_DIR / "evaluaciones_docentes.csv"

# ── Paleta Universidad del Rosario ────────────────────────────────────────────
# Fuente: Manual de Marca UR 2020 (Pantone 7621 / DA0921)
UR_RED = "#DA0921"          # Rojo Mutisia Clematis — color institucional principal
UR_DARK_NAVY = "#242839"    # Azul muy oscuro — complementario principal
UR_BLUE = "#3100A0"         # Azul profundo — complementario secundario
UR_TECH = "#0E6A8C"         # Azul tecnología — Escuela Ing. Ciencia y Tecnología
UR_DARK_TEXT = "#202020"    # Casi negro para texto
UR_LIGHT_BG = "#F7F4F4"     # Fondo cálido muy claro (tono rosarista)
UR_WHITE = "#FFFFFF"
UR_GRAY = "#6B6B6B"
UR_LIGHT_GRAY = "#EBEBEB"

# Semáforo de riesgo (alineado con la paleta UR)
RISK_RED = "#DA0921"        # En riesgo
RISK_AMBER = "#E8670C"      # Advertencia (naranja facultad Economía UR)
RISK_GREEN = "#1A6E3A"      # Mejora / bueno

# Clases del modelo
CLASS_LABELS = ["En riesgo", "Estable", "Mejora"]
CLASS_COLORS = {
    "En riesgo": RISK_RED,
    "Estable": UR_TECH,
    "Mejora": RISK_GREEN,
}

# ── Hiperparámetros del modelo ─────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# CNN 1D
CNN_MAX_LEN = 15        # Longitud máxima de secuencia (tokens)
CNN_VOCAB_SIZE = 300    # Tamaño del vocabulario
CNN_EMBED_DIM = 32      # Dimensión del embedding
CNN_FILTERS = 64        # Filtros por tamaño de kernel
CNN_KERNELS = (2, 3, 4) # Tamaños de kernel (bigramas, trigramas, 4-gramas)
CNN_DROPOUT = 0.4       # Dropout para regularización
CNN_EPOCHS = 30
CNN_BATCH_SIZE = 64
CNN_PATIENCE = 6        # Early stopping

# Random Forest
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 12
RF_MIN_SAMPLES_LEAF = 4
RF_CLASS_WEIGHT = "balanced"

# Features numéricos para Random Forest
NUMERIC_FEATURES = [
    "puntaje_claridad",
    "puntaje_metodologia",
    "puntaje_evaluacion",
    "puntaje_promedio",
    "numero_estudiantes",
    "semestre_num",
    "asignatura_enc",
]
