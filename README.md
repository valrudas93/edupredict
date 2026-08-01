# EduPredict 🎓

> **Sistema predictivo de riesgo docente** para la Universidad del Rosario  
> Samsung Innovation Campus 2025 · Reto 4 · Analítica Educativa

[![Python](https://img.shields.io/badge/Python-3.10+-DA0921?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-242839?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-242839?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Ruff](https://img.shields.io/badge/linter-ruff-DA0921?style=flat-square)](https://docs.astral.sh/ruff)
[![Tests](https://img.shields.io/badge/tests-22%20passed-1A6E3A?style=flat-square&logo=pytest)](tests/)
[![License](https://img.shields.io/badge/license-MIT-3100A0?style=flat-square)](LICENSE)

---

## ¿Qué es EduPredict?

EduPredict evoluciona el sistema de evaluación docente de la Universidad del Rosario de un enfoque basado en reglas estáticas a un **modelo dual de machine learning** que anticipa la tendencia de desempeño docente antes de que impacte a los estudiantes.

El sistema clasifica cada evaluación en tres categorías: **En riesgo**, **Estable** o **Mejora**, combinando dos pipelines complementarios mediante una capa de fusión.

---

## Arquitectura

```
evaluaciones_docentes.csv
         │
         ├─── comentario ──────────► CNN 1D (TF/Keras)
         │    (texto libre)              │
         │                         vector representación (64d)
         │                              │
         └─── puntajes numéricos ──► Random Forest         
              + contexto               │
                                  probabilidades clase (3d)
                                       │
                              ┌────────┴────────┐
                              │  Capa de Fusión │
                              │  Dense(32) + DO │
                              └────────┬────────┘
                                       │
                          [En riesgo | Estable | Mejora]
```

### Métricas del modelo entrenado

| Modelo | Accuracy | F1-macro | AUC-ROC |
|---|---|---|---|
| CNN 1D (solo texto) | 0.493 | 0.341 | 0.694 |
| Random Forest (solo numérico) | 0.895 | 0.903 | 0.980 |
| **Fusión CNN + RF** ✅ | **0.920** | **0.924** | **0.983** |

> Ablation study completo disponible en [`notebooks/03_fusion_ablation.ipynb`](notebooks/03_fusion_ablation.ipynb)

---

## Estructura del repositorio

```
edupredict/
├── notebooks/                    # Notebooks por fase (uno por integrante)
│   ├── 00_eda.ipynb             # EDA exhaustivo (Angela)
│   ├── 01_preprocessing.ipynb   # Preprocesamiento y features (Johan)
│   ├── 02_cnn1d.ipynb           # Pipeline CNN 1D (Valeria)
│   ├── 03_random_forest.ipynb   # Pipeline Random Forest + CV (Johan)
│   ├── 04_fusion_ablation.ipynb # Fusión y ablation study (Isaac)
│   └── 05_resultados.ipynb      # Métricas finales y visualizaciones (Isaac)
│
├── src/                          # Paquete Python del proyecto
│   ├── __init__.py
│   ├── config.py                # Configuración, paleta UR, hiperparámetros
│   ├── preprocessing.py         # Carga, validación, feature engineering
│   ├── models.py                # CNN 1D, Random Forest, modelo de fusión
│   └── train.py                 # Pipeline de entrenamiento completo
│
├── app/
│   └── app.py                   # Dashboard Streamlit (identidad UR)
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py         # 22 tests unitarios
│
├── data/
│   └── evaluaciones_docentes.csv
│
├── models/                       # Artefactos del modelo (generados por train)
│   ├── cnn_model.keras
│   ├── fusion_model.keras
│   ├── rf_model.pkl
│   ├── tokenizer.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   └── training_results.json    # Métricas y curvas de entrenamiento
│
├── docs/
│   └── arquitectura.md          # Documentación técnica
│
├── .github/
│   └── workflows/
│       └── ci.yml               # CI: tests + ruff en cada push
│
├── pyproject.toml               # Configuración ruff + pytest
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Inicio rápido

### 1. Clonar e instalar

```bash
git clone https://github.com/sinconvergencia/edupredict.git
cd edupredict
pip install -r requirements.txt
```

### 2. Entrenar el modelo

```bash
python -m src.train
```

Salida esperada:
```
[1/5] Preprocesando datos...      Train: 2400 | Test: 600
[2/5] Entrenando CNN 1D...
[3/5] Entrenando Random Forest... CV F1-macro: 0.9100 ± 0.0071
[4/5] Entrenando modelo de fusión...
[5/5] Ablation study:
  Fusión CNN + RF    Accuracy: 0.9200    F1-macro: 0.9244    AUC-ROC: 0.9830
```

### 3. Lanzar el dashboard

```bash
streamlit run app/app.py
```

### 4. Correr tests

```bash
pytest tests/ -v
```

### 5. Linter

```bash
ruff check src/ tests/ app/
```

---

## Docker

### Levantar el dashboard

```bash
docker compose up -d edupredict
```

Si `models/` está vacío, el contenedor entrena automáticamente antes de arrancar
Streamlit (mismo pipeline que `python -m src.train`). Si ya existen modelos
entrenados en el host, se montan por volumen y el dashboard arranca directo.

Dashboard disponible en [http://localhost:8501](http://localhost:8501).

### Entrenar sin levantar el dashboard

```bash
docker compose run --rm train
```

### Correr los tests dentro del contenedor

```bash
docker build -t edupredict .
docker run --rm edupredict test
```

Los artefactos (`models/`, `outputs/`) se persisten en el host vía volúmenes,
por lo que sobreviven a reconstrucciones de la imagen.

---

## Dataset

| Campo | Descripción |
|---|---|
| `id_docente` | Identificador anonimizado del docente |
| `asignatura` | Materia evaluada (7 asignaturas) |
| `semestre` | Período académico (2020-1 → 2023-2) |
| `numero_estudiantes` | Tamaño del grupo |
| `puntaje_claridad` | Puntaje de claridad (1.0–5.0) |
| `puntaje_metodologia` | Puntaje de metodología (1.0–5.0) |
| `puntaje_evaluacion` | Puntaje de evaluación (1.0–5.0) |
| `comentario` | Comentario cualitativo del estudiante |
| `tendencia_desempeno` | **Variable objetivo**: Estable / Mejora / En riesgo |

3.000 registros · 50 docentes · 8 semestres · distribución 50/30/20

---

## Anti-overfitting

- Dropout 0.4 (CNN) y 0.3 (Fusión)
- EarlyStopping con restauración de mejores pesos
- ReduceLROnPlateau
- Regularización L2 en capas densas
- `max_depth=12` y `min_samples_leaf=4` en Random Forest
- `class_weight='balanced'` para clases desbalanceadas
- Validación cruzada estratificada 5-fold
- Split estratificado 80/20

---

## Equipo Sin Convergencia

| Integrante | Rol | Notebooks |
|---|---|---|
| **Valeria Rudas Ruiz** *(líder)* | Arquitectura, CNN 1D, integración | `02_cnn1d.ipynb` |
| **Johan A. Vera Lozano** | Preprocesamiento, Random Forest | `01_preprocessing.ipynb`, `03_random_forest.ipynb` |
| **Angela Y. Quiñones Martinez** | EDA, visualizaciones, informe técnico | `00_eda.ipynb` |
| **Isaac Oviedo** | Fusión, métricas, dashboard, video | `04_fusion_ablation.ipynb`, `05_resultados.ipynb` |

---

## Cronograma

| Semana | Fechas | Entregable |
|---|---|---|
| S1 | 13–20 jul | EDA + preprocesamiento |
| S2 | 21–27 jul | CNN 1D + Random Forest |
| S3 | 28 jul–1 ago | Fusión + ablation study |
| S4 | 2–4 ago | Dashboard + informe técnico + video |
| — | **4 ago** | **Entrega reportes finales** |
| — | 10 ago | Evaluación panel de expertos |
| — | 28 ago | Certificación SIC 2025 |

---

## Ética y privacidad

Los datos de evaluación docente son tratados con anonimización completa. No se almacenan ni procesan identificadores personales. El modelo es transparente mediante SHAP values (ver `04_fusion_ablation.ipynb`).

---

*Universidad del Rosario · Escuela de Administración · Samsung Innovation Campus 2025*
