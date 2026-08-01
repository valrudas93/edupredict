# Arquitectura técnica — EduPredict

## Pipeline dual de machine learning

EduPredict implementa dos pipelines paralelos que se fusionan en una capa de decisión final.

---

## Pipeline 1 — CNN 1D sobre texto

**Entrada:** comentario cualitativo del estudiante (texto libre)

```
comentario (str)
    │
    ▼
Limpieza de texto
  lowercase · remove puntuación · strip whitespace
    │
    ▼
Tokenización + Padding
  vocab_size=300 · max_len=15 tokens
    │
    ▼
Embedding Layer (entrenable)
  input_dim=300 · output_dim=32
    │
    ▼
Conv1D paralelas (3 ramas)
  ├── filtros=64 · kernel=2 (bigramas)
  ├── filtros=64 · kernel=3 (trigramas)
  └── filtros=64 · kernel=4 (4-gramas)
    │
    ▼
GlobalMaxPooling1D (por rama)
    │
    ▼
Concatenate (192d)
    │
    ▼
Dense(64, relu) + L2(1e-4)
    │
    ▼
vector_cnn (64d)
```

**Técnicas anti-overfitting:** Dropout(0.4), L2 regularization, EarlyStopping(patience=6), ReduceLROnPlateau

---

## Pipeline 2 — Random Forest sobre datos numéricos

**Entrada:** puntajes de evaluación + variables de contexto

```
Features numéricos (7)
  puntaje_claridad · puntaje_metodologia · puntaje_evaluacion
  puntaje_promedio · numero_estudiantes · semestre_num · asignatura_enc
    │
    ▼
StandardScaler (ajustado solo en train)
    │
    ▼
Random Forest Classifier
  n_estimators=300 · max_depth=12
  min_samples_leaf=4 · class_weight='balanced'
    │
    ▼
Validación cruzada estratificada 5-fold
    │
    ▼
probabilidades_rf (3d: [P(En riesgo), P(Estable), P(Mejora)])
```

**Técnicas anti-overfitting:** max_depth limitado, min_samples_leaf, class_weight balanced, CV 5-fold

---

## Capa de fusión

```
vector_cnn (64d) + probabilidades_rf (3d)
    │
    ▼
Concatenate (67d)
    │
    ▼
Dense(32, relu) + L2(1e-4)
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(3, softmax)
    │
    ▼
[P(En riesgo), P(Estable), P(Mejora)]
    │
    ▼
argmax → predicción final
```

---

## Feature engineering

| Feature | Origen | Tipo |
|---|---|---|
| `puntaje_claridad` | Dataset crudo | Numérico continuo [1.0–5.0] |
| `puntaje_metodologia` | Dataset crudo | Numérico continuo [1.0–5.0] |
| `puntaje_evaluacion` | Dataset crudo | Numérico continuo [1.0–5.0] |
| `puntaje_promedio` | **Derivado** — media de los 3 puntajes | Numérico continuo |
| `numero_estudiantes` | Dataset crudo | Numérico entero [15–45] |
| `semestre_num` | **Derivado** — codificación ordinal del semestre | Entero [1–8] |
| `asignatura_enc` | **Derivado** — LabelEncoding de asignatura | Entero [0–6] |
| `comentario_clean` | **Derivado** — texto limpio para CNN | String |

---

## Decisiones de diseño

### Por qué embedding entrenable y no FastText/BERT
El vocabulario del dataset es pequeño (~20 frases únicas recurrentes). Un modelo preentrenado en corpus general agregaría ruido y costo computacional innecesario. El embedding entrenable desde cero es suficiente y más rápido.

### Por qué CNN 1D y no LSTM/GRU
Los comentarios son cortos (media 6.5 palabras) y los patrones relevantes son locales (bigramas como "no explica", "muy claro"). CNN 1D captura exactamente eso con menor costo que secuencias recurrentes.

### Por qué Random Forest y no XGBoost
RF es más robusto ante features poco informativos (numero_estudiantes tiene r=-0.03 con target). XGBoost tiende a sobreajustar features ruidosos en datasets pequeños. Además, RF es más interpretable con SHAP values por el equipo evaluador.

### Por qué no fusionar antes
Entrenar CNN y RF por separado primero (y medir su contribución individual con ablation study) es mejor práctica: permite detectar si alguno de los dos es redundante y justifica la complejidad de la fusión.

---

## Ablation study — resultados

| Modelo | Accuracy | F1-macro | AUC-ROC | Δ F1 vs RF |
|---|---|---|---|---|
| CNN 1D solo | 0.493 | 0.341 | 0.694 | -56.3% |
| Random Forest solo | 0.895 | 0.903 | 0.980 | baseline |
| **Fusión CNN + RF** | **0.920** | **0.924** | **0.983** | **+2.2%** |

La fusión mejora en **+2.2 puntos de F1-macro** sobre el mejor modelo individual, confirmando que los comentarios cualitativos aportan señal complementaria a los puntajes numéricos.

---

## Consideraciones éticas

- Datos anonimizados: `id_docente` es un código sin correspondencia a nombres reales.
- El modelo no toma decisiones automáticas sobre docentes. Produce alertas para revisión humana.
- Transparencia: SHAP values disponibles para explicar cada predicción.
- Sesgo potencial: los comentarios textuales pueden reflejar sesgos de los estudiantes evaluadores. Documentado en el informe técnico.
