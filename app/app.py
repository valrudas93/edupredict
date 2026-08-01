"""
EduPredict — Interfaz de usuario Streamlit.

Dashboard de predicción de riesgo docente para la Universidad del Rosario.
Identidad visual basada en el Manual de Marca UR 2020.
Rojo Mutisia Clematis (#DA0921) como color institucional principal.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Configuración de página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduPredict · Universidad del Rosario",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta Universidad del Rosario ────────────────────────────────────────────
UR_RED = "#DA0921"
UR_NAVY = "#242839"
UR_BLUE = "#3100A0"
UR_TECH = "#0E6A8C"
UR_LIGHT = "#F7F4F4"
UR_GRAY = "#6B6B6B"
RISK_GREEN = "#1A6E3A"

CLASS_LABELS = ["En riesgo", "Estable", "Mejora"]
CLASS_COLORS = {
    "En riesgo": UR_RED,
    "Estable": UR_TECH,
    "Mejora": RISK_GREEN,
}
CLASS_ICONS = {"En riesgo": "🔴", "Estable": "🔵", "Mejora": "🟢"}

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        background-color: {UR_LIGHT};
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {UR_NAVY};
        border-right: 3px solid {UR_RED};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E8E8E8 !important;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: #FFFFFF !important;
    }}

    /* Header strip */
    .ur-header {{
        background: linear-gradient(135deg, {UR_NAVY} 0%, {UR_BLUE} 100%);
        border-left: 6px solid {UR_RED};
        border-radius: 0 8px 8px 0;
        padding: 20px 28px;
        margin-bottom: 24px;
    }}
    .ur-header h1 {{
        color: #FFFFFF;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    .ur-header p {{
        color: #C8D0E0;
        font-size: 0.9rem;
        margin: 4px 0 0 0;
    }}
    .ur-badge {{
        background: {UR_RED};
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 12px;
        letter-spacing: 1px;
        text-transform: uppercase;
        display: inline-block;
        margin-bottom: 8px;
    }}

    /* Metric cards */
    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        border-top: 4px solid {UR_RED};
        box-shadow: 0 2px 8px rgba(36,40,57,0.08);
        text-align: center;
    }}
    .metric-card .value {{
        font-size: 2rem;
        font-weight: 700;
        color: {UR_NAVY};
        line-height: 1;
    }}
    .metric-card .label {{
        font-size: 0.78rem;
        color: {UR_GRAY};
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Risk badge */
    .risk-badge-riesgo {{
        background: {UR_RED};
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
    }}
    .risk-badge-estable {{
        background: {UR_TECH};
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
    }}
    .risk-badge-mejora {{
        background: {RISK_GREEN};
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: 700;
        display: inline-block;
    }}

    /* Section titles */
    .section-title {{
        font-size: 1rem;
        font-weight: 600;
        color: {UR_NAVY};
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid {UR_RED};
        padding-bottom: 6px;
        margin: 24px 0 16px 0;
    }}

    /* Prob bars */
    .prob-container {{
        background: #F0EEF0;
        border-radius: 6px;
        overflow: hidden;
        height: 10px;
        margin: 4px 0;
    }}

    /* Info box */
    .info-box {{
        background: white;
        border-left: 4px solid {UR_TECH};
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.88rem;
        color: {UR_NAVY};
    }}

    /* Footer */
    .ur-footer {{
        background: {UR_NAVY};
        color: #A0A8B8;
        font-size: 0.78rem;
        text-align: center;
        padding: 14px;
        border-top: 3px solid {UR_RED};
        border-radius: 8px 8px 0 0;
        margin-top: 40px;
    }}

    /* Hide streamlit default elements */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1rem; padding-bottom: 0; }}

    /* Stmetric override */
    [data-testid="stMetric"] {{
        background: white;
        border-radius: 10px;
        padding: 14px 18px;
        border-top: 3px solid {UR_RED};
        box-shadow: 0 2px 8px rgba(36,40,57,0.08);
    }}
    [data-testid="stMetric"] * {{
        color: {UR_NAVY} !important;
    }}
    [data-testid="stMetricLabel"] * {{
        color: {UR_GRAY} !important;
    }}

    /* Landing page — Hero */
    .hero {{
        background: linear-gradient(135deg, {UR_NAVY} 0%, {UR_BLUE} 100%);
        border-radius: 12px;
        padding: 36px 40px;
        margin-bottom: 8px;
    }}
    .hero h1 {{
        color: #FFFFFF;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 10px 0 14px 0;
        letter-spacing: -0.5px;
    }}
    .hero p {{
        color: #C8D0E0;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 760px;
        margin: 0;
    }}

    /* Landing page — problema / solución */
    .ps-card {{
        background: white;
        border-radius: 10px;
        padding: 20px 24px;
        height: 100%;
        box-shadow: 0 2px 8px rgba(36,40,57,0.08);
    }}
    .ps-card.problem {{ border-top: 4px solid {UR_RED}; }}
    .ps-card.solution {{ border-top: 4px solid {RISK_GREEN}; }}
    .ps-card h3 {{
        margin: 0 0 8px 0;
        color: {UR_NAVY};
        font-size: 0.95rem;
    }}
    .ps-card p {{
        color: {UR_NAVY};
        font-size: 0.9rem;
        line-height: 1.55;
        margin: 0;
    }}

    /* Landing page — pasos del pipeline */
    .step-card {{
        background: white;
        border-radius: 10px;
        padding: 20px 16px;
        height: 100%;
        text-align: center;
        box-shadow: 0 2px 8px rgba(36,40,57,0.08);
        border-top: 4px solid {UR_TECH};
    }}
    .step-card .icon {{ font-size: 1.9rem; }}
    .step-card h4 {{
        margin: 8px 0 6px 0;
        color: {UR_NAVY};
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .step-card p {{
        margin: 0;
        color: {UR_GRAY};
        font-size: 0.82rem;
        line-height: 1.45;
    }}
    .step-arrow {{
        text-align: center;
        font-size: 1.6rem;
        color: {UR_GRAY};
        margin-top: 44px;
    }}

    /* Barra de navegación superior (independiente del sidebar colapsable) */
    .topnav-wrap {{
        margin-bottom: 18px;
    }}
    .topnav-wrap .stButton button {{
        border-radius: 20px;
        font-size: 0.82rem;
        padding: 4px 4px;
    }}
    .stButton button[kind="primary"] {{
        background-color: {UR_RED};
        border-color: {UR_RED};
    }}
    .stButton button[kind="primary"]:hover {{
        background-color: #B5081B;
        border-color: #B5081B;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constantes ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "evaluaciones_docentes.csv"
RESULTS_PATH = ROOT / "outputs" / "training_results.json"

NUMERIC_FEATURES = [
    "puntaje_claridad",
    "puntaje_metodologia",
    "puntaje_evaluacion",
    "puntaje_promedio",
    "numero_estudiantes",
    "semestre_num",
    "asignatura_enc",
]

ASIGNATURAS = [
    "Bases de Datos",
    "Estructuras de Datos",
    "Ingeniería de Software",
    "Inteligencia Artificial",
    "Matemáticas Básicas",
    "Programación Orientada a Objetos",
    "Ética Profesional",
]

SEMESTRES = [
    "2020-1", "2020-2", "2021-1", "2021-2",
    "2022-1", "2022-2", "2023-1", "2023-2",
]


# ── Carga de artefactos ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Carga todos los artefactos del modelo entrenado."""
    try:
        import tensorflow as tf

        cnn = tf.keras.models.load_model(MODEL_DIR / "cnn_model.keras")
        fusion = tf.keras.models.load_model(MODEL_DIR / "fusion_model.keras")
        repr_model = tf.keras.Model(
            inputs=cnn.input,
            outputs=cnn.get_layer("dense_repr").output,
        )

        with open(MODEL_DIR / "rf_model.pkl", "rb") as f:
            rf = pickle.load(f)
        with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)

        return cnn, fusion, repr_model, rf, tokenizer, scaler, le, True
    except Exception as e:
        return None, None, None, None, None, None, None, str(e)


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


@st.cache_data(show_spinner=False)
def load_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            return json.load(f)
    return None


def predict_single(
    comentario: str,
    puntaje_claridad: float,
    puntaje_metodologia: float,
    puntaje_evaluacion: float,
    numero_estudiantes: int,
    semestre: str,
    asignatura: str,
    fusion, repr_model, rf, tokenizer, scaler, le,
) -> dict:
    """Realiza una predicción con el modelo de fusión."""
    import re

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # Preprocesar texto
    text = comentario.lower().strip()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    seq = pad_sequences(
        tokenizer.texts_to_sequences([text]),
        maxlen=15,
        padding="post",
        truncating="post",
    )

    # Features numéricos
    sem_map = {s: i + 1 for i, s in enumerate(SEMESTRES)}
    asig_map = {a: i for i, a in enumerate(ASIGNATURAS)}
    promedio = (puntaje_claridad + puntaje_metodologia + puntaje_evaluacion) / 3

    x_num = np.array([[
        puntaje_claridad,
        puntaje_metodologia,
        puntaje_evaluacion,
        promedio,
        numero_estudiantes,
        sem_map.get(semestre, 1),
        asig_map.get(asignatura, 0),
    ]])
    x_num_scaled = scaler.transform(x_num)

    # Representación CNN
    repr_vec = repr_model.predict(seq, verbose=0)

    # Probabilidades RF
    rf_proba = rf.predict_proba(x_num_scaled)

    # Fusión
    x_fusion = np.concatenate([repr_vec, rf_proba], axis=1)
    proba = fusion.predict(x_fusion, verbose=0)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = le.inverse_transform([pred_idx])[0]

    return {
        "clase": pred_class,
        "probabilidades": {cls: float(p) for cls, p in zip(CLASS_LABELS, proba, strict=False)},
        "promedio": round(promedio, 2),
        "confianza": float(proba[pred_idx]),
    }


def _go_to(target: str) -> None:
    """Callback para que los botones de la landing naveguen el radio del sidebar."""
    st.session_state.nav_page = target


# ── Sidebar ────────────────────────────────────────────────────────────────────
NAV_OPTIONS = [
    "🏠 Inicio",
    "🔍 Predicción",
    "📊 Dashboard EDA",
    "📈 Resultados del Modelo",
    "ℹ️ Acerca de",
]

with st.sidebar:
    st.markdown("### 🎓 EduPredict")
    st.markdown("**Universidad del Rosario**")
    st.markdown("Analítica Educativa · SIC 2025")
    st.markdown("---")

    page = st.radio(
        "Navegación",
        NAV_OPTIONS,
        key="nav_page",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.75rem; color:#8090A8; line-height:1.6;'>
        <b>Equipo Sin Convergencia</b><br>
        Valeria Rudas Ruiz<br>
        Johan A. Vera Lozano<br>
        Angela Y. Quiñones M.<br>
        Isaac Oviedo<br><br>
        <i>Samsung Innovation Campus 2025</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ur-header">
      <div class="ur-badge">SIC 2025 · Reto 4</div>
      <h1>EduPredict</h1>
      <p>Sistema predictivo de riesgo docente · Universidad del Rosario · Equipo Sin Convergencia</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Barra de navegación superior ─────────────────────────────────────────────
# Redundante con el radio del sidebar a propósito: si el usuario colapsa el
# sidebar, el navegador recuerda ese estado y Streamlit no ofrece forma de
# reabrirlo desde Python, así que esta barra queda como vía de navegación
# siempre visible en el cuerpo principal.
st.markdown('<div class="topnav-wrap">', unsafe_allow_html=True)
nav_cols = st.columns(len(NAV_OPTIONS))
for nav_col, nav_opt in zip(nav_cols, NAV_OPTIONS, strict=False):
    with nav_col:
        st.button(
            nav_opt,
            key=f"topnav_{nav_opt}",
            use_container_width=True,
            type="primary" if nav_opt == page else "secondary",
            on_click=_go_to,
            args=(nav_opt,),
        )
st.markdown("</div>", unsafe_allow_html=True)

# ── Cargar artefactos ─────────────────────────────────────────────────────────
cnn, fusion, repr_model, rf, tokenizer, scaler, le, status = load_artifacts()
models_ready = status is True

if not models_ready:
    st.warning(
        f"⚠️  Modelos no entrenados aún. Ejecuta `python -m src.train` primero.\n\n"
        f"Error: {status}",
        icon="⚠️",
    )

df = load_dataset()
results = load_results()


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 0: INICIO
# ══════════════════════════════════════════════════════════════════════════════
if "Inicio" in page:
    st.markdown(
        """
        <div class="hero">
            <div class="ur-badge">SIC 2025 · Reto 4 · Analítica Educativa</div>
            <h1>De reglas fijas a predicción anticipada</h1>
            <p>
                La Universidad del Rosario evaluaba el desempeño docente con un sistema de
                reglas estáticas: umbrales fijos sobre los puntajes que solo describen lo que
                ya ocurrió, sin aprovechar los comentarios de los estudiantes ni anticipar
                hacia dónde va la tendencia. <b>EduPredict</b> lo reemplaza por un modelo de
                aprendizaje automático que predice la tendencia de cada evaluación
                —<b> En riesgo</b>, <b>Estable</b> o <b>Mejora</b>— antes de que se convierta
                en un problema, y entrega una recomendación concreta para actuar.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">¿Cuál es el problema?</div>', unsafe_allow_html=True)
    col_prob, col_sol = st.columns(2, gap="large")
    with col_prob:
        st.markdown(
            """
            <div class="ps-card problem">
                <h3>🔴 Antes — sistema basado en reglas</h3>
                <p>
                Cada evaluación se clasificaba comparando puntajes contra umbrales fijos:
                un enfoque <b>descriptivo</b>, que solo dice qué pasó, no hacia dónde va un
                docente — y que descarta por completo el comentario cualitativo del
                estudiante, justo donde suele aparecer la señal más temprana de un problema.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_sol:
        st.markdown(
            """
            <div class="ps-card solution">
                <h3>🟢 Ahora — modelo predictivo dual</h3>
                <p>
                Un pipeline de <b>machine learning</b> combina el texto del comentario con
                los puntajes numéricos y el contexto (asignatura, semestre, tamaño de grupo)
                para anticipar la tendencia de desempeño y sugerir una acción concreta —
                antes de que el problema escale.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">¿Cómo funciona?</div>', unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Entrada", "Comentario del estudiante + puntajes numéricos y contexto "
                            "(asignatura, semestre, tamaño de grupo)"),
        ("2️⃣", "Dos modelos en paralelo", "CNN 1D interpreta el texto · Random Forest "
                                            "interpreta los puntajes"),
        ("3️⃣", "Fusión + predicción", "Se combinan ambas señales y se genera la tendencia "
                                        "final junto con una recomendación accionable"),
    ]
    flow_cols = st.columns([3, 0.6, 3, 0.6, 3])
    step_cols = [flow_cols[0], flow_cols[2], flow_cols[4]]
    for col, (icon, title, desc) in zip(step_cols, steps, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="step-card">
                    <div class="icon">{icon}</div>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with flow_cols[1]:
        st.markdown('<div class="step-arrow">→</div>', unsafe_allow_html=True)
    with flow_cols[3]:
        st.markdown('<div class="step-arrow">→</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Resultados del modelo de fusión</div>',
        unsafe_allow_html=True,
    )
    if results is not None:
        m = results["metrics_fusion"]
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Accuracy", f"{m['accuracy']:.1%}")
        with mc2:
            st.metric("F1-macro", f"{m['f1_macro']:.1%}")
        with mc3:
            st.metric("AUC-ROC", f"{m['auc_roc_macro']:.1%}")
    else:
        st.info("Entrena el modelo con `python -m src.train` para ver las métricas aquí.")

    st.markdown('<div class="section-title">Explora el sistema</div>', unsafe_allow_html=True)
    cta1, cta2, cta3 = st.columns(3)
    with cta1:
        st.button(
            "🔍 Probar el predictor →",
            key="cta_pred",
            type="primary",
            use_container_width=True,
            on_click=_go_to,
            args=("🔍 Predicción",),
        )
    with cta2:
        st.button(
            "📊 Ver el análisis de datos",
            key="cta_eda",
            use_container_width=True,
            on_click=_go_to,
            args=("📊 Dashboard EDA",),
        )
    with cta3:
        st.button(
            "📈 Ver resultados del modelo",
            key="cta_res",
            use_container_width=True,
            on_click=_go_to,
            args=("📈 Resultados del Modelo",),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1: PREDICCIÓN
# ══════════════════════════════════════════════════════════════════════════════
elif "Predicción" in page:
    st.markdown('<div class="section-title">Predicción Individual</div>', unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        with st.form("pred_form"):
            st.markdown("**Datos de la evaluación**")

            asignatura = st.selectbox("Asignatura", ASIGNATURAS)
            semestre = st.selectbox("Semestre", SEMESTRES, index=len(SEMESTRES) - 1)
            numero_estudiantes = st.slider("N° estudiantes", 15, 45, 30)

            st.markdown("**Puntajes de evaluación** (1.0 – 5.0)")
            c1, c2, c3 = st.columns(3)
            with c1:
                claridad = st.number_input("Claridad", 1.0, 5.0, 3.8, 0.1)
            with c2:
                metodologia = st.number_input("Metodología", 1.0, 5.0, 3.9, 0.1)
            with c3:
                evaluacion = st.number_input("Evaluación", 1.0, 5.0, 3.7, 0.1)

            comentario = st.text_area(
                "Comentario del estudiante",
                placeholder="Ej: La clase es muy dinámica y entretenida.",
                height=80,
            )

            submitted = st.form_submit_button(
                "Predecir tendencia →",
                use_container_width=True,
                type="primary",
            )

    with col_result:
        if submitted and models_ready and comentario.strip():
            with st.spinner("Analizando..."):
                pred = predict_single(
                    comentario=comentario,
                    puntaje_claridad=claridad,
                    puntaje_metodologia=metodologia,
                    puntaje_evaluacion=evaluacion,
                    numero_estudiantes=numero_estudiantes,
                    semestre=semestre,
                    asignatura=asignatura,
                    fusion=fusion, repr_model=repr_model, rf=rf,
                    tokenizer=tokenizer, scaler=scaler, le=le,
                )

            clase = pred["clase"]
            icon = CLASS_ICONS[clase]
            badge_class = {
                "En riesgo": "risk-badge-riesgo",
                "Estable": "risk-badge-estable",
                "Mejora": "risk-badge-mejora",
            }[clase]

            st.markdown("**Resultado de la predicción**")
            st.markdown(
                f'<div class="{badge_class}">{icon} {clase}</div>',
                unsafe_allow_html=True,
            )

            confianza = pred["confianza"]
            st.markdown(f"Confianza del modelo: **{confianza:.1%}**")
            st.progress(confianza)

            st.markdown("**Probabilidades por clase**")
            for cls, prob in sorted(
                pred["probabilidades"].items(), key=lambda x: -x[1]
            ):
                col_l, col_r = st.columns([3, 1])
                with col_l:
                    st.progress(prob, text=f"{CLASS_ICONS[cls]} {cls}")
                with col_r:
                    st.markdown(f"**{prob:.1%}**")

            st.markdown(f"Puntaje promedio calculado: **{pred['promedio']:.2f} / 5.0**")

            # Recomendaciones
            st.markdown("**Recomendaciones**")
            if clase == "En riesgo":
                st.error(
                    "🔴 Intervención sugerida: revisar metodología de evaluación, "
                    "solicitar mentoría académica y establecer plan de mejora en las "
                    "próximas 4 semanas."
                )
            elif clase == "Estable":
                st.info(
                    "🔵 Desempeño estable. Se recomienda mantener las prácticas "
                    "actuales e identificar oportunidades de innovación pedagógica."
                )
            else:
                st.success(
                    "🟢 Tendencia de mejora sostenida. Reconocer el desempeño y "
                    "compartir buenas prácticas con el resto del cuerpo docente."
                )

        elif submitted and not comentario.strip():
            st.warning("Por favor ingresa un comentario del estudiante.")
        elif submitted and not models_ready:
            st.error("Modelos no disponibles. Entrena el sistema primero.")
        else:
            st.markdown(
                '<div class="info-box">📝 Completa el formulario y presiona <b>Predecir</b> '
                "para obtener la tendencia de desempeño docente predicha por el modelo "
                "de fusión CNN 1D + Random Forest.</div>",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2: DASHBOARD EDA
# ══════════════════════════════════════════════════════════════════════════════
elif "EDA" in page:
    st.markdown('<div class="section-title">Análisis Exploratorio de Datos</div>', unsafe_allow_html=True)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total evaluaciones", f"{len(df):,}")
    with k2:
        st.metric("Docentes únicos", df["id_docente"].nunique())
    with k3:
        st.metric("Semestres", df["semestre"].nunique())
    with k4:
        pct_riesgo = (df["tendencia_desempeno"] == "En riesgo").mean()
        st.metric("% En riesgo", f"{pct_riesgo:.1%}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Distribuciones", "📈 Temporal", "💬 Comentarios"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribución de clases**")
            dist = df["tendencia_desempeno"].value_counts()
            chart_data = pd.DataFrame(
                {"Clase": dist.index, "Evaluaciones": dist.values}
            )
            st.bar_chart(chart_data.set_index("Clase"), color=UR_RED)

        with c2:
            st.markdown("**Puntaje promedio por clase**")
            df_temp = df.copy()
            df_temp["promedio"] = df_temp[
                ["puntaje_claridad", "puntaje_metodologia", "puntaje_evaluacion"]
            ].mean(axis=1)
            means = df_temp.groupby("tendencia_desempeno")["promedio"].mean().round(3)
            st.bar_chart(means, color=UR_TECH)

        st.markdown("**Puntajes medios por clase y dimensión**")
        score_by_class = df.groupby("tendencia_desempeno")[
            ["puntaje_claridad", "puntaje_metodologia", "puntaje_evaluacion"]
        ].mean().round(3)
        st.dataframe(score_by_class, use_container_width=True)

    with tab2:
        st.markdown("**Evolución de tendencias por semestre**")
        pivot = (
            df.groupby(["semestre", "tendencia_desempeno"])
            .size()
            .unstack(fill_value=0)
        )
        st.line_chart(pivot, color=[UR_RED, UR_TECH, RISK_GREEN])

        st.markdown("**Distribución por semestre (tabla)**")
        pivot["% En riesgo"] = (
            pivot["En riesgo"] / pivot.sum(axis=1) * 100
        ).round(1)
        st.dataframe(pivot, use_container_width=True)

    with tab3:
        st.markdown("**Top 10 comentarios más frecuentes**")
        top10 = df["comentario"].value_counts().head(10).reset_index()
        top10.columns = ["Comentario", "Frecuencia"]
        top10["% del total"] = (top10["Frecuencia"] / len(df) * 100).round(1)
        st.dataframe(top10, use_container_width=True)

        st.markdown("**Docentes con mayor % de evaluaciones en riesgo**")
        riesgo_doc = (
            df.groupby("id_docente")["tendencia_desempeno"]
            .apply(lambda x: (x == "En riesgo").mean() * 100)
            .sort_values(ascending=False)
            .head(10)
            .round(1)
            .reset_index()
        )
        riesgo_doc.columns = ["Docente", "% En riesgo"]
        st.dataframe(riesgo_doc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3: RESULTADOS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif "Resultados" in page:
    st.markdown('<div class="section-title">Resultados del Modelo</div>', unsafe_allow_html=True)

    if results is None:
        st.info("Entrena el modelo primero con `python -m src.train`.")
    else:
        # Comparación de modelos
        st.markdown("**Ablation Study — Comparación de modelos**")
        ablation_data = []
        for key, name in [
            ("metrics_cnn", "CNN 1D (texto)"),
            ("metrics_rf", "Random Forest (numérico)"),
            ("metrics_fusion", "Fusión CNN + RF"),
        ]:
            m = results[key]
            ablation_data.append({
                "Modelo": name,
                "Accuracy": m["accuracy"],
                "F1-macro": m["f1_macro"],
                "AUC-ROC": m["auc_roc_macro"],
            })

        abl_df = pd.DataFrame(ablation_data).set_index("Modelo")
        styled_abl = abl_df.style.apply(
            lambda col: [
                "color: white; font-weight: 700;" if v == col.max() else ""
                for v in col
            ],
            axis=0,
        )
        st.dataframe(styled_abl, use_container_width=True)

        st.bar_chart(abl_df[["F1-macro", "AUC-ROC"]])

        st.markdown("**F1-score por clase — Modelo de Fusión**")
        f1_fusion = results["metrics_fusion"]["f1_per_class"]
        f1_df = pd.DataFrame(
            {"Clase": list(f1_fusion.keys()), "F1-score": list(f1_fusion.values())}
        ).set_index("Clase")
        st.bar_chart(f1_df, color=UR_RED)

        # Validación cruzada RF
        st.markdown("**Validación cruzada Random Forest (5-fold)**")
        cv = results["cv_results_rf"]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("CV Accuracy (media)", f"{cv['cv_accuracy_mean']:.4f}")
        with c2:
            st.metric("CV Accuracy (std)", f"±{cv['cv_accuracy_std']:.4f}")
        with c3:
            st.metric("CV F1-macro (media)", f"{cv['cv_f1_macro_mean']:.4f}")
        with c4:
            st.metric("CV F1-macro (std)", f"±{cv['cv_f1_macro_std']:.4f}")

        # Curvas de entrenamiento CNN
        st.markdown("**Curvas de aprendizaje — CNN 1D**")
        h = results["cnn_history"]
        if "accuracy" in h:
            history_df = pd.DataFrame({
                "Train accuracy": h["accuracy"],
                "Val accuracy": h.get("val_accuracy", []),
            })
            st.line_chart(history_df)

        # Reporte de clasificación completo
        with st.expander("📋 Reporte de clasificación completo (Fusión)"):
            report = results["metrics_fusion"]["classification_report"]
            report_df = pd.DataFrame(report).T.round(3)
            st.dataframe(report_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4: ACERCA DE
# ══════════════════════════════════════════════════════════════════════════════
elif "Acerca" in page:
    st.markdown('<div class="section-title">Acerca del Proyecto</div>', unsafe_allow_html=True)

    st.markdown(
        """
        **EduPredict** es un sistema de analítica educativa predictiva desarrollado como
        proyecto Capstone del Samsung Innovation Campus 2025 para la Universidad del Rosario.

        ### Arquitectura del modelo
        El sistema implementa un pipeline dual de machine learning:

        **Pipeline 1 — CNN 1D (TensorFlow / Keras)**
        Procesa los comentarios cualitativos de los estudiantes mediante una red neuronal
        convolucional con múltiples tamaños de kernel (bigramas, trigramas, 4-gramas).
        Captura patrones semánticos locales asociados a cada nivel de desempeño.

        **Pipeline 2 — Random Forest (scikit-learn)**
        Clasifica usando los puntajes numéricos estructurados por dimensión
        (claridad, metodología, evaluación) con validación cruzada estratificada 5-fold
        y class_weight='balanced' para manejar el desbalance de clases.

        **Fusión final**
        Los vectores de representación de ambos modelos se concatenan y pasan por
        un clasificador denso con Dropout que produce la predicción final en 3 clases:
        `En riesgo`, `Estable`, `Mejora`.

        ### Anti-overfitting
        - Dropout (0.4 CNN, 0.3 Fusión)
        - EarlyStopping con restauración de mejores pesos
        - ReduceLROnPlateau
        - max_depth limitado en Random Forest
        - Regularización L2 en capas densas
        - Validación cruzada estratificada 5-fold

        ### Dataset
        - **3.000 evaluaciones** · 50 docentes · 8 semestres (2020-1 → 2023-2)
        - 7 asignaturas · 9 variables
        - Clases: Estable (50%), Mejora (30%), En riesgo (20%)

        ### Equipo Sin Convergencia
        | Integrante | Rol |
        |---|---|
        | Valeria Rudas Ruiz *(líder)* | Arquitectura, CNN 1D, integración |
        | Johan A. Vera Lozano | Preprocesamiento, Random Forest |
        | Angela Y. Quiñones Martinez | EDA, visualizaciones, informe técnico |
        | Isaac Oviedo | Evaluación, dashboard, video |

        ### Cronograma
        | Fecha | Hito |
        |---|---|
        | 13–20 jul | Datos y EDA |
        | 21–27 jul | Modelos base |
        | 28 jul–1 ago | Fusión y optimización |
        | 2–4 ago | Entregables finales |
        | 4 ago | **Entrega reportes** |
        | 10 ago | Evaluación panel expertos |
        | 28 ago | Certificación SIC 2025 |
        """
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="ur-footer">
        EduPredict · Capstone Project SIC 2025 · Universidad del Rosario ·
        Equipo Sin Convergencia · Reto 4 — Analítica Educativa
    </div>
    """,
    unsafe_allow_html=True,
)
