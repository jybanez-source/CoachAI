import os
import random
import pandas as pd
import joblib
import streamlit as st

# =========================
# CONFIG (MUST BE FIRST UI)
# =========================
st.set_page_config(page_title="CoachAI", page_icon="⚽", layout="wide")

st.markdown("""
<style>

/* Fondo general con degradado oscuro premium */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #0b1120 100%);
    color: white;
}

/* Quitar borde feo de Streamlit */
header {visibility: hidden;}
footer {visibility: hidden;}

/* Títulos más grandes y elegantes */
h1, h2, h3 {
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* Botones más modernos */
div.stButton > button {
    border-radius: 12px;
    background-color: #1f2937;
    color: white;
    border: 1px solid #374151;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:hover {
    background-color: #2563eb;
    border-color: #2563eb;
    transform: scale(1.02);
}

/* Métricas más limpias */
[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #1f2937;
    padding: 10px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

DATA_PATH = "jugadas_futbol.xlsx"
MODEL_PATH = "modelo_final.joblib"


# =========================
# ROUTER
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go(page: str):
    st.session_state.page = page


# =========================
# LOAD DATA / MODEL
# =========================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encuentro el Excel: {path}")

    df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip().lower() for c in df.columns]

    required = ["liga", "equipo", "zona", "jugada", "pases", "resultado"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    for c in ["liga", "equipo", "zona", "jugada", "resultado"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    df["pases"] = pd.to_numeric(df["pases"], errors="coerce")
    df = df.dropna(subset=["liga", "equipo", "zona", "jugada", "pases", "resultado"])

    # objetivo binario
    df["finaliza"] = df["resultado"].apply(
        lambda x: "finaliza" if x in ["gol", "tiro"] else "no_finaliza"
    )

    # grupo pases (para recomendaciones y entrenamiento)
    df["grupo_pases"] = pd.cut(
        df["pases"],
        bins=[-1, 3, 8, 50],
        labels=["directa (0-3)", "media (4-8)", "elaborada (9+)"]
    )
    return df


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encuentro el modelo: {path}")
    return joblib.load(path)


def pct(x: float) -> str:
    return f"{x*100:.1f}%"


def representative_passes(grupo_pases: str) -> int:
    g = str(grupo_pases)
    if "0-3" in g:
        return 2
    if "4-8" in g:
        return 6
    return 10


def prob_from_model(clf, liga: str, equipo: str, zona: str, jugada: str, pases: int):
    X = pd.DataFrame([{
        "liga": liga,
        "equipo": equipo,
        "zona": zona,
        "jugada": jugada,
        "pases": int(pases)
    }])

    pred = clf.predict(X)[0]
    prob = None
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        clases = list(clf.classes_)
        if "finaliza" in clases:
            prob = float(proba[clases.index("finaliza")])
    return pred, prob


def combo_prob_finaliza(df: pd.DataFrame) -> pd.Series:
    """P(finaliza) por (zona, jugada, grupo_pases)"""
    combo = (
        df.groupby(["zona", "jugada", "grupo_pases"])["finaliza"]
          .value_counts(normalize=True)
          .unstack()
          .fillna(0)
    )
    if "finaliza" not in combo.columns:
        combo["finaliza"] = 0.0
    return combo["finaliza"].copy()


def top_combos(p_finaliza: pd.Series, zonas_ok: set, jugadas_ok: set, k: int = 8):
    cand = p_finaliza[
        [(idx[0] in zonas_ok) and (idx[1] in jugadas_ok) for idx in p_finaliza.index]
    ]
    if len(cand) == 0:
        return []
    cand = cand.sort_values(ascending=False).head(k)
    out = []
    for idx, v in cand.items():
        out.append((idx[0], idx[1], idx[2], float(v)))
    return out


def pick_planA_planB(p_finaliza: pd.Series, ritmo: str, fortaleza: str, estilo: str):
    # filtros por perfil
    zonas = {"centro", "banda"}
    if fortaleza == "centro":
        zonas = {"centro"}
    elif fortaleza == "banda":
        zonas = {"banda"}

    jugadas = {"posesion", "contraataque", "presion"}
    if estilo == "posesion":
        jugadas = {"posesion"}
    elif estilo == "transicion":
        jugadas = {"contraataque", "presion"}

    jugadas_pref = set(jugadas)
    if ritmo == "lento" and "contraataque" in jugadas_pref:
        jugadas_pref.remove("contraataque")

    lista = top_combos(p_finaliza, zonas, jugadas_pref, k=10)
    if not lista:
        lista = top_combos(p_finaliza, zonas, jugadas, k=10)
    if not lista:
        lista = top_combos(p_finaliza, {"centro", "banda"}, jugadas, k=10)
    if not lista:
        idx = p_finaliza.idxmax()
        lista = [(idx[0], idx[1], idx[2], float(p_finaliza.loc[idx]))]

    planA = lista[0]
    planB = None
    for cand in lista[1:]:
        # que sea realmente diferente (zona o jugada o grupo)
        if (cand[0] != planA[0]) or (cand[1] != planA[1]) or (cand[2] != planA[2]):
            planB = cand
            break

    if planB is None:
        # fallback global distinto
        top_global = p_finaliza.sort_values(ascending=False).head(30)
        for idx, v in top_global.items():
            cand = (idx[0], idx[1], idx[2], float(v))
            if (cand[0] != planA[0]) or (cand[1] != planA[1]) or (cand[2] != planA[2]):
                planB = cand
                break

    if planB is None:
        planB = planA

    return planA, planB


# =========================
# MINIJUEGOS STATE
# =========================
if "score" not in st.session_state:
    st.session_state.score = 0
if "streak" not in st.session_state:
    st.session_state.streak = 0
if "quiz_order_seed" not in st.session_state:
    st.session_state.quiz_order_seed = 12345
if "quiz_idx" not in st.session_state:
    st.session_state.quiz_idx = 0
if "quiz_correct" not in st.session_state:
    st.session_state.quiz_correct = 0
if "challenge" not in st.session_state:
    st.session_state.challenge = None
if "plan_seed" not in st.session_state:
    st.session_state.plan_seed = 2026


# =========================
# PAGE: HOME (MEJORADO)
# =========================
if st.session_state.page == "home":
    st.markdown("# 🚀 Bienvenido a la revolución del fútbol")
st.markdown("""
> *"La innovación y el análisis hacen mejor a los jugadores."*  
> — **Pep Guardiola**
""")
    st.write("**CoachAI** combina datos + modelo para recomendar tácticas y generar entrenamientos adaptados a tu equipo.")

    # estadísticas reales del Excel (si está disponible)
    try:
        df_home = load_data(DATA_PATH)
        n_jugadas = len(df_home)
        tasa_finaliza_global = float((df_home["finaliza"] == "finaliza").mean())

        pfin = combo_prob_finaliza(df_home)
        idx_top = pfin.idxmax()
        top_prob = float(pfin.loc[idx_top])
        top_patron = f"{idx_top[0]} + {idx_top[1]} + {idx_top[2]}"

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Jugadas analizadas", n_jugadas)
        with m2:
            st.metric("Tasa global de finalización", pct(tasa_finaliza_global))
        with m3:
            st.metric("Patrón top actual", pct(top_prob))

        st.caption(f"Patrón top: **{top_patron}**")

    except Exception:
        st.info("Para ver estadísticas en la portada, asegúrate de tener el Excel en la carpeta.")

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("⚽ CoachAI", use_container_width=True):
            go("coach")
    with c2:
        if st.button("🎮 Minijuegos", use_container_width=True):
            go("games")
    with c3:
        if st.button("ℹ️ Cómo funciona", use_container_width=True):
            go("how")

    st.stop()


# =========================
# PAGE: COACH (PLAN A + PLAN B, ENTRENAMIENTO VARIABLE)
# =========================
if st.session_state.page == "coach":
    top = st.columns([1, 6])
    with top[0]:
        if st.button("⬅️ Inicio"):
            go("home")
            st.stop()
    with top[1]:
        st.markdown("# ⚽ CoachAI")

    # Cargar datos/modelo
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error("No se pudieron cargar los datos del Excel.")
        st.code(str(e))
        st.stop()

    try:
        clf = load_model(MODEL_PATH)
    except Exception as e:
        st.error("No se pudo cargar el modelo entrenado.")
        st.code(str(e))
        st.stop()

    ligas = sorted(df["liga"].unique().tolist())
    equipos = sorted(df["equipo"].unique().tolist())

    # Inputs (solo perfil, como quieres)
    st.subheader("Perfil del equipo")
    a, b, c = st.columns(3)
    with a:
        ritmo = st.selectbox("Ritmo", ["rapido", "lento"], index=0)
    with b:
        fortaleza = st.selectbox("Fortaleza", ["centro", "banda", "ambas"], index=2)
    with c:
        estilo = st.selectbox("Estilo", ["transicion", "posesion", "mezcla"], index=2)

    duracion = st.slider("Duración de la sesión (min)", 60, 90, 75, step=5)

    # Contexto del modelo (sin “inventar”: por defecto usa lo más común del dataset)
    st.subheader("Contexto para el modelo (solo para calcular probabilidad)")
    st.caption("El perfil manda en la recomendación. Esto solo sirve para que el modelo pueda calcular una probabilidad.")
    usar_contexto_manual = st.checkbox("Elegir liga y equipo manualmente", value=False)

    if usar_contexto_manual:
        c1, c2 = st.columns(2)
        with c1:
            liga_sel = st.selectbox("Liga", ligas)
        with c2:
            equipo_sel = st.selectbox("Equipo", equipos)
    else:
        liga_sel = str(df["liga"].mode().iloc[0]) if len(df) else ""
        equipo_sel = str(df["equipo"].mode().iloc[0]) if len(df) else ""
        st.info(f"Contexto automático: liga **{liga_sel}**, equipo **{equipo_sel}**")

    # Plan A + Plan B (desde datos)
    p_finaliza = combo_prob_finaliza(df)
    planA, planB = pick_planA_planB(p_finaliza, ritmo, fortaleza, estilo)

    zonaA, jugadaA, grupoA, probA = planA
    zonaB, jugadaB, grupoB, probB = planB

    st.divider()
    st.subheader("Recomendación táctica (desde datos + perfil)")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("### ✅ Plan A (principal)")
        st.write(f"- **Zona:** {zonaA}")
        st.write(f"- **Tipo de jugada:** {jugadaA}")
        st.write(f"- **Construcción:** {grupoA}")
        st.write(f"- **Tasa de finalización (datos):** **{pct(probA)}**")
    with colB:
        st.markdown("### 🅱️ Plan B (alternativa)")
        st.write(f"- **Zona:** {zonaB}")
        st.write(f"- **Tipo de jugada:** {jugadaB}")
        st.write(f"- **Construcción:** {grupoB}")
        st.write(f"- **Tasa de finalización (datos):** **{pct(probB)}**")

    # Validación del modelo (A y B)
    st.divider()
    st.subheader("Validación del modelo (probabilidad)")

    pasesA = representative_passes(grupoA)
    predA, probModelA = prob_from_model(clf, liga_sel, equipo_sel, zonaA, jugadaA, pasesA)

    pasesB = representative_passes(grupoB)
    predB, probModelB = prob_from_model(clf, liga_sel, equipo_sel, zonaB, jugadaB, pasesB)

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("### Plan A")
        st.write(f"Pases usados: **{pasesA}**")
        st.write("Predicción:", "FINALIZA" if predA == "finaliza" else "NO FINALIZA")
        if probModelA is not None:
            st.progress(int(probModelA * 100))
            st.write(f"Probabilidad: **{pct(probModelA)}**")
        else:
            st.info("El modelo no devuelve probabilidades.")
    with v2:
        st.markdown("### Plan B")
        st.write(f"Pases usados: **{pasesB}**")
        st.write("Predicción:", "FINALIZA" if predB == "finaliza" else "NO FINALIZA")
        if probModelB is not None:
            st.progress(int(probModelB * 100))
            st.write(f"Probabilidad: **{pct(probModelB)}**")
        else:
            st.info("El modelo no devuelve probabilidades.")

    # Pizarra táctica (simple y útil)
    st.divider()
    st.subheader("Pizarra táctica")
    p1, p2 = st.columns([1.1, 1])

    with p1:
        st.markdown("### Principios sugeridos (Plan A)")
        if jugadaA == "contraataque":
            st.write("- Tras robo: primer pase hacia delante.")
            st.write("- Atacar el espacio con pocos toques.")
            st.write("- Finalizar sin alargar la acción.")
        elif jugadaA == "posesion":
            st.write("- Paciencia + movilidad para generar línea de pase.")
            st.write("- Cambios de orientación y tercer hombre.")
            st.write("- Preparar el último pase antes de finalizar.")
        else:  # presion
            st.write("- Presión tras pérdida (5 segundos).")
            st.write("- Robo alto y finalización rápida.")
            st.write("- Compactar para recuperar cerca del área rival.")

        if zonaA == "centro":
            st.write("- Clave zona: **interior** (pared / filtrado / tiro frontal).")
        else:
            st.write("- Clave zona: **banda** (desborde / centro tenso / pase atrás).")

        st.write(f"- Construcción recomendada: **{grupoA}**")

        st.markdown("### Plan B (cuándo usarlo)")
        st.write("Usa Plan B cuando el rival te cierre la primera opción o cambie el partido (marcador/energía).")
        st.write(f"- Alternativa: **{zonaB} + {jugadaB} + {grupoB}**")

    with p2:
        st.markdown("### Pizarra libre (notas)")
        notas = st.text_area(
            "Escribe movimientos, roles y flechas en texto",
            height=240,
            placeholder="Ej: Extremo fija, interior ataca espacio.\nFlechas: RB -> MC -> MP -> 9 -> tiro"
        )
        st.download_button(
            "⬇️ Descargar pizarra (.txt)",
            data=notas.encode("utf-8"),
            file_name="pizarra_tactica.txt",
            mime="text/plain",
            use_container_width=True
        )

    # =========================
    # ENTRENAMIENTO VARIABLE (basado en Plan A)
    # =========================
    st.divider()
    st.subheader(f"Plan de entrenamiento ({duracion} min) — Basado en Plan A")

    # Seed para variar tareas sin que sea siempre lo mismo
    cseed = st.columns([1, 1, 3])
    with cseed[0]:
        if st.button("🔁 Generar otra sesión"):
            st.session_state.plan_seed += 1
            st.rerun()
    with cseed[1]:
        st.caption(f"Variante: {st.session_state.plan_seed}")

    rng = random.Random(st.session_state.plan_seed)

    # bloques base (escalan con duración)
    base = {"cal": 10, "tec": 12, "p1": 16, "p2": 16, "fin": 12, "cool": 7}
    factor = duracion / sum(base.values())
    t = {k: max(5, int(round(v * factor))) for k, v in base.items()}
    drift = duracion - sum(t.values())
    t["p2"] = max(5, t["p2"] + drift)

    # regla de pases a partir del grupoA
    gtxt = str(grupoA)
    if "0-3" in gtxt:
        regla_pases = "máximo 3 pases antes de finalizar"
    elif "4-8" in gtxt:
        regla_pases = "mínimo 4 y máximo 8 pases antes de finalizar"
    else:
        regla_pases = "mínimo 9 pases antes de finalizar"

    st.markdown("### 🎯 Objetivo")
    st.write(f"Entrenar el patrón **Plan A**: **{jugadaA} por {zonaA}** con **{grupoA}** (regla: {regla_pases}).")

    # Biblioteca de tareas (varía según estilo y zona)
    # (No dependemos de equipos reales, solo del perfil + Plan A)
    calentamientos = [
        "Rondo 5v2/6v2: al recuperar, 3 segundos para pase hacia delante.",
        "Rondo con comodines: puntúa doble si progresas por la zona objetivo.",
        "Activación + rondo: cada 5 pases, cambio de ritmo (conducción/tercer hombre).",
    ]

    tecnica_transicion = [
        "Circuito transición: pase vertical → apoyo → tercer hombre → tiro.",
        "3 carriles: recuperación en medio → 2 pases hacia delante → finalización.",
        "Olas cortas 2v1/3v2: decisión rápida y tiro.",
    ]

    tecnica_posesion = [
        "Conservación 6v3: atraer → encontrar el hombre libre → último pase y tiro.",
        "Posesión con 2 comodines: objetivo cambiar orientación antes de finalizar.",
        "Tarea de tercer hombre: pared + pase filtrado + tiro.",
    ]

    tecnica_presion = [
        "Presión tras pérdida: 5 segundos para recuperar y rematar.",
        "Juego 4v4+2: al perder, presión inmediata; al recuperar, tiro rápido.",
        "Encerrona en zona: robo en banda/centro y finalización en 3 pases.",
    ]

    principal_transicion = [
        "6v6: cada acción empieza con robo simulado y contraataque.",
        "7v7: tras recuperación, atacar en 8 segundos; éxito = tiro.",
        "8v8: si recuperas en campo rival, el tiro vale doble.",
        "Transiciones 4v3 repetidas: roles claros (apoyo/ruptura).",
        "Juego condicionado: prohibido volver atrás tras recuperar (solo hacia delante).",
        "Contraataque por carril central (si zona centro) o por fuera (si banda).",
    ]

    principal_posesion = [
        "7v7: objetivo mantener y finalizar cumpliendo regla de pases.",
        "8v8: puntúa doble si el último pase rompe línea (filtrado/tercer hombre).",
        "Juego de posición: 3 zonas, el balón debe pasar por zona interior o banda según objetivo.",
        "Conservación con finalización: tras X pases, buscar tiro en 10 segundos.",
        "Superioridad 7v5: automatismos para crear ocasión.",
        "Partidillo: goles tras secuencia organizada valen doble.",
    ]

    principal_presion = [
        "8v8: regla 5 segundos tras pérdida; si recuperas, finaliza.",
        "Juego en espacio reducido: presión alta, robos y tiros.",
        "6v6+2 comodines: presión coordinada (saltos) para recuperar arriba.",
        "Partido con zonas: si robas en zona alta, tiro obligatorio en 6 segundos.",
        "Reto de robos: cada robo alto suma punto extra si termina en tiro.",
        "Encadenar 3 presiones buenas = bonus.",
    ]

    # Elegimos biblioteca por estilo input (y dentro variamos aleatoriamente)
    if estilo == "posesion":
        tec_pool = tecnica_posesion
        p1_pool = principal_posesion
        p2_pool = principal_posesion
    elif estilo == "transicion":
        tec_pool = tecnica_transicion
        p1_pool = principal_transicion
        p2_pool = principal_transicion
    else:  # mezcla
        # Mezcla: mitad posesión y mitad transición/presión, pero SIEMPRE basado en Plan A
        # (El Plan B lo dejamos como alternativa táctica, no como base del entreno)
        tec_pool = tecnica_posesion + tecnica_transicion
        p1_pool = principal_posesion
        p2_pool = principal_transicion

    # Ajuste por jugadaA (Plan A) para que de verdad mande el patrón recomendado
    if jugadaA == "posesion":
        tec_pool = tecnica_posesion
        p1_pool = principal_posesion
    elif jugadaA == "contraataque":
        tec_pool = tecnica_transicion
        p1_pool = principal_transicion
    else:
        tec_pool = tecnica_presion
        p1_pool = principal_presion

    # Elegimos tareas (varían según seed)
    cal = rng.choice(calentamientos)
    tec = rng.choice(tec_pool)
    p1 = rng.choice(p1_pool)

    # P2 depende del estilo (si mezcla, es distinto a P1)
    if estilo == "mezcla":
        # escoger del pool de transición si p1 era posesión, o viceversa
        if p1 in principal_posesion:
            p2 = rng.choice(principal_transicion)
        else:
            p2 = rng.choice(principal_posesion)
    else:
        p2 = rng.choice(p2_pool)

    # Ajuste por zonaA (centro/banda) para darle coherencia
    extra_zona = ""
    if zonaA == "centro":
        extra_zona = "Extra: puntúa doble si la progresión pasa por interior antes del tiro."
    else:
        extra_zona = "Extra: puntúa doble si la finalización viene de banda (desborde/centro/pase atrás)."

    st.markdown("### 1) Calentamiento")
    st.write(f"**{t['cal']} min** — {cal}")

    st.markdown("### 2) Técnica específica")
    st.write(f"**{t['tec']} min** — {tec}")
    st.write(f"Regla de sesión: **{regla_pases}**.")

    st.markdown("### 3) Tarea principal 1 (alineada con Plan A)")
    st.write(f"**{t['p1']} min** — {p1}")
    st.write(f"Condición: **{regla_pases}**. Éxito = **tiro** (o gol) cumpliendo la condición.")
    st.write(extra_zona)

    st.markdown("### 4) Tarea principal 2 (variación según estilo)")
    st.write(f"**{t['p2']} min** — {p2}")
    st.write("Gamificación: +1 tiro, +2 gol, +2 extra si se cumple el patrón recomendado.")

    st.markdown("### 5) Finalización")
    st.write(f"**{t['fin']} min** — Finalizar con intención.")
    if jugadaA == "contraataque":
        st.write("- Olas 3v2 / 4v3 desde medio campo. Finalizar rápido.")
    elif jugadaA == "posesion":
        st.write("- Ataque organizado: último pase + tiro. No precipitar.")
    else:
        st.write("- Robo alto + tiro en pocos segundos.")
    st.write(f"Regla: **{regla_pases}** (adaptada al patrón).")

    st.markdown("### 6) Vuelta a la calma")
    st.write(f"**{t['cool']} min** — Trote suave + movilidad + estiramientos.")
    st.write("Cierre: 2 minutos de reflexión (qué funcionó, qué ajustar).")

    st.markdown("### 🧪 Indicadores (para tu evaluación)")
    st.write("- % de acciones que terminan en tiro cuando se intenta el Plan A.")
    st.write("- Cumplimiento de zona y regla de pases.")
    st.write("- Comparación entre 2 sesiones distintas (otra variante).")

    st.stop()


# =========================
# PAGE: GAMES (RETOS DESDE FILA REAL + QUIZ VARIADO)
# =========================
if st.session_state.page == "games":
    top = st.columns([1, 6])
    with top[0]:
        if st.button("⬅️ Inicio"):
            go("home")
            st.stop()
    with top[1]:
        st.markdown("# 🎮 Minijuegos")

    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error("No se pudieron cargar los datos para minijuegos.")
        st.code(str(e))
        st.stop()

    try:
        clf = load_model(MODEL_PATH)
    except Exception as e:
        st.error("No se pudo cargar el modelo para minijuegos.")
        st.code(str(e))
        st.stop()

    st.info(f"Puntos: **{st.session_state.score}** | Racha: **{st.session_state.streak}**")

    tab1, tab2 = st.tabs(["🧠 Quiz táctico", "🎯 Adivina la probabilidad"])

    # ---------- QUIZ ----------
    with tab1:
        st.subheader("Quiz táctico (varía en cada intento)")

        # Banco de preguntas (no siempre las mismas, se baraja)
        bank = [
            ("En transición ofensiva, ¿qué es clave en los primeros segundos?",
             ["Esperar a que suban todos", "Atacar el espacio mirando hacia delante", "Dar siempre pases hacia atrás"],
             1,
             "En transición el rival suele estar desordenado: atacar el espacio aumenta opciones."),
            ("Si tu objetivo es progresar por el centro, ¿qué ayuda más?",
             ["Pared/tercer hombre y pase interior", "Centros desde muy lejos", "Conservar sin intención"],
             0,
             "El carril central permite paredes/filtrados y tiros desde zonas peligrosas."),
            ("¿Qué es presión tras pérdida?",
             ["Replegar siempre", "Presionar inmediatamente tras perder para recuperar", "Sacar rápido de portería"],
             1,
             "Presionar tras perder busca recuperar con el rival sin estructura."),
            ("¿Para qué sirve una regla de pases en un ejercicio?",
             ["Para castigar", "Para controlar el tipo de ataque y el ritmo", "Para que el juego sea más lento siempre"],
             1,
             "Te fuerza a construir o finalizar rápido según el objetivo."),
            ("¿Qué indicador es más estable para medir mejora ofensiva en tareas?",
             ["Solo goles", "% de jugadas que acaban en tiro cumpliendo condición", "Posesión total"],
             1,
             "El tiro es más estable que el gol, que depende de más factores."),
            ("Si el equipo es lento, ¿qué suele ser más coherente?",
             ["Contraataque directo 0–3 pases todo el rato", "Posesión organizada y progresión paciente", "Solo balones largos"],
             1,
             "Equipo lento suele rendir mejor con organización y estructura."),
            ("Si tu fortaleza es banda, ¿qué finalización es típica?",
             ["Centro tenso o pase atrás", "Siempre tiro desde 40 metros", "Nunca usar banda"],
             0,
             "En banda son comunes desborde, centro y pase atrás."),
        ]

        # Preparar orden aleatorio con seed
        rng = random.Random(st.session_state.quiz_order_seed)
        order = list(range(len(bank)))
        rng.shuffle(order)
        selected = order[:5]  # 5 preguntas

        i = st.session_state.quiz_idx
        if i >= len(selected):
            st.success(f"Quiz terminado. Aciertos: {st.session_state.quiz_correct}/5")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔁 Repetir (nuevo quiz)"):
                    st.session_state.quiz_idx = 0
                    st.session_state.quiz_correct = 0
                    st.session_state.streak = 0
                    st.session_state.quiz_order_seed += 1
                    st.rerun()
            with c2:
                if st.button("🧹 Reset puntos"):
                    st.session_state.score = 0
                    st.session_state.streak = 0
                    st.rerun()
        else:
            q_idx = selected[i]
            q, opts, ans, why = bank[q_idx]

            # Barajar opciones manteniendo respuesta correcta
            pairs = list(enumerate(opts))
            rng_opts = random.Random(st.session_state.quiz_order_seed + i + 999)
            rng_opts.shuffle(pairs)
            shuffled_opts = [p[1] for p in pairs]
            new_ans = [p[0] for p in pairs].index(ans)

            st.write(f"**Pregunta {i+1}/5**")
            st.write(q)
            choice = st.radio("Elige una opción:", shuffled_opts, index=None, key=f"quiz_{i}")

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Responder", use_container_width=True):
                    if choice is None:
                        st.warning("Elige una opción.")
                    else:
                        ok = (shuffled_opts.index(choice) == new_ans)
                        if ok:
                            st.success("✅ Correcto")
                            st.session_state.score += 10
                            st.session_state.streak += 1
                            st.session_state.quiz_correct += 1
                        else:
                            st.error("❌ Incorrecto")
                            st.session_state.streak = 0
                        st.info("Explicación: " + why)
            with c2:
                if st.button("Siguiente ➜", use_container_width=True):
                    st.session_state.quiz_idx += 1
                    st.rerun()

    # ---------- ADIVINA PROB (DESDE FILA REAL) ----------
    with tab2:
        st.subheader("Adivina la probabilidad (retos desde datos reales)")
        st.write("Generamos una situación **real del Excel** y tú apuestas si la probabilidad de finalizar es baja/media/alta.")

        if st.button("🎲 Generar reto real", use_container_width=True):
            row = df.sample(1).iloc[0]
            st.session_state.challenge = {
                "liga": row["liga"],
                "equipo": row["equipo"],
                "zona": row["zona"],
                "jugada": row["jugada"],
                "pases": int(row["pases"]),
            }

        ch = st.session_state.challenge
        if ch is None:
            st.info("Pulsa **Generar reto real** para empezar.")
        else:
            st.write("### Situación")
            st.write(f"- Liga: **{ch['liga']}**")
            st.write(f"- Equipo: **{ch['equipo']}**")
            st.write(f"- Zona: **{ch['zona']}**")
            st.write(f"- Jugada: **{ch['jugada']}**")
            st.write(f"- Pases: **{ch['pases']}**")

            guess = st.radio("Tu apuesta:", ["baja (0–39%)", "media (40–59%)", "alta (60–100%)"], index=None)

            if st.button("Comprobar", use_container_width=True):
                pred, prob = prob_from_model(
                    clf, ch["liga"], ch["equipo"], ch["zona"], ch["jugada"], ch["pases"]
                )
                st.write("Predicción:", "FINALIZA" if pred == "finaliza" else "NO FINALIZA")

                if prob is None:
                    st.warning("El modelo no devuelve probabilidades (solo predicción).")
                else:
                    st.write(f"Probabilidad (modelo): **{pct(prob)}**")

                    bucket = "baja" if prob < 0.40 else ("media" if prob < 0.60 else "alta")
                    guessed_bucket = None
                    if guess:
                        guessed_bucket = "baja" if guess.startswith("baja") else ("media" if guess.startswith("media") else "alta")

                    if guessed_bucket is None:
                        st.warning("Elige una apuesta antes de comprobar.")
                    else:
                        if guessed_bucket == bucket:
                            st.success("✅ Acierto")
                            st.session_state.score += 15
                            st.session_state.streak += 1
                        else:
                            st.error("❌ Fallo")
                            st.session_state.streak = 0

    st.stop()


# =========================
# PAGE: HOW IT WORKS
# =========================
if st.session_state.page == "how":
    top = st.columns([1, 6])
    with top[0]:
        if st.button("⬅️ Inicio"):
            go("home")
            st.stop()
    with top[1]:
        st.markdown("# ℹ️ Cómo funciona CoachAI")

    with st.expander("📦 Datos (dataset)"):
        st.write("- Fuente: jugadas anotadas en Excel.")
        st.write("- Variables: liga, equipo, zona, tipo de jugada y número de pases.")
        st.write("- Etiqueta: **finaliza** (tiro o gol) / **no_finaliza** (pérdida).")

    with st.expander("🧠 Modelo (Machine Learning)"):
        st.write("- Modelo usado: **Logistic Regression** (clasificación).")
        st.write("- Las variables categóricas se convierten en números (OneHotEncoder).")
        st.write("- Entrena con ejemplos y luego predice para nuevas jugadas.")

    with st.expander("📊 Recomendación + validación"):
        st.write("- La recomendación sale de los **datos**: qué combinaciones finalizan más (zona + jugada + grupo de pases).")
        st.write("- El perfil del equipo filtra lo que tiene sentido.")
        st.write("- Luego el modelo calcula una probabilidad para Plan A y Plan B.")

    with st.expander("⚠️ Limitaciones"):
        st.write("- Pocas variables: faltan posiciones exactas, rival, fatiga, contexto, etc.")
        st.write("- Datos manuales: puede haber sesgo o errores de anotación.")
        st.write("- Por eso es más fiable como **recomendador/analizador** que como predictor perfecto.")

    with st.expander("🚀 Mejoras futuras"):
        st.write("- Añadir variables más informativas (baratas de anotar): zona final, tipo de último pase, superioridad numérica, altura de recuperación.")
        st.write("- Más jugadas y mejor equilibrio de clases.")
        st.write("- Pizarra avanzada (arrastrar jugadores/dibujar) con una librería específica si se permite.")

    st.stop()
