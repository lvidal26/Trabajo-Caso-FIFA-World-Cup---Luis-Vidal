import io
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN DE LA PÁGINA (primera instrucción de Streamlit)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FIFA WC 2026 – EDA",
    page_icon="⚽",
    layout="wide",
)

AUTOR = "Luis Aaron Vidal Vela"
CURSO = "Especialización en Python for Analytics – DMC Institute"
AÑO = 2026
LOCAL_DATASET = Path(__file__).parent / "data" / "fifa_world_cup_2026_player_performance.csv"

REQUIRED_COLUMNS = [
    "player_id", "player_name", "team", "position", "match_id", "match_date",
    "tournament_stage", "match_result", "minutes_played", "player_rating",
    "performance_score", "pass_accuracy", "distance_covered_km", "top_speed_kmh",
]

KEY_NUMERIC = ["player_rating", "performance_score", "pass_accuracy",
               "distance_covered_km", "top_speed_kmh"]

METRIC_GROUPS = {
    "Ofensivas": ["goals", "assists", "shots", "shots_on_target", "expected_goals_xg",
                  "expected_assists_xa", "key_passes", "offensive_contribution"],
    "Defensivas": ["tackles", "interceptions", "clearances", "blocks", "recoveries",
                   "defensive_actions", "aerial_duels_won", "defensive_contribution"],
    "Físicas": ["distance_covered_km", "sprint_distance_km", "top_speed_kmh",
                "accelerations", "decelerations", "stamina_score"],
    "Rendimiento": ["player_rating", "performance_score", "creativity_score",
                    "consistency_score", "clutch_performance_score", "pressure_resistance"],
}

GOALKEEPER_METRICS = ["saves", "save_percentage", "punches", "clean_sheet",
                      "goals_conceded", "penalty_saves"]

TOURNAMENT_COLUMNS = ["total_goals_tournament", "total_assists_tournament",
                      "total_minutes_tournament", "player_of_match_awards",
                      "tournament_rating"]

STAGE_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarter Finals",
               "Semi Finals", "Third Place Match", "Final"]
RESULT_ORDER = ["W", "D", "L"]
RESULT_LABELS = {"W": "Victoria", "D": "Empate", "L": "Derrota"}
POSITION_ORDER = ["Goalkeeper", "Defender", "Midfielder", "Forward"]


# -----------------------------------------------------------------------------
# 4. FUNCIONES PERSONALIZADAS
# -----------------------------------------------------------------------------
def classify_columns(df: pd.DataFrame) -> dict:
    """Clasifica las columnas de un DataFrame en numéricas, categóricas y de fecha."""
    dates = df.select_dtypes(include="datetime").columns.tolist()
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in dates]
    return {"Numéricas": numeric, "Categóricas": categorical, "Fecha": dates}


def show_fig(fig) -> None:
    """Muestra una figura de Matplotlib en Streamlit y libera memoria."""
    st.pyplot(fig)
    plt.close(fig)


def fmt_num(value: float, decimals: int = 2) -> str:
    """Formatea números con separador de miles."""
    return f"{value:,.{decimals}f}"


@st.cache_data(show_spinner="Leyendo el archivo CSV...")
def load_csv(source) -> pd.DataFrame:
    """Lee el CSV una sola vez y lo guarda en caché."""
    return pd.read_csv(source)


# -----------------------------------------------------------------------------
# 5. CLASE PRINCIPAL (POO)
# -----------------------------------------------------------------------------
class DataAnalyzer:
    """Encapsula la validación, preparación, estadísticas, visualización y
    filtrado del dataset de rendimiento de jugadores."""

    def __init__(self, df: pd.DataFrame):
        self.raw = df
        self.df = df.copy()
        self._prepare()

    # ---------- Carga y validación ----------
    @staticmethod
    def validate(df: pd.DataFrame) -> list:
        """Devuelve las columnas obligatorias que faltan en el archivo."""
        return [col for col in REQUIRED_COLUMNS if col not in df.columns]

    def _prepare(self) -> None:
        """Convierte tipos y crea columnas auxiliares."""
        self.df["match_date"] = pd.to_datetime(self.df["match_date"], errors="coerce")
        self.df["played"] = self.df["minutes_played"] > 0
        self.df["tournament_stage"] = pd.Categorical(
            self.df["tournament_stage"], categories=STAGE_ORDER, ordered=True)

    def data(self, only_played: bool = True) -> pd.DataFrame:
        """Registros completos o solo de jugadores que disputaron minutos."""
        return self.df[self.df["played"]] if only_played else self.df

    # ---------- Ítem 1: información general ----------
    def info_text(self) -> str:
        buffer = io.StringIO()
        self.raw.info(buf=buffer)
        return buffer.getvalue()

    def dtypes_table(self) -> pd.DataFrame:
        counts = self.raw.dtypes.astype(str).value_counts()
        return counts.rename_axis("Tipo de dato").reset_index(name="Columnas")

    def duplicated_count(self) -> int:
        return int(self.raw.duplicated().sum())

    def null_count(self) -> int:
        return int(self.raw.isna().sum().sum())

    # ---------- Ítem 2: clasificación ----------
    def classify_variables(self) -> dict:
        return classify_columns(self.raw.assign(
            match_date=pd.to_datetime(self.raw["match_date"], errors="coerce")))

    # ---------- Ítem 3: estadísticas descriptivas ----------
    def describe_numeric(self, columns: list, only_played: bool = True) -> pd.DataFrame:
        data = self.data(only_played)[columns]
        desc = data.describe().T
        desc["mediana"] = data.median()
        desc["moda"] = data.mode().iloc[0]
        desc["IQR"] = desc["75%"] - desc["25%"]
        desc["CV (%)"] = (desc["std"] / desc["mean"] * 100).round(1)
        return desc.round(3)

    def outliers_iqr(self, columns: list, only_played: bool = True) -> pd.DataFrame:
        """Detecta valores extremos con la regla de 1.5 × IQR."""
        data = self.data(only_played)
        rows = []
        for col in columns:
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((data[col] < low) | (data[col] > high)).sum())
            rows.append({"Variable": col, "Límite inferior": round(low, 3),
                         "Límite superior": round(high, 3), "Outliers": n_out,
                         "% Outliers": round(n_out / len(data) * 100, 2)})
        return pd.DataFrame(rows)

    # ---------- Ítem 4: faltantes ----------
    def missing_table(self) -> pd.DataFrame:
        missing = self.raw.isna().sum()
        return pd.DataFrame({
            "Nulos": missing,
            "Porcentaje (%)": (missing / len(self.raw) * 100).round(2),
        }).sort_values("Nulos", ascending=False)

    def zero_minutes_summary(self) -> dict:
        n_zero = int((~self.df["played"]).sum())
        return {"n": n_zero, "pct": n_zero / len(self.df) * 100}

    # ---------- Ítem 6: categóricas ----------
    def frequency_table(self, column: str, only_played: bool = False) -> pd.DataFrame:
        data = self.data(only_played)[column]
        freq = data.value_counts()
        return pd.DataFrame({"Frecuencia": freq,
                             "Proporción (%)": (freq / freq.sum() * 100).round(2)})

    # ---------- Ítem 7: numérico vs categórico ----------
    def group_stats(self, num: str, cat: str, only_played: bool = True) -> pd.DataFrame:
        data = self.data(only_played)
        return (data.groupby(cat, observed=True)[num]
                .agg(["count", "mean", "median", "std", "min", "max"]).round(3))

    def per90(self, metrics: list, by: str = "position") -> pd.DataFrame:
        """Métricas por cada 90 minutos jugados, agregadas por grupo."""
        data = self.data(True)
        grouped = data.groupby(by, observed=True)
        totals = grouped[metrics].sum()
        minutes = grouped["minutes_played"].sum()
        return (totals.div(minutes, axis=0) * 90).round(3)

    # ---------- Ítem 8: categórico vs categórico ----------
    def crosstab(self, row: str, col: str, normalize=False,
                 only_played: bool = False) -> pd.DataFrame:
        data = self.data(only_played)
        ct = pd.crosstab(data[row], data[col], normalize=normalize)
        return (ct * 100).round(1) if normalize else ct

    def team_results(self) -> pd.DataFrame:
        """Resultados por selección contados por partido (no por jugador)."""
        matches = self.df.drop_duplicates(["team", "match_id"])
        ct = pd.crosstab(matches["team"], matches["match_result"])
        ct = ct.reindex(columns=RESULT_ORDER, fill_value=0)
        ct["Partidos"] = ct.sum(axis=1)
        ct["% Victorias"] = (ct["W"] / ct["Partidos"] * 100).round(1)
        return ct.sort_values("% Victorias", ascending=False)

    # ---------- Ítem 9: filtros ----------
    def filter_data(self, filters: dict, ranges: dict, only_played: bool = True) -> pd.DataFrame:
        data = self.data(only_played)
        for col, values in filters.items():
            if values:
                data = data[data[col].isin(values)]
        for col, (low, high) in ranges.items():
            data = data[data[col].between(low, high)]
        return data

    def player_ranking(self, data: pd.DataFrame, metric: str, min_matches: int,
                       top_n: int) -> pd.DataFrame:
        agg = (data.groupby(["player_name", "team", "position"])
               .agg(partidos=("match_id", "nunique"), promedio=(metric, "mean"))
               .reset_index())
        agg = agg[agg["partidos"] >= min_matches]
        return agg.sort_values("promedio", ascending=False).head(top_n).round(3)

    def tournament_snapshot(self) -> pd.DataFrame:
        """Último registro de cada jugador: evita sumar variables acumuladas."""
        last = self.df.sort_values("match_date").groupby("player_id").tail(1)
        return last[["player_name", "team", "position"] + TOURNAMENT_COLUMNS]

    # ---------- Visualizaciones ----------
    def plot_hist(self, column: str, by_position: bool, only_played: bool = True):
        data = self.data(only_played)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        if by_position:
            sns.histplot(data=data, x=column, hue="position", hue_order=POSITION_ORDER,
                         element="step", stat="density", common_norm=False, bins=30, ax=ax)
        else:
            sns.histplot(data=data, x=column, bins=30, kde=True, ax=ax)
            ax.axvline(data[column].mean(), color="crimson", ls="--", label="Media")
            ax.axvline(data[column].median(), color="black", ls=":", label="Mediana")
            ax.legend()
        ax.set_title(f"Distribución de {column}")
        return fig

    def plot_bar(self, column: str, only_played: bool = False):
        freq = self.frequency_table(column, only_played)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(x=freq["Proporción (%)"], y=freq.index.astype(str), ax=ax,
                    hue=freq.index.astype(str), legend=False)
        for i, v in enumerate(freq["Proporción (%)"]):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.set_xlabel("Proporción (%)")
        ax.set_ylabel("")
        ax.set_title(f"Distribución de {column}")
        return fig

    def plot_box(self, num: str, cat: str, order=None, only_played: bool = True):
        data = self.data(only_played)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.boxplot(data=data, x=cat, y=num, order=order, hue=cat, hue_order=order,
                    legend=False, showmeans=True, fliersize=1.5, ax=ax,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black"})
        ax.set_title(f"{num} según {cat}")
        ax.set_xlabel("")
        return fig

    @staticmethod
    def plot_heatmap(table: pd.DataFrame, title: str, fmt: str = ".1f"):
        height = max(3.5, 0.28 * len(table))
        fig, ax = plt.subplots(figsize=(7, height))
        sns.heatmap(table, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        return fig

    # ---------- Ítem 10 / conclusiones ----------
    def key_metrics(self) -> dict:
        played = self.data(True)
        by_pos = played.groupby("position")
        by_res = played.groupby("match_result")["player_rating"].mean()
        per90 = self.per90(["goals", "assists", "defensive_actions"])
        corr = played[["player_rating", "performance_score", "offensive_contribution",
                       "defensive_contribution", "distance_covered_km",
                       "market_value_eur"]].corr()["player_rating"]
        stage = played.groupby("tournament_stage", observed=True)["player_rating"].mean()
        return {
            "zero": self.zero_minutes_summary(),
            "rating_all": self.df["player_rating"].mean(),
            "rating_played": played["player_rating"].mean(),
            "rating_pos": by_pos["player_rating"].mean().sort_values(ascending=False),
            "rating_res": by_res,
            "per90": per90,
            "pass_pos": by_pos["pass_accuracy"].mean().sort_values(ascending=False),
            "dist_pos": by_pos["distance_covered_km"].mean().sort_values(ascending=False),
            "speed_pos": by_pos["top_speed_kmh"].mean().sort_values(ascending=False),
            "corr": corr.drop("player_rating").sort_values(ascending=False),
            "stage": stage,
        }


# -----------------------------------------------------------------------------
# 6. MÓDULO 1: HOME
# -----------------------------------------------------------------------------
def page_home() -> None:
    st.title("⚽ FIFA World Cup 2026 – Player Performance EDA")
    st.markdown(
        "Aplicación interactiva de **Análisis Exploratorio de Datos (EDA)** sobre el "
        "rendimiento de jugadores y selecciones del Mundial 2026. Explora métricas "
        "ofensivas, defensivas, físicas y contextuales para comparar por **posición, "
        "equipo, fase del torneo y jugador**. \n\n"
        "> El objetivo es **describir y comparar**, no construir modelos predictivos."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("👤 Autor")
        st.markdown(f"**{AUTOR}**  \n{CURSO}  \nAño {AÑO}")
    with col2:
        st.subheader("📊 Dataset")
        st.markdown(
            "`fifa_world_cup_2026_player_performance.csv`  \n"
            "54,600 registros · 75 variables  \n"
            "1,248 jugadores · 1,050 partidos · 48 selecciones"
        )
    with col3:
        st.subheader("🛠️ Tecnologías")
        st.markdown("Python · Pandas · NumPy  \nMatplotlib · Seaborn  \nStreamlit · POO")

    st.subheader("¿Qué contiene cada registro?")
    st.markdown(
        "Cada fila es la **actuación de un jugador en un partido**: datos personales "
        "(edad, posición, club, valor de mercado), contexto del partido (fecha, estadio, "
        "rival, fase, resultado) y métricas de ataque, pases, defensa, portería, carga "
        "física y puntajes de rendimiento. Algunas variables son **acumuladas del torneo** "
        "y no deben sumarse por partido."
    )
    st.info("👉 Ve a **Carga del dataset** en el menú lateral para comenzar.")


# -----------------------------------------------------------------------------
# 7. MÓDULO 2: CARGA DEL DATASET
# -----------------------------------------------------------------------------
def page_carga() -> None:
    st.title("📂 Carga del dataset")
    file = st.file_uploader("Sube el archivo `fifa_world_cup_2026_player_performance.csv`",
                            type="csv")
    use_local = False
    if LOCAL_DATASET.exists():
        use_local = st.checkbox("Usar el dataset incluido en el repositorio (carpeta data/)")

    source = file if file is not None else (LOCAL_DATASET if use_local else None)
    if source is None:
        if "analyzer" in st.session_state:
            st.success("✅ Ya hay un dataset cargado. Puedes ir al módulo de análisis.")
        else:
            st.info("Carga el archivo CSV para habilitar el análisis.")
        return

    try:
        df = load_csv(source)
    except Exception as exc:
        st.error(f"No se pudo leer el archivo: {exc}")
        return

    missing_cols = DataAnalyzer.validate(df)
    if missing_cols:
        st.error(f"El archivo no corresponde al dataset esperado. "
                 f"Faltan columnas: {', '.join(missing_cols)}")
        return

    st.session_state["analyzer"] = DataAnalyzer(df)
    st.success("✅ Archivo cargado y validado correctamente.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", f"{df.shape[0]:,}")
    c2.metric("Columnas", df.shape[1])
    c3.metric("Jugadores", f"{df['player_id'].nunique():,}")
    c4.metric("Selecciones", df["team"].nunique())

    n_rows = st.slider("Filas a mostrar en la vista previa", 5, 50, 5)
    st.dataframe(df.head(n_rows))


# -----------------------------------------------------------------------------
# 8. MÓDULO 3: EDA (10 ÍTEMS)
# -----------------------------------------------------------------------------
def item_1(an: DataAnalyzer) -> None:
    st.subheader("Ítem 1 · Información general del dataset")
    st.caption("Estructura, tipos de datos, valores nulos y duplicados.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Registros", f"{len(an.raw):,}")
    c2.metric("Valores nulos", an.null_count())
    c3.metric("Filas duplicadas", an.duplicated_count())

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.markdown("**Salida de `.info()`**")
        st.code(an.info_text(), language="text")
    with col_b:
        st.markdown("**Tipos de datos**")
        st.dataframe(an.dtypes_table(), hide_index=True)
    st.markdown(
        "El dataset está **completo y sin duplicados**. `match_date` llega como texto, "
        "por eso se convierte a fecha antes del análisis temporal."
    )


def item_2(an: DataAnalyzer) -> None:
    st.subheader("Ítem 2 · Clasificación de variables")
    st.caption("Usa la función personalizada `classify_columns()`.")
    groups = an.classify_variables()
    cols = st.columns(len(groups))
    for col, (name, variables) in zip(cols, groups.items()):
        col.metric(name, len(variables))
    tipo = st.selectbox("Ver variables del grupo", list(groups.keys()))
    st.dataframe(pd.DataFrame({"Variable": groups[tipo]}), hide_index=True, height=250)
    st.markdown(
        "Predominan las variables **numéricas** (métricas por partido y puntajes). Las "
        "**categóricas** describen al jugador y el contexto del partido. Ojo: `player_id`, "
        "`match_id` y `jersey_number` son numéricas pero funcionan como identificadores, "
        "y `clean_sheet` es un indicador binario."
    )


def item_3(an: DataAnalyzer, only_played: bool) -> None:
    st.subheader("Ítem 3 · Estadísticas descriptivas")
    st.caption("`.describe()` + mediana, moda, IQR y coeficiente de variación.")
    options = an.classify_variables()["Numéricas"]
    selected = st.multiselect("Variables a describir", options, default=KEY_NUMERIC)
    if not selected:
        st.warning("Selecciona al menos una variable.")
        return
    st.dataframe(an.describe_numeric(selected, only_played))

    st.markdown("**Detección preliminar de valores extremos (regla 1.5 × IQR)**")
    st.dataframe(an.outliers_iqr(selected, only_played), hide_index=True)

    comp = an.describe_numeric(["player_rating"], False).iloc[0]
    st.markdown(
        f"- Con **todos los registros**, `player_rating` tiene media **{comp['mean']:.2f}** "
        f"y mediana **{comp['mediana']:.2f}**: la media se hunde por los partidos en que "
        "el jugador no entró (rating = 0).\n"
        "- Con solo jugadores con minutos, media y mediana se acercan: la distribución es "
        "casi simétrica.\n"
        "- `pass_accuracy` tiene baja dispersión (CV pequeño): la precisión de pase es "
        "bastante homogénea entre jugadores."
    )


def item_4(an: DataAnalyzer) -> None:
    st.subheader("Ítem 4 · Análisis de valores faltantes")
    missing = an.missing_table()
    total = int(missing["Nulos"].sum())
    c1, c2 = st.columns(2)
    c1.metric("Total de valores nulos", total)
    c2.metric("Variables con nulos", int((missing["Nulos"] > 0).sum()))
    if st.checkbox("Ver tabla de nulos por variable"):
        st.dataframe(missing)

    zero = an.zero_minutes_summary()
    st.markdown(
        f"No hay valores nulos, así que **no se requiere imputación**. Sin embargo, existe "
        f"un \"faltante encubierto\": **{zero['n']:,} registros ({zero['pct']:.1f}%)** "
        "tienen `minutes_played = 0`. Son jugadores convocados que no entraron, y en esos "
        "registros las métricas de rendimiento y físicas valen 0."
    )
    data = an.df.groupby("position", observed=True)["played"].mean().mul(100).reindex(POSITION_ORDER)
    fig, ax = plt.subplots(figsize=(7, 3))
    sns.barplot(x=data.values, y=data.index, hue=data.index, legend=False, ax=ax)
    ax.set_xlabel("% de registros con minutos jugados")
    ax.set_ylabel("")
    ax.set_title("Participación efectiva por posición")
    show_fig(fig)
    st.markdown(
        "**Decisión:** se **conservan** todos los registros, pero los análisis de "
        "rendimiento usan por defecto solo los registros con minutos (checkbox del menú "
        "lateral). Así no se confunde \"no jugó\" con \"jugó mal\"."
    )


def item_5(an: DataAnalyzer, only_played: bool) -> None:
    st.subheader("Ítem 5 · Distribución de variables numéricas")
    c1, c2 = st.columns([2, 1])
    variable = c1.selectbox("Variable", KEY_NUMERIC)
    by_position = c2.checkbox("Separar por posición", value=True)
    show_fig(an.plot_hist(variable, by_position, only_played))

    data = an.data(only_played)[variable]
    st.markdown(
        f"Media **{data.mean():.2f}** · Mediana **{data.median():.2f}** · "
        f"Asimetría **{data.skew():.2f}** · Curtosis **{data.kurt():.2f}**"
    )
    st.markdown(
        "- Si se desmarca *Solo jugadores con minutos*, aparece un pico en 0: es la "
        "masa de suplentes que no jugó.\n"
        "- `distance_covered_km` depende de los minutos jugados y de la posición. Los "
        "porteros forman un grupo separado con valores mucho menores.\n"
        "- `top_speed_kmh` de los porteros también es menor, así que **compararlos con "
        "jugadores de campo sería inadecuado**. Por eso se separa por posición.\n"
        "- `pass_accuracy` se concentra en un rango estrecho, más alto en mediocampistas."
    )

    if st.checkbox("Ver métricas exclusivas de porteros"):
        gk = an.data(only_played)
        gk = gk[gk["position"] == "Goalkeeper"]
        st.dataframe(gk[GOALKEEPER_METRICS].describe().T.round(3))


def item_6(an: DataAnalyzer) -> None:
    st.subheader("Ítem 6 · Análisis de variables categóricas")
    cat_vars = ["position", "preferred_foot", "tournament_stage", "match_result",
                "team", "club_name", "city"]
    variable = st.selectbox("Variable categórica", cat_vars)
    freq = an.frequency_table(variable)
    if len(freq) > 15:
        top = st.slider("Mostrar las N categorías más frecuentes", 5, len(freq), 15)
        freq = freq.head(top)
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(freq)
    with c2:
        fig, ax = plt.subplots(figsize=(7, max(3.5, 0.3 * len(freq))))
        sns.barplot(x=freq["Proporción (%)"], y=freq.index.astype(str),
                    hue=freq.index.astype(str), legend=False, ax=ax)
        ax.set_ylabel("")
        ax.set_title(f"Proporción por {variable}")
        show_fig(fig)

    pos = an.frequency_table("position")["Proporción (%)"]
    foot = an.frequency_table("preferred_foot")["Proporción (%)"]
    st.markdown(
        f"- **Posiciones:** defensas {pos['Defender']:.1f}%, mediocampistas "
        f"{pos['Midfielder']:.1f}%, delanteros {pos['Forward']:.1f}% y porteros "
        f"{pos['Goalkeeper']:.1f}%, coherente con la composición de una plantilla.\n"
        f"- **Pie preferido:** {foot['Right']:.1f}% diestros frente a {foot['Left']:.1f}% zurdos.\n"
        "- **Fases:** la mayoría de registros es de fase de grupos, y la cantidad cae "
        "a medida que avanzan las rondas eliminatorias.\n"
        "- **Resultados:** victorias y derrotas tienen la misma frecuencia porque cada "
        "partido produce un ganador y un perdedor."
    )


def item_7(an: DataAnalyzer, only_played: bool) -> None:
    st.subheader("Ítem 7 · Análisis bivariado: numérico vs categórico")
    t1, t2, t3, t4 = st.tabs(["Rating por posición", "Score por resultado",
                              "Físico por posición", "Producción por 90'"])
    with t1:
        show_fig(an.plot_box("player_rating", "position", POSITION_ORDER, only_played))
        st.dataframe(an.group_stats("player_rating", "position", only_played))
        st.markdown("Las medianas de rating por posición son muy parecidas: el rating "
                    "está **normalizado por rol** y sirve para comparar entre posiciones.")
    with t2:
        show_fig(an.plot_box("performance_score", "match_result", RESULT_ORDER, only_played))
        st.dataframe(an.group_stats("performance_score", "match_result", only_played))
        st.markdown("Los jugadores de equipos **ganadores** obtienen un performance_score "
                    "levemente superior; empates y derrotas son prácticamente iguales.")
    with t3:
        metric = st.selectbox("Métrica física", ["distance_covered_km", "top_speed_kmh",
                                                 "sprint_distance_km", "stamina_score"])
        show_fig(an.plot_box(metric, "position", POSITION_ORDER, only_played))
        st.dataframe(an.group_stats(metric, "position", only_played))
    with t4:
        per90 = an.per90(["goals", "assists", "shots", "key_passes", "tackles",
                          "interceptions", "defensive_actions"]).reindex(POSITION_ORDER)
        st.dataframe(per90)
        fig, ax = plt.subplots(figsize=(7, 3.5))
        per90[["goals", "assists"]].plot(kind="barh", ax=ax)
        ax.set_title("Goles y asistencias por 90 minutos")
        show_fig(fig)
        st.markdown("Normalizar por 90 minutos permite comparar jugadores con distinto "
                    "tiempo en cancha. Aquí sí se ven **diferencias claras por rol**.")


def item_8(an: DataAnalyzer) -> None:
    st.subheader("Ítem 8 · Análisis bivariado: categórico vs categórico")
    normalize = st.checkbox("Mostrar porcentajes en lugar de conteos", value=True)
    t1, t2, t3 = st.tabs(["Posición vs fase", "Equipo vs resultado", "Pie vs posición"])
    with t1:
        ct = an.crosstab("tournament_stage", "position",
                         normalize="index" if normalize else False).reindex(STAGE_ORDER)
        show_fig(an.plot_heatmap(ct[POSITION_ORDER], "Posición según fase del torneo",
                                 ".1f" if normalize else "d"))
        st.markdown("La proporción de cada posición se mantiene estable en todas las "
                    "fases: las plantillas no cambian su composición al avanzar.")
    with t2:
        results = an.team_results()
        top = st.slider("Selecciones a mostrar", 5, len(results), 15)
        show_fig(an.plot_heatmap(results.head(top)[RESULT_ORDER],
                                 f"Resultados por selección (top {top} por % de victorias)",
                                 "d"))
        st.dataframe(results.head(top))
        st.caption("Los resultados se cuentan **por partido**, no por jugador, para no "
                   "inflar los conteos con cada jugador convocado.")
    with t3:
        ct = an.crosstab("preferred_foot", "position",
                         normalize="columns" if normalize else False)
        show_fig(an.plot_heatmap(ct[POSITION_ORDER], "Pie preferido según posición",
                                 ".1f" if normalize else "d"))
        st.markdown("Los diestros dominan en todas las posiciones. La proporción de "
                    "zurdos es algo mayor en defensas y delanteros.")


def item_9(an: DataAnalyzer, only_played: bool) -> None:
    st.subheader("Ítem 9 · Explorador dinámico por parámetros")
    df = an.data(only_played)

    with st.expander("🎛️ Filtros", expanded=True):
        c1, c2, c3 = st.columns(3)
        teams = c1.multiselect("Selección", sorted(df["team"].unique()))
        positions = c2.multiselect("Posición", POSITION_ORDER)
        stages = c3.multiselect("Fase", STAGE_ORDER)
        c4, c5 = st.columns([1, 2])
        results = c4.multiselect("Resultado", RESULT_ORDER,
                                 format_func=lambda r: RESULT_LABELS[r])
        pool = df if not teams else df[df["team"].isin(teams)]
        players = c5.multiselect("Jugador", sorted(pool["player_name"].unique()))
        min_lo, min_hi = st.slider("Minutos jugados", 0, 90, (0 if not only_played else 1, 90))
        age_lo, age_hi = st.slider("Edad", int(df["age"].min()), int(df["age"].max()),
                                   (int(df["age"].min()), int(df["age"].max())))

    filtered = an.filter_data(
        {"team": teams, "position": positions, "tournament_stage": stages,
         "match_result": results, "player_name": players},
        {"minutes_played": (min_lo, min_hi), "age": (age_lo, age_hi)},
        only_played,
    )
    if filtered.empty:
        st.warning("No hay registros con esos filtros.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Registros", f"{len(filtered):,}")
    c2.metric("Jugadores", filtered["player_id"].nunique())
    c3.metric("Rating medio", fmt_num(filtered["player_rating"].mean()))
    c4.metric("Goles", int(filtered["goals"].sum()))

    group = st.selectbox("Tipo de métricas", list(METRIC_GROUPS.keys()))
    metric = st.selectbox("Métrica", METRIC_GROUPS[group])

    tab_a, tab_b, tab_c = st.tabs(["Ranking de jugadores", "Comparar grupos", "Evolución temporal"])
    with tab_a:
        cc1, cc2 = st.columns(2)
        min_matches = cc1.slider("Mínimo de partidos jugados", 1, 15, 3)
        top_n = cc2.slider("Top N", 5, 30, 10)
        ranking = an.player_ranking(filtered, metric, min_matches, top_n)
        if ranking.empty:
            st.info("Ningún jugador cumple el mínimo de partidos.")
        else:
            fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(ranking))))
            sns.barplot(data=ranking, x="promedio", y="player_name", hue="position",
                        dodge=False, ax=ax)
            ax.set_xlabel(f"{metric} promedio por partido")
            ax.set_ylabel("")
            ax.set_title(f"Top {len(ranking)} jugadores en {metric}")
            show_fig(fig)
            st.dataframe(ranking, hide_index=True)
    with tab_b:
        by = st.selectbox("Comparar por", ["position", "match_result", "tournament_stage", "team"])
        stats = (filtered.groupby(by, observed=True)[metric]
                 .agg(["count", "mean", "median", "std"]).round(3)
                 .sort_values("mean", ascending=False))
        st.dataframe(stats)
        fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(stats))))
        sns.barplot(x=stats["mean"], y=stats.index.astype(str),
                    hue=stats.index.astype(str), legend=False, ax=ax)
        ax.set_xlabel(f"{metric} (media)")
        ax.set_ylabel("")
        show_fig(fig)
    with tab_c:
        daily = filtered.groupby("match_date")[metric].mean()
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(daily.index, daily.values, marker="o", ms=3)
        ax.set_title(f"Evolución de {metric} a lo largo del torneo")
        ax.set_xlabel("Fecha del partido")
        fig.autofmt_xdate()
        show_fig(fig)
        st.caption("`match_date` se convierte a datetime al cargar el dataset.")

    if st.checkbox("Ver registros filtrados"):
        st.dataframe(filtered.head(500))


def item_10(an: DataAnalyzer) -> None:
    st.subheader("Ítem 10 · Hallazgos clave")
    k = an.key_metrics()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    k["rating_pos"].reindex(POSITION_ORDER).plot(kind="bar", ax=axes[0, 0], color="#4C72B0")
    axes[0, 0].set_title("Rating medio por posición")
    axes[0, 0].set_ylim(5.5, 6.6)
    k["rating_res"].reindex(RESULT_ORDER).rename(RESULT_LABELS).plot(
        kind="bar", ax=axes[0, 1], color=["#55A868", "#8C8C8C", "#C44E52"])
    axes[0, 1].set_title("Rating medio según resultado")
    axes[0, 1].set_ylim(5.5, 6.6)
    k["per90"].reindex(POSITION_ORDER)[["goals", "assists"]].plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Goles y asistencias por 90'")
    k["corr"].plot(kind="barh", ax=axes[1, 1], color="#8172B2")
    axes[1, 1].set_title("Correlación con player_rating")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=0)
        ax.set_xlabel("")
    fig.tight_layout()
    show_fig(fig)

    best_pos = k["rating_pos"].index[0]
    top_corr = k["corr"].index[0]
    st.markdown(
        f"1. **{k['zero']['pct']:.1f}% de los registros son de jugadores sin minutos.** "
        f"El rating medio pasa de {k['rating_all']:.2f} (todos) a "
        f"{k['rating_played']:.2f} (solo con minutos).\n"
        f"2. **El rating está equilibrado entre posiciones**: el máximo es de "
        f"{best_pos} ({k['rating_pos'].iloc[0]:.2f}) y el mínimo de "
        f"{k['rating_pos'].index[-1]} ({k['rating_pos'].iloc[-1]:.2f}).\n"
        f"3. **Ganar sube el rating**: {k['rating_res']['W']:.2f} en victorias frente a "
        f"{k['rating_res']['L']:.2f} en derrotas.\n"
        f"4. **Los roles se notan en la producción por 90'**: los delanteros marcan "
        f"{k['per90'].loc['Forward', 'goals']:.2f} goles/90 y los defensas acumulan "
        f"{k['per90'].loc['Defender', 'defensive_actions']:.1f} acciones defensivas/90.\n"
        f"5. **`{top_corr}` es la variable más asociada al rating** "
        f"(r = {k['corr'].iloc[0]:.2f}), mientras que la distancia recorrida casi no se "
        f"relaciona con él (r = {k['corr']['distance_covered_km']:.2f})."
    )
    st.info("**Recomendaciones de interpretación:** filtrar siempre a jugadores con "
            "minutos, comparar porteros solo entre porteros, usar métricas por 90' para "
            "comparar jugadores con distinto tiempo en cancha y no sumar variables "
            "acumuladas del torneo registro por registro.")

    if st.checkbox("Ver resumen acumulado del torneo (un registro por jugador)"):
        snap = an.tournament_snapshot().sort_values("tournament_rating", ascending=False)
        st.dataframe(snap.head(20), hide_index=True)


def page_eda() -> None:
    st.title("🔎 Análisis Exploratorio de Datos")
    an = st.session_state.get("analyzer")
    if an is None:
        st.warning("⚠️ Primero carga el dataset en el módulo **Carga del dataset**.")
        return

    only_played = st.sidebar.checkbox(
        "Solo jugadores con minutos", value=True,
        help="Excluye los registros con minutes_played = 0 (suplentes que no entraron).")

    tabs = st.tabs(["1 · Info", "2 · Variables", "3 · Descriptivas", "4 · Faltantes",
                    "5 · Distribuciones", "6 · Categóricas", "7 · Num vs Cat",
                    "8 · Cat vs Cat", "9 · Explorador", "10 · Hallazgos"])
    with tabs[0]:
        item_1(an)
    with tabs[1]:
        item_2(an)
    with tabs[2]:
        item_3(an, only_played)
    with tabs[3]:
        item_4(an)
    with tabs[4]:
        item_5(an, only_played)
    with tabs[5]:
        item_6(an)
    with tabs[6]:
        item_7(an, only_played)
    with tabs[7]:
        item_8(an)
    with tabs[8]:
        item_9(an, only_played)
    with tabs[9]:
        item_10(an)


# -----------------------------------------------------------------------------
# 9. MÓDULO 4: CONCLUSIONES
# -----------------------------------------------------------------------------
def page_conclusiones() -> None:
    st.title("✅ Conclusiones")
    an = st.session_state.get("analyzer")
    if an is None:
        st.warning("⚠️ Primero carga el dataset en el módulo **Carga del dataset**.")
        return
    k = an.key_metrics()
    per90 = k["per90"]

    conclusions = [
        ("Filtrar por participación es obligatorio",
         f"El {k['zero']['pct']:.1f}% de los registros corresponde a jugadores que no "
         f"entraron al campo. Incluirlos baja el rating medio de {k['rating_played']:.2f} "
         f"a {k['rating_all']:.2f}. Cualquier reporte de rendimiento debe partir de "
         "registros con minutos jugados.",
         "Ítems 3 y 4"),
        ("El rating permite comparar entre posiciones",
         f"Las medias por posición van de {k['rating_pos'].min():.2f} a "
         f"{k['rating_pos'].max():.2f}. La diferencia es mínima, así que el rating sirve "
         "como métrica común. Para evaluar el aporte específico de cada rol conviene "
         "complementarlo con métricas propias de la posición.",
         "Ítem 7 · Rating por posición"),
        ("El resultado del partido condiciona la evaluación individual",
         f"El rating medio es {k['rating_res']['W']:.2f} en victorias, "
         f"{k['rating_res']['D']:.2f} en empates y {k['rating_res']['L']:.2f} en derrotas. "
         "Al evaluar a un jugador hay que considerar el contexto del equipo para no "
         "castigar actuaciones individuales buenas en partidos perdidos.",
         "Ítem 7 · Score por resultado"),
        ("Las métricas por 90 minutos revelan el rol real",
         f"Los delanteros producen {per90.loc['Forward', 'goals']:.2f} goles/90 frente a "
         f"{per90.loc['Defender', 'goals']:.2f} de los defensas, mientras que los defensas "
         f"registran {per90.loc['Defender', 'defensive_actions']:.1f} acciones defensivas/90 "
         f"frente a {per90.loc['Forward', 'defensive_actions']:.1f} de los delanteros. "
         "Para scouting o selección conviene comparar por 90' y dentro de la misma posición.",
         "Ítem 7 · Producción por 90'"),
        ("El rating se explica por el aporte con balón, no por el esfuerzo físico",
         f"`{k['corr'].index[0]}` tiene la mayor correlación con el rating "
         f"(r = {k['corr'].iloc[0]:.2f}), mientras que la distancia recorrida casi no se "
         f"asocia (r = {k['corr']['distance_covered_km']:.2f}). Correr más no garantiza "
         "una mejor calificación; las métricas físicas deben leerse como carga de "
         "trabajo, no como calidad.",
         "Ítem 10 · Correlaciones"),
    ]
    for i, (title, text, evidence) in enumerate(conclusions, start=1):
        with st.container(border=True):
            st.markdown(f"### {i}. {title}")
            st.markdown(text)
            st.caption(f"📎 Evidencia: {evidence}")


def main() -> None:
    st.sidebar.title("⚽ WC 2026 EDA")
    page = st.sidebar.radio(
        "Navegación",
        ["Home", "Carga del dataset", "Análisis exploratorio", "Conclusiones"],
    )
    loaded = "analyzer" in st.session_state
    st.sidebar.markdown("---")
    st.sidebar.caption("Dataset: " + ("✅ cargado" if loaded else "⏳ pendiente"))

    if page == "Home":
        page_home()
    elif page == "Carga del dataset":
        page_carga()
    elif page == "Análisis exploratorio":
        page_eda()
    else:
        page_conclusiones()

    st.sidebar.markdown("---")
    st.sidebar.caption(f"{AUTOR} · {AÑO}")


if __name__ == "__main__":
    main()
