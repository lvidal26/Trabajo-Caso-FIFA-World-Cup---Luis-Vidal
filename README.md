# ⚽ FIFA World Cup 2026 – Player Performance EDA

Aplicación interactiva construida con **Python y Streamlit** para el **Análisis Exploratorio de Datos (EDA)** del rendimiento de jugadores y selecciones en el FIFA World Cup 2026.

El objetivo **no es predecir resultados**, sino explorar el rendimiento con métricas técnicas, ofensivas, defensivas, físicas y contextuales de cada partido, y comparar por **posición, equipo, fase del torneo y jugador**.
---

##  Contenido de la aplicación

La app tiene 4 módulos que se eligen desde el **sidebar**:

| Módulo | Descripción |
|---|---|
| **Home** | Presentación del proyecto, autor, dataset y tecnologías |
| **Carga del dataset** | Carga del CSV con `st.file_uploader`, validación de columnas, vista previa y dimensiones |
| **Análisis exploratorio** | 10 ítems de análisis organizados en tabs (se habilita después de cargar el CSV) |
| **Conclusiones** | 5 conclusiones calculadas desde los datos, cada una con el ítem que la respalda |

### Ítems del EDA

1. **Información general:** `.info()`, tipos de datos, nulos y duplicados
2. **Clasificación de variables:** numéricas, categóricas y de fecha, con la función `classify_columns()`
3. **Estadísticas descriptivas:** media, mediana, moda, desviación, IQR, CV y outliers por IQR
4. **Valores faltantes:** tabla de nulos y análisis de los registros con 0 minutos jugados
5. **Distribución de variables numéricas:** histogramas, con opción de separar por posición
6. **Variables categóricas:** frecuencias, proporciones y gráficos de barras
7. **Numérico vs categórico:** rating por posición, score por resultado, métricas físicas y producción por 90'
8. **Categórico vs categórico:** heatmaps de posición vs fase, equipo vs resultado y pie vs posición
9. **Explorador dinámico:** filtros por selección, posición, fase, resultado, jugador, minutos y edad; ranking de jugadores, comparación de grupos y evolución temporal
10. **Hallazgos clave:** resumen visual y recomendaciones de interpretación

---

## Capturas

### Home
![Home](img/home.png)

### Carga del dataset
![Carga](img/carga_dataset.png)

### Análisis exploratorio
![EDA](img/Eda.png)

### Explorador dinámico
![Filtros](img/Filtros.png)

### Conclusiones
![Conclusiones](img/Conclusiones.png)

---

## Estructura del proyecto

```
trabajo-caso-fifa-world-cup---luis-vidal/
├── app.py                 # Aplicación Streamlit
├── requirements.txt       # Dependencias
├── README.md
├── .gitignore
├── data/
│   └── fifa_world_cup_2026_player_performance.csv
└── img/                   # Capturas para el README
```

---

##  Cómo ejecutar el proyecto

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/trabajo-caso-fifa-world-cup---luis-vidal.git
   cd trabajo-caso-fifa-world-cup---luis-vidal
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # Mac/Linux
   ```

3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Ejecutar la aplicación:
   ```bash
   streamlit run app.py
   ```

5. En el módulo **Carga del dataset**, sube el CSV o marca la opción *Usar el dataset incluido en el repositorio*.

---

##  Dataset

`fifa_world_cup_2026_player_performance.csv` tiene **54,600 registros y 75 variables**. Cada fila es la actuación de un jugador en un partido.

- **1,248** jugadores · **1,050** partidos · **48** selecciones
- Sin valores nulos ni filas duplicadas

### Variables principales

| Grupo | Variables | Descripción |
|---|---|---|
| Jugador | `player_name`, `age`, `position`, `team`, `preferred_foot`, `market_value_eur` | Datos personales y de selección |
| Partido | `match_date`, `tournament_stage`, `match_result`, `opponent_team`, `stadium` | Contexto del encuentro |
| Ofensivas | `goals`, `assists`, `shots`, `expected_goals_xg`, `key_passes` | Producción en ataque |
| Pases | `total_passes`, `successful_passes`, `pass_accuracy` | Circulación del balón |
| Defensivas | `tackles`, `interceptions`, `clearances`, `recoveries`, `defensive_actions` | Recuperación y contención |
| Portería | `saves`, `save_percentage`, `clean_sheet`, `goals_conceded` | Métricas exclusivas de porteros |
| Físicas | `distance_covered_km`, `sprint_distance_km`, `top_speed_kmh`, `stamina_score` | Carga física |
| Rendimiento | `player_rating`, `performance_score`, `creativity_score`, `clutch_performance_score` | Puntajes de evaluación |
| Acumuladas | `total_goals_tournament`, `total_minutes_tournament`, `tournament_rating` | Totales del torneo (no se suman por partido) |

### Consideraciones del análisis

- El **42.2% de los registros (23,042)** tiene `minutes_played = 0`. Son suplentes que no entraron y sus métricas valen 0. Por eso el análisis de rendimiento usa por defecto solo los registros con minutos (checkbox *Solo jugadores con minutos* del sidebar).
- `match_date` se convierte a tipo fecha antes del análisis temporal.
- Las métricas de porteros se analizan aparte de las de los jugadores de campo.
- Las variables acumuladas del torneo se toman una sola vez por jugador (de su último registro) para no duplicar los totales.
- La producción se compara **por cada 90 minutos** jugados.
- Los resultados por selección se cuentan **por partido**, no por jugador.

---

## Principales hallazgos

1. Al incluir a los jugadores sin minutos, el rating medio baja de **6.29 a 3.63**, así que filtrar por participación es obligatorio.
2. El rating está equilibrado entre posiciones: va de 6.21 (porteros) a 6.36 (delanteros).
3. Ganar sube el rating: **6.38** en victorias frente a **6.23** en derrotas.
4. Los roles se notan en la producción por 90': los delanteros marcan **0.34 goles/90** y los defensas suman **10.9 acciones defensivas/90**.
5. El rating se asocia más al valor de mercado (r = 0.41) y al aporte con balón que a la distancia recorrida (r = 0.02).

---

##Tecnologías utilizadas

- **Python 3**
- **Pandas** y **NumPy**: manipulación y análisis de datos
- **Altair**: visualizaciones interactivas
- **Streamlit**: interfaz y despliegue en Streamlit Community Cloud
- **Programación Orientada a Objetos**: la clase `DataAnalyzer` encapsula la validación, las estadísticas, los gráficos y los filtros

---

##Autor

**Luis Aaron Vidal Vela**
Especialización en Python for Analytics – DMC Institute
2026
