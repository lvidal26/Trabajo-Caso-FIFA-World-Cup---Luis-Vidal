# FIFA World Cup 2026 – Player Performance EDA

App interactiva en **Streamlit** para el análisis exploratorio del rendimiento de jugadores del Mundial 2026. No construye modelos predictivos: describe y compara por posición, equipo, fase y jugador.

![Home](https://github.com/lvidal26/Trabajo-Caso-FIFA-World-Cup---Luis-Vidal/blob/38c43bc6cd0b71a7f72bb404fc07078b54a8b853/home.png)

## Módulos

| Módulo | Contenido |
|---|---|
| **Home** | Proyecto, autor, dataset y tecnologías |
| **Carga del dataset** | Subida del CSV, validación, dimensiones y vista previa |
| **Análisis exploratorio** | 10 ítems en tabs: info general, clasificación de variables, descriptivas, faltantes, distribuciones, categóricas, análisis bivariado, explorador con filtros y hallazgos |
| **Conclusiones** | 5 conclusiones con la evidencia que las respalda |

## Carga DataSet
![Carga](https://github.com/lvidal26/Trabajo-Caso-FIFA-World-Cup---Luis-Vidal/blob/961ed405e7a36628c8ec2452dded6644f3e3ab94/carga_dataset.png)

## Analisis Exploratorio
![EDA](https://github.com/lvidal26/Trabajo-Caso-FIFA-World-Cup---Luis-Vidal/blob/76a8bb9991676240006e25f986f68832d4eead0e/Eda.png)

## Explorador Dinamico
![Filtros](https://github.com/lvidal26/Trabajo-Caso-FIFA-World-Cup---Luis-Vidal/blob/53d222fa42dfdfe02045f1798519bcef38fa6101/Filtros.png) 

## Conclusiones
![Conclusiones](https://github.com/lvidal26/Trabajo-Caso-FIFA-World-Cup---Luis-Vidal/blob/78bdcd37ab7d1e1e4a4ad8cdbf4dd8a039420da9/Conclusiones.png)

## Dataset

`fifa_world_cup_2026_player_performance.csv`: **54,600 registros × 75 variables**, con 1,248 jugadores, 1,050 partidos y 48 selecciones. Cada fila es la actuación de un jugador en un partido. No tiene nulos ni duplicados.

**Variables principales:** `position`, `team`, `tournament_stage`, `match_result`, `minutes_played`, `player_rating`, `performance_score`, `goals`, `assists`, `pass_accuracy`, `defensive_actions`, `distance_covered_km`, `top_speed_kmh` y `market_value_eur`.

**Hallazgos:**
- El 42.2% de los registros tiene 0 minutos jugados (suplentes). Incluirlos baja el rating medio de 6.29 a 3.63.
- El rating es parejo entre posiciones (de 6.21 a 6.36) y sube en victorias (6.38 frente a 6.23 en derrotas).
- Por cada 90 minutos, los delanteros marcan 0.34 goles y los defensas suman 10.9 acciones defensivas.
- El rating se asocia al valor de mercado (r = 0.41), no a la distancia recorrida (r = 0.02).

##  Ejecución local

```bash
git clone https://github.com/TU_USUARIO/trabajo-caso-fifa-world-cup---luis-vidal.git
cd trabajo-caso-fifa-world-cup---luis-vidal
pip install -r requirements.txt
streamlit run app.py
```

Después, en **Carga del dataset**, sube el CSV o usa el que viene en `data/`.

##  Tecnologías

Python · Pandas · NumPy · Altair · Streamlit · POO (clase `DataAnalyzer`)

##  Autor

**Luis Aaron Vidal Vela**
Especialización en Python for Analytics – DMC Institute · 2026
