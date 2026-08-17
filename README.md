# Ortoph Gemini Test: Visual Odometry & Matching

Este proyecto proporciona un pipeline completo para la simulación de vuelos de drones sobre ortofotografías de alta resolución, la generación y emparejamiento de teselas (imágenes recortadas) mediante algoritmos clásicos y profundos, y la evaluación del error en la estimación de la posición (Odometría Visual).

## 🚀 Características Principales

*   **Simulación de Vuelo:** Genera imágenes simuladas desde la perspectiva de un dron siguiendo trayectorias configurables (línea recta, curva, lemniscata) con opciones de altura, rotación, velocidad y ruido.
*   **Procesamiento de Ortofotografías:** Trocea grandes imágenes GeoTIFF en teselas (tiles) más manejables para crear bases de datos de búsqueda.
*   **Emparejamiento (Matching) Modular:** Soporte integrado para SIFT, ORB y métodos profundos basados en Kornia (SuperPoint + LightGlue).
*   **Odometría y Refinamiento:** Estimación de posición a partir de la matriz fundamental y filtrado de correspondencias aberrantes mediante RANSAC y análisis espacial.
*   **Evaluación y Métricas:** Cálculos estadísticos (RMSE, MAE, MEE, CEP) con eliminación de outliers basada en rangos intercuartílicos (IQR) y generación automática de visualizaciones comparativas.
*   **Pipeline Automatizado:** CLI amigable para automatizar el ciclo completo de simulación, matching y evaluación.

---

## 📂 Estructura del Proyecto

El proyecto está diseñado siguiendo principios de modularidad y bajo acoplamiento:

*   `cli/`: Herramienta unificada de comandos. Contiene módulos separados para `tile`, `simulate`, `match`, `evaluate`, y `pipeline`.
*   `core/`: Lógica de negocio (núcleo matemático y algorítmico).
    *   `evaluation/`: Métricas estadísticas y ejecución de la evaluación de error.
    *   `matching/`: Extractores de características y emparejadores.
    *   `processing/`: Procesamiento de imágenes (tiling).
    *   `simulation/`: Generación de trayectorias y captura de cámara del dron virtual.
*   `visualization/`: Módulos para generar gráficas 2D y 3D, y pintar las trayectorias sobre la ortofoto.
*   `utils/`: Funciones auxiliares como lectura y escritura segura de JSON.
*   `cli.py`: Punto de entrada principal para el uso del proyecto.

> Para más detalles técnicos, revisa el archivo `PROJECT_STRUCTURE.md`.

---

## 💻 Uso de la Interfaz de Línea de Comandos (CLI)

El proyecto utiliza un único entry-point `cli.py` con sub-comandos para cada funcionalidad. 
Puedes usar `python cli.py <comando> -h` para ver la ayuda específica de cada uno.

### 1. Trocear Ortofoto (`tile`)
Divide una ortofoto en secciones (X por Y) para construir la base de datos de búsqueda.
```bash
python cli.py tile -i ortophotos/mi_mapa.tif -x 10 -y 10
```

### 2. Simular Vuelo (`simulate`)
Genera la trayectoria de un dron y recorta las imágenes orientadas paso a paso.
```bash
python cli.py simulate -o ortophotos/mi_mapa.tif -j ortophotos/sections/mi_mapa_metadata.json -n "vuelo_1" -t curve --speed 10.0 --out-dir trajectories/mis_vuelos
```
*(Nota: Usa el parámetro opcional `--out-dir` para guardar la simulación en un directorio personalizado en vez de usar la ruta por defecto `trajectories/<nombre>`)*

### 3. Emparejamiento (`match`)
Busca cada imagen de la trayectoria simulada dentro de la base de datos de teselas usando algoritmos clásicos o de Deep Learning.
```bash
python cli.py match -t trajectories/vuelo_1 -d ortophotos/sections/mi_mapa_metadata.json -m superpoint_lightglue
```

### 4. Evaluación (`evaluate`)
Compara la posición real (Ground Truth) con la posición estimada y calcula las métricas de error.
```bash
python cli.py evaluate -t trajectories/vuelo_1 -mod single -m superpoint_lightglue --remove-outliers -s median
```

> **NOTA SOBRE EL CÁLCULO DE ERROR (SHIFT):** 
> Los métodos de matching (como SIFT o ORB) tienden a presentar un sesgo global sistemático debido a cómo calculan la transformación geométrica sobre la ortofoto. Para realizar mediciones más exactas y evaluar la dispersión real de las estimaciones independientemente de su sesgo, se recomienda utilizar el parámetro `--shift` (o `-s`). 
> - Usar `-s mean` o `-s median` extraerá automáticamente la tendencia global del error en los ejes X e Y, calculando una única corrección matemática en base a todas las muestras.
> - Este desplazamiento se resta a los datos unificando el centro de la matriz de puntos en el origen (0,0), lo que permite cuantificar con gran precisión la calidad del emparejamiento (por ejemplo, el radio en el que caen el 50% de las muestras - métrica CEP) sin que el sesgo local arruine los promedios globales.

### 5. Pipeline Automatizado (`pipeline`)
Ejecuta los pasos 2, 3 y 4 de forma consecutiva. Útil para lanzar múltiples pruebas automatizadas (familias de vuelos).
```bash
python cli.py pipeline -o mi_mapa -t lemniscate -m sift --runs 5
```

---

## 🛠 Instalación y Requisitos

Requiere **Python 3.8+**. Se recomienda usar un entorno virtual.

```bash
pip install -r requirements.txt
```

Dependencias principales:
*   `numpy`, `opencv-python` (cv2), `matplotlib`, `scipy`
*   `torch`, `kornia` (Para el matcher de SuperPoint + LightGlue)
*   `pyproj`, `rasterio` (Para el procesamiento geoespacial)
