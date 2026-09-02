# UAV Visual Localization

Pipeline completo de **localización y odometría visual** para drones (UAV) que vuelan sobre **ortofotografías georreferenciadas de alta resolución**. El proyecto abarca todo el ciclo: simulación de vuelo, troceado de la ortofoto en teselas, emparejamiento de imágenes con algoritmos clásicos y de *deep learning*, y evaluación estadística del error de posición frente a la trayectoria real (*ground truth*).

---

## 📌 Descripción y propósito

Este proyecto es el Trabajo de Fin de Grado de Pablo Nieto Arias para el Grado en Ingeniería Informática de la Universidad de Alcalá de Henares. Tiene como objetivo **estudiar y comparar la precisión de distintos algoritmos de emparejamiento visual** cuando un dron debe localizarse sobre un mapa aéreo de referencia:

1. Se **simula** el vuelo de un dron sobre una ortofoto, generando imágenes sintéticas con la perspectiva y orientación correctas.
2. Cada imagen capturada se **empareja** contra una base de datos de teselas de la ortofoto para estimar dónde se encuentra el dron.
3. Se **evalúa** el error entre la posición estimada y la posición real, produciendo métricas y gráficas comparativas entre métodos.

Los métodos comparados son **SIFT**, **ORB** (clásicos, basados en OpenCV) y **SuperPoint + LightGlue** (basados en redes profundas), lo que permite analizar el equilibrio entre precisión, robustez y tiempo de cómputo en un problema real de localización visual.

---

## 📂 Estructura del proyecto

El código está organizado siguiendo principios de **modularidad y bajo acoplamiento**: la interfaz de usuario está separada del núcleo algorítmico, y cada responsabilidad vive en su propio módulo.

```text
uav-visual-localization/
│
├── cli.py                          # Entry-point principal (CLI unificada)
├── cli/                            # Capa de interfaz de comandos
│   └── commands/
│       ├── tile.py                 # Comando: dividir ortofoto en teselas
│       ├── simulate.py             # Comando: simular vuelo de dron
│       ├── match.py                # Comando: emparejar contra la BD de teselas
│       ├── evaluate.py             # Comando: evaluar error y generar gráficas
│       └── pipeline.py             # Comando: pipeline completo automatizado
│
├── core/                           # Núcleo de la lógica de negocio
│   ├── evaluation/
│   │   ├── metrics.py              # RMSE, MAE, MEE, CEP, filtros IQR/MAD, shift
│   │   └── runner.py               # Clase Evaluator (single / batch / familias)
│   ├── matching/
│   │   ├── extractors.py           # Extractores de características (SIFT, ORB)
│   │   ├── matchers.py             # ClassicalMatcher y DeepExtractorMatcher
│   │   ├── batch_processor.py      # Orquestador BatchMatcher para flujos masivos
│   │   ├── odometry.py             # Odometría visual entre frames consecutivos
│   │   └── location_refinement.py  # Refinamiento de localización (DBSCAN)
│   ├── processing/
│   │   └── tiling.py               # Troceado de GeoTIFFs grandes en teselas
│   └── simulation/
│       ├── camera.py               # Cámara virtual orientada (recortes sintéticos)
│       ├── flight.py               # Simulador de secuencias completas de vuelo
│       └── paths.py                # Rutas: recta, curva (Bézier), lemniscata
│
├── visualization/                  # Generación de gráficas 2D/3D y mapas
│   ├── maps.py                     # Trayectorias sobre la ortofoto (top-down)
│   ├── evaluation.py               # Gráficas de error para el Evaluator
│   ├── refinement.py               # Inliers vs outliers
│   ├── charts_3d.py                # Ploteos 3D
│   └── ...                         # Scripts auxiliares de visualización
│
├── utils/
│   └── io.py                       # Lectura/escritura segura de JSON y ficheros
│
├── ortophotos/                     # Ortofotos GeoTIFF + bases de datos de teselas
│   └── .gitkeep                    # Solo placeholder en el repositorio (datos locales)
│
├── trajectories/                   # Resultados de simulación/emparejamiento
│   └── .gitkeep                    # Solo placeholder en el repositorio (datos locales)
│
├── showcase_trajectory/            # Ejemplo de trayectoria de demostración
├── environment.yml                 # Dependencias (conda)
└── PROJECT_STRUCTURE.md            # Descripción pormenorizada de la arquitectura
```

> Para conocer la responsabilidad de cada archivo en detalle, consulta [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).
>
> **Nota sobre datos:** los directorios `ortophotos/` y `trajectories/` se conservan en el repositorio únicamente con un `.gitkeep`. Las ortofotos y los resultados de simulación/emparejamiento son **datos locales no versionados** que debes aportar o generar al ejecutar el pipeline.

---

## 🛠 Guía de instalación

### Requisitos

* **Python 3.11** (recomendado, definido en `environment.yml`).
* [Miniconda / Anaconda](https://docs.conda.io/) (recomendado).
* *Opcional:* GPU NVIDIA con CUDA para acelerar el método **SuperPoint + LightGlue**.

### Instalación con Conda (recomendada)

Las dependencias están declaradas en `environment.yml`. Crea el entorno (por defecto se llamará `TFG`, puedes cambiarlo con `-n`):

```bash
conda env create -f environment.yml
conda activate uav-visloc
```

Este proceso instala automáticamente, entre otros:

* `numpy`, `opencv-python`, `matplotlib`, `scipy`, `scikit-learn`, `Pillow`, `tqdm`
* `pytorch`, `torchvision`, `pytorch-cuda` (para el matcher profundo)
* `kornia` y `lightglue` (desde el repositorio `cvg/LightGlue`, necesario para SuperPoint + LightGlue)
* `rasterio` (procesamiento geoespacial de las ortofotos GeoTIFF)

### Instalación alternativa con `pip`

Si prefieres `pip` + entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy opencv-python "matplotlib<3.9" scipy scikit-learn \
            torch torchvision kornia rasterio Pillow tqdm
pip install "lightglue @ git+https://github.com/cvg/LightGlue.git"
```

> **Nota:** algunos métodos (*SuperPoint/LightGlue*) requieren `torch`/`kornia` y solo estarán disponibles si se instalaron correctamente. El resto de comandos funcionan sin ellos.

---

## 🧭 Guía de usuario

Todo el proyecto se controla desde un único punto de entrada con **subcomandos**:

```bash
python cli.py <comando> -h     # Ayuda específica de cada comando
python cli.py -h               # Vista general de todos los comandos
```

Comandos disponibles: `tile`, `simulate`, `match`, `evaluate` y `pipeline`.

### Flujo de trabajo típico (paso a paso)

#### 1. Trocear la ortofoto (`tile`)
Divide un GeoTIFF grande en una cuadrícula de teselas (X×Y) que formarán la **base de datos de búsqueda**, generando además el JSON de metadatos necesario para el resto del proceso.

```bash
python cli.py tile -i ortophotos/mi_ortofoto.tif -x 10 -y 10
```

* `-o`: directorio base de salida (por defecto `ortophotos/sections`).
* El resultado queda en `ortophotos/sections/<ortofoto>/10_10/` con su `10_10_metadata.json` (donde `<ortofoto>` es el nombre base de tu imagen, p. ej. `mi_ortofoto`).

#### 2. Simular un vuelo (`simulate`)
Genera la trayectoria del dron (recta, curva o lemniscata), recorta las imágenes sintéticas paso a paso y guarda el *ground truth* en JSON.

```bash
python cli.py simulate -o ortophotos/mi_ortofoto.tif \
                       -j ortophotos/sections/mi_ortofoto/10_10/10_10_metadata.json \
                       -n vuelo_1 -t curve --speed 10.0
```

Opciones más útiles: `--start-x/--start-y/--end-x/--end-y` (si no se indican, se eligen al azar dentro de la ortofoto), `-f` (frecuencia de captura en Hz), `-l` (vueltas, útil en lemniscata), `-a` (ángulo de rotación en grados), `--out-w/--out-h` (resolución de imagen) y `--out-dir` (directorio de salida personalizado).

#### 3. Emparejar (`match`)
Localiza cada imagen simulada dentro de la base de datos de teselas usando el algoritmo elegido.

```bash
python cli.py match -t trajectories/vuelo_1 \
                    -d ortophotos/sections/mi_ortofoto/10_10/10_10_metadata.json \
                    -m superpoint_lightglue
```

* `-m`: método (`sift`, `orb` o `superpoint_lightglue`).
* `-t`: puede ser una trayectoria concreta o una **familia** de trayectorias (procesa todas las que encuentre).
* `-th`: radio de búsqueda de teselas candidatas en metros (por defecto 100 m).
* `-f/--force`: re-ejecuta aunque ya existan resultados previos.

#### 4. Evaluar (`evaluate`)
Compara la posición estimada con el *ground truth* y genera métricas y gráficas.

```bash
python cli.py evaluate -t trajectories/vuelo_1 -mod single -m superpoint_lightglue \
                       -of mad -s median
```

* `-mod`: `single` (una trayectoria) o `batch` (familia/completo).
* `-of/--outlier-filter`: `none`, `iqr` (rango intercuartílico) o `mad` (Z-score modificado).
* `-s/--shift`: corrección del sesgo sistemático global con `mean`, `median` o una tupla `(x,y)`.

#### 5. Pipeline automático (`pipeline`)
Ejecuta simulación + matching + evaluación de forma consecutiva. Es la forma más rápida de lanzar **baterías de pruebas** (familias de vuelos con múltiples ejecuciones).

```bash
python cli.py pipeline -o mi_ortofoto -t lemniscate -m sift --runs 5
```

* `--runs N`: genera y evalúa N trayectorias (crea una *familia* y la evalúa en conjunto).
* Si se omite `-o` o `-t`, el pipeline los elige aleatoriamente (en modo familia).

### Salidas generadas

Por cada trayectoria se generan, entre otros:

* `images/`: recortes sintéticos capturados por el dron.
* `<nombre>_metadata.json`: *ground truth* georreferenciado de la trayectoria.
* `matching_results.json`: estimaciones por imagen y método (top de teselas candidatas, scores, inliers…).
* `<nombre>_stats_summary_*.json`: resúmenes estadísticos.
* `charts/`: gráficas de error, comparativas entre métodos, mapas *ground truth vs estimado*, etc.
* `odometry_path.json`: trayectoria estimada por odometría visual pura.

---

## ⚙️ Descripción detallada de las características

### 1. Simulación de vuelo fotorrealista y georreferenciada
* **Trayectorias configurables:** línea recta, curva (interpolación cuadrática de Bézier) y **lemniscata** (trayectoria cerrada en forma de ∞), con soporte de múltiples vueltas.
* **Cámara virtual:** extrae de la ortofoto (vía `rasterio`) recortes orientados según la dirección de vuelo, simulando la perspectiva del dron a resolución arbitraria.
* **Ground truth exacto:** cada captura queda asociada a su posición global real (coordenadas del CRS de la ortofoto), lo que permite medir el error de forma rigurosa.
* Inicio/fin autogenerados de forma aleatoria y segura dentro de los límites de la ortofoto.

### 2. Procesamiento geoespacial y teselado
* Divide ortofotografías GeoTIFF de gran tamaño en cuadrículas de teselas (ej. `5×5`, `10×10`) manteniendo su georreferenciación.
* Genera los metadatos (origen global, tamaño en metros/píxeles, fila/columna) que permiten convertir entre píxeles, teselas y coordenadas del mundo real.

### 3. Motor de emparejamiento modular
* **SIFT** — detector/descriptor clásico de alta precisión invariante a escala y rotación; búsqueda k-NN sobre **FLANN/KD-Tree** + *Lowe's ratio test* + **RANSAC** sobre la matriz fundamental.
* **ORB** — alternativa binaria muy rápida (FAST + BRIEF orientado); búsqueda k-NN sobre **FLANN/LSH** + *Lowe's ratio test* + **RANSAC**.
* **SuperPoint + LightGlue** — CNN auto-supervisada para extraer puntos y descriptores, emparejados por un Transformer con *self/cross-attention* que poda *outliers* de forma nativa (sin RANSAC).
* **Puntuación robusta de emparejamientos:** la función `compute_robust_score` combina nº de *inliers*, ratio de *inliers*, cobertura espacial y distancia media de descriptores para elegir la tesela más fiable y evitar falsos positivos en texturas repetitivas.
* **Refinamiento de localización:** las candidatas se filtran con **DBSCAN** ponderando la heurística y se fusionan en una posición final refinada.
* **Odometría visual:** estimación de movimiento entre frames consecutivos mediante homografía + RANSAC como fuente de información independiente.

### 4. Evaluación y métricas estadísticas
* Métricas por eje y distancia: **MSE, RMSE, MAE, MEE (sesgo medio)** y **CEP** (radio que contiene el 50 % de las muestras).
* **Filtrado de outliers robusto:** `IQR` (rango intercuartílico) o `MAD` (Z-score modificado).
* **Corrección de sesgo (`shift`):** resta la tendencia global de los ejes X/Y (`mean`/`median`) para medir la dispersión real del emparejamiento independientemente del sesgo sistemático del método.
* Evaluación en modos `single`, `batch` (familias) y global, con agrupación por tipo de trayectoria y resúmenes estadísticos consolidados (`global_stats_summary_*.json`).

### 5. Visualización avanzada
Generación automática de gráficas: distribución y evolución temporal del error, *boxplots*, CDF (función de distribución acumulada), dispersión XY del error, error vs tiempo de cómputo, tablas de métricas/tiempos, **comparativas entre métodos** (SIFT vs ORB vs SuperPoint+LightGlue) y **mapas** con la trayectoria real y la estimada superpuestas sobre la ortofoto (incluidos zooms por zonas de interés).

### 6. Pipeline automatizado y reproductibilidad
El comando `pipeline` orquesta todo el ciclo y permite lanzar **familias de experimentos** con múltiples ejecuciones, trayectorias y ángulos aleatorizados para obtener conclusiones estadísticamente significativas.

---

## 👥 Créditos

* **Autor y desarrollo:** [Pablo Nieto Arias](https://github.com/PabloNietoA) — proyecto desarrollado como Trabajo de Fin de Grado.
* **Algoritmos de emparejamiento:**
  * **SIFT** y **ORB** — implementados con [OpenCV](https://opencv.org/).
  * **SuperPoint** — *DeTone et al.*, "SuperPoint: Self-Supervised Interest Point Detection and Description".
  * **LightGlue** — *Lindenberger et al.*, "LightGlue: Local Feature Matching at Light Speed", implementación de [cvg/LightGlue](https://github.com/cvg/LightGlue).
  * **Kornia** — librería de visión por computador diferenciable en PyTorch.
* **Datos de ejemplo:** ortofotografías del **PNOA** (Plan Nacional de Ortofotografía Aérea), Instituto Geográfico Nacional de España.
* **Agradecimientos:** a los tutores y al equipo docente por la dirección y el apoyo durante el desarrollo del proyecto.

> Las ortofotos y datos incluidos en `ortophotos/` y `trajectories/` se emplean únicamente con fines académicos y de investigación.
