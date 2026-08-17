# Estructura del Proyecto

Esta es la nueva arquitectura modularizada y profesional del proyecto tras la refactorización (Fase 3).

```text
Ortoph_Gemini_Test/
│
├── cli/                            # Nueva interfaz de línea de comandos (CLI)
│   ├── __init__.py
│   └── commands/                   # Manejadores individuales para cada sub-comando
│       ├── evaluate.py             # Lógica para `python cli.py evaluate`
│       ├── match.py                # Lógica para `python cli.py match`
│       ├── pipeline.py             # Lógica para `python cli.py pipeline`
│       ├── simulate.py             # Lógica para `python cli.py simulate`
│       └── tile.py                 # Lógica para `python cli.py tile`
│
├── core/                           # Núcleo de la lógica de negocio
│   ├── evaluation/                 # Módulo de evaluación de resultados y estadísticas
│   │   ├── metrics.py              # Cálculo de RMSE, MAE, MEE, CEP, y filtrado IQR
│   │   └── runner.py               # Clase principal `Evaluator` que procesa los datos
│   │
│   ├── matching/                   # Lógica de emparejamiento (Matching Engine)
│   │   ├── batch_processor.py      # Orquestador `BatchMatcher` para flujos masivos
│   │   ├── extractors.py           # Clases abstractas y concretas de extracción (SIFT, ORB)
│   │   └── matchers.py             # `ClassicalMatcher` y `DeepExtractorMatcher`
│   │
│   ├── processing/                 # Procesamiento general
│   │   └── tiling.py               # Lógica para trocear GeoTIFFs grandes
│   │
│   ├── simulation/                 # Motores de simulación
│   │   ├── camera.py               # Lógica de dron/cámara virtual
│   │   ├── flight.py               # Simulador de secuencias completas de vuelo
│   │   └── paths.py                # Generación matemática de rutas (curvas, rectas)
│   │
│   └── odometry.py                 # Estimación de la Odometría Visual paso a paso
│   └── location_refinement.py      # Refinamiento global de la localización (inliers)
│
├── visualization/                  # Utilidades para generación de gráficas y mapas
│   ├── charts_3d.py                # Ploteos en 3D
│   ├── evaluation.py               # Gráficas de error para la clase Evaluator
│   ├── maps.py                     # Visualización de trayectorias en top-down
│   └── refinement.py               # Gráficas de inliers vs outliers
│
├── utils/                          # Utilidades genéricas e infraestructura
│   └── io.py                       # Manejo seguro de lectura/escritura de JSON y ficheros
│
└── cli.py                          # Entry-point principal (script orquestador)
```

## Beneficios
1. **Separación de Responsabilidades:** El CLI ahora está aislado del núcleo (`core/`).
2. **Modularidad en Matching y Evaluación:** Si necesitas un nuevo extractor o una nueva métrica matemática, sabes exactamente en qué archivo mirar (`extractors.py` o `metrics.py`).
3. **Escalabilidad Visual:** Toda la generación de matplotlib se concentra en `visualization/`, impidiendo que ensucie los algoritmos numéricos.
4. **Legibilidad:** Nombres de archivo como `flight.py` o `paths.py` explican su contenido con solo leerlos.
