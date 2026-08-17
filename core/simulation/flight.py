"""
Módulo Drone Simulator.

Contiene la lógica para simular un vuelo de dron sobre una ortofoto georreferenciada.
Se generan las trayectorias (rectas, curvas o lemniscatas), se extraen imágenes sintéticas 
y se guarda el Ground Truth (JSON).
"""

import os
import random
from pathlib import Path
import rasterio
from PIL import Image
from tqdm import tqdm
import math

from core.simulation.paths import generate_straight_path, generate_curved_path, generate_lemniscate_path
from core.simulation.camera import extract_oriented_image, find_matching_tile
from utils.io import create_directory, read_json, write_json
from visualization.maps import plot_trajectory_on_map

def _get_random_point_in_bounds(transform, width, height, margin_m):
    """Genera una coordenada aleatoria segura dentro de la ortofoto."""
    origin_global = transform * (0, 0)
    ortho_width_m = width * abs(transform.a)
    ortho_height_m = height * abs(transform.e)
    
    min_x = origin_global[0] + margin_m
    max_x = origin_global[0] + ortho_width_m - margin_m
    max_y = origin_global[1] - margin_m
    min_y = origin_global[1] - ortho_height_m + margin_m
    
    rx = random.uniform(min_x, max_x)
    ry = random.uniform(min_y, max_y)
    return (rx, ry)

def simulate_drone_flight(
    ortho_path: str,
    tiles_json_path: str,
    trajectory_name: str,
    traj_type: str,
    start: tuple = None,
    end: tuple = None,
    speed_m_s: float = 15.0,
    freq_hz: float = 1.0,
    out_w: int = 1280,
    out_h: int = 720,
    scale: float = 1.0,
    laps: int = 1,
    angle_deg: float = 0.0,
    output_dir: str = None
):
    """
    Ejecuta una simulación de vuelo completa, generando recortes de imagen y metadatos.

    Args:
        ortho_path (str): Ruta al archivo GeoTIFF de la ortofoto base.
        tiles_json_path (str): Ruta al JSON con metadatos de las teselas (para cruzar).
        trajectory_name (str): Nombre que identificará la simulación/trayectoria.
        traj_type (str): 'straight', 'curve', o 'lemniscate'.
        start (tuple, optional): (X, Y) globales de inicio. Si es None, se aleatoriza.
        end (tuple, optional): (X, Y) globales de fin. Si es None, se aleatoriza.
        speed_m_s (float): Velocidad del dron en m/s.
        freq_hz (float): Frecuencia de captura en fotogramas por segundo.
        out_w (int): Ancho de la imagen virtual extraída en píxeles.
        out_h (int): Alto de la imagen virtual extraída en píxeles.
        scale (float): Escala de captura (1.0 = resolución nativa).
        laps (int): Número de vueltas (sólo aplica si traj_type='lemniscate').
    """
    print(f"--- Iniciando Simulación de Vuelo: {trajectory_name} ---")
    
    # 1. Leer dimensiones de la ortofoto para conocer los límites seguros
    with rasterio.open(ortho_path) as dataset:
        ortho_width_m = dataset.width * abs(dataset.transform.a)
        ortho_height_m = dataset.height * abs(dataset.transform.e)
        origin_global = dataset.transform * (0, 0)
        
        diagonal_px = (out_w**2 + out_h**2)**0.5
        margin_m = diagonal_px * abs(dataset.transform.a)
        
        # Autogenerar start/end si son None
        if start is None:
            start = _get_random_point_in_bounds(dataset.transform, dataset.width, dataset.height, margin_m)
            print(f"Inicio autogenerado: {start}")
        if end is None:
            end = _get_random_point_in_bounds(dataset.transform, dataset.width, dataset.height, margin_m)
            print(f"Fin autogenerado: {end}")

        min_x = origin_global[0] + margin_m
        max_x = origin_global[0] + ortho_width_m - margin_m
        max_y = origin_global[1] - margin_m
        min_y = origin_global[1] - ortho_height_m + margin_m

    # 2. Generar puntos de la trayectoria
    if traj_type == "straight":
        points = generate_straight_path(start, end, speed_m_s, freq_hz)
    elif traj_type == "curve":
        points = generate_curved_path(start, end, speed_m_s, freq_hz)
    elif traj_type == "lemniscate":
        center_x, center_y = start[0], start[1]
        
        dist = ((end[0]-start[0])**2 + (end[1]-start[1])**2)**0.5
        width = dist * 0.8
        height = width * 0.5
        
        # El radio máximo de la lemniscata rotada está acotado por sqrt(width^2 + height^2).
        max_radius = math.sqrt(width**2 + height**2)
        
        if 2 * max_radius > (max_x - min_x):
            max_radius = (max_x - min_x) / 2.0
            width = max_radius / math.sqrt(1.25)
            height = width * 0.5
            
        if 2 * max_radius > (max_y - min_y):
            max_radius = (max_y - min_y) / 2.0
            width = max_radius / math.sqrt(1.25)
            height = width * 0.5
            
        if center_x - max_radius < min_x:
            center_x = min_x + max_radius
        elif center_x + max_radius > max_x:
            center_x = max_x - max_radius
            
        if center_y - max_radius < min_y:
            center_y = min_y + max_radius
        elif center_y + max_radius > max_y:
            center_y = max_y - max_radius
            
        points = generate_lemniscate_path((center_x, center_y), width, height, speed_m_s, freq_hz, laps=laps, angle_deg=angle_deg)
    else:
        raise ValueError(f"Tipo de trayectoria desconocido: {traj_type}")
        
    print(f"Trayectoria generada con {len(points)} puntos.")
    
    # 3. Cargar metadatos de las teselas
    tiles_data = read_json(tiles_json_path)
    if not tiles_data:
        raise ValueError(f"No se pudo leer la base de datos de teselas en {tiles_json_path}")
        
    tiles_metadata = tiles_data.get("tiles", [])
    
    # 4. Preparar directorios de salida
    if output_dir:
        base_out = Path(output_dir)
    else:
        base_out = Path(f"trajectories/{trajectory_name}")
    images_dir = base_out / "images"
    charts_dir = base_out / "charts"
    
    create_directory(str(images_dir))
    create_directory(str(charts_dir))
    
    # 5. Procesar ortofoto y extraer imágenes
    images_metadata = []
    with rasterio.open(ortho_path) as dataset:
        for pt in tqdm(points, desc=f"Generando imágenes ({trajectory_name})"):
            t = pt["timestamp"]
            gx, gy = pt["x"], pt["y"]
            dx, dy = pt["dir_x"], pt["dir_y"]
            
            img_array = extract_oriented_image(
                dataset, (gx, gy), (dx, dy),
                out_w, out_h, scale
            )
            
            matching_tile = find_matching_tile((gx, gy), tiles_metadata)
            
            img_name = f"{t:.2f}s_{trajectory_name}.png"
            out_img_path = images_dir / img_name
            
            pil_img = Image.fromarray(img_array)
            pil_img.save(out_img_path)
            
            inv_transform = ~dataset.transform
            col, row = inv_transform * (gx, gy)
            
            images_metadata.append({
                "name": img_name,
                "timestamp": t,
                "center_global": [gx, gy],
                "center_local_px": [int(col), int(row)],
                "matching_tile": matching_tile,
                "direction_vector": [dx, dy],
                "speed_m_s": speed_m_s
            })
            
    # 6. Guardar JSON de trayectoria (Ground Truth)
    final_json = {
        "trajectory_metadata": {
            "name": trajectory_name,
            "traj_type": traj_type,
            "resolution_px": [out_w, out_h],
            "orthophoto_name": Path(ortho_path).stem,
            "orthophoto_path": str(ortho_path).replace('\\', '/'),
            "dimensions_m": [ortho_width_m, ortho_height_m],
            "origin_global": list(origin_global),
            "database_path": str(base_out).replace('\\', '/'),
            "tiles_database_path": str(Path(tiles_json_path).parent).replace('\\', '/')
        },
        "images": images_metadata
    }
    
    json_path = base_out / f"{trajectory_name}_metadata.json"
    write_json(str(json_path), final_json)
    
    # 7. Dibujar mapa general
    print("Generando mapa de trayectoria...")
    map_path = charts_dir / "trajectory_map.png"
    plot_trajectory_on_map(ortho_path, points, str(map_path))
    
    print(f"--- Simulación Completada. Resultados en: {base_out} ---")
