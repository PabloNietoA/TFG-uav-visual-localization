import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
from PIL import Image

# Necesitamos añadir el directorio padre al sys.path o usar importación relativa si se corre como módulo.
# Aquí asumimos que el script main estará en el directorio raíz del proyecto.
from utils.io import create_directory, write_json

# Aumentamos el límite para imágenes grandes en Pillow para evitar errores de DoS
Image.MAX_IMAGE_PIXELS = None

def get_orthophoto_metadata(filepath: str) -> dict:
    """
    Lee un archivo GeoTIFF y extrae sus dimensiones, matriz de transformación y origen.
    """
    with rasterio.open(filepath) as dataset:
        width_px = dataset.width
        height_px = dataset.height
        transform = dataset.transform
        
        # El origen global de la imagen es el píxel (0, 0)
        origin_global_x, origin_global_y = transform * (0, 0)
        
        # Dimensiones de los píxeles (a y e en la matriz de transformación)
        pixel_width_m = abs(transform.a)
        pixel_height_m = abs(transform.e)
        
        return {
            "width_px": width_px,
            "height_px": height_px,
            "width_m": width_px * pixel_width_m,
            "height_m": height_px * pixel_height_m,
            "origin_global": [origin_global_x, origin_global_y],
            "transform": transform,
            "pixel_width_m": pixel_width_m,
            "pixel_height_m": pixel_height_m
        }

def split_orthophoto_and_save(filepath: str, x_sections: int, y_sections: int, output_base_dir: str):
    """
    Divide una ortofotografía en x_sections horizontales y y_sections verticales,
    guarda las teselas en formato PNG y genera el archivo JSON de metadatos.
    """
    path = Path(filepath)
    ortho_name = path.stem
    
    metadata = get_orthophoto_metadata(filepath)
    width_px = metadata["width_px"]
    height_px = metadata["height_px"]
    transform = metadata["transform"]
    
    # Calcular tamaño base de cada sección
    base_tile_width_px = width_px // x_sections
    base_tile_height_px = height_px // y_sections
    
    # Carpetas de salida
    sections_folder_name = f"{y_sections}_{x_sections}"
    base_out_dir = Path(output_base_dir) / ortho_name / sections_folder_name
    images_dir = base_out_dir / "images"
    create_directory(str(images_dir))
    
    tiles_metadata = []
    
    with rasterio.open(filepath) as dataset:
        for row in range(y_sections):
            for col in range(x_sections):
                # Calcular inicio y tamaño de la ventana
                col_start = col * base_tile_width_px
                row_start = row * base_tile_height_px
                
                # La última fila y columna absorben el resto si la división no es exacta
                current_tile_width = base_tile_width_px if col < x_sections - 1 else width_px - col_start
                current_tile_height = base_tile_height_px if row < y_sections - 1 else height_px - row_start
                
                # Origen de la tesela
                origin_ortho_px = [col_start, row_start]
                origin_global_x, origin_global_y = transform * (col_start, row_start)
                
                # Dimensiones en metros de la tesela
                tile_width_m = current_tile_width * metadata["pixel_width_m"]
                tile_height_m = current_tile_height * metadata["pixel_height_m"]
                
                tile_name = f"{row}_{col}.png"
                
                # Guardar información de la tesela
                tile_info = {
                    "name": tile_name,
                    "row": row,
                    "column": col,
                    "origin_global": [origin_global_x, origin_global_y],
                    "origin_orthophoto": origin_ortho_px,
                    "width_px": current_tile_width,
                    "height_px": current_tile_height,
                    "width_m": tile_width_m,
                    "height_m": tile_height_m
                }
                tiles_metadata.append(tile_info)
                
                # Leer la ventana usando rasterio
                window = Window(col_start, row_start, current_tile_width, current_tile_height)
                # Leer las tres bandas (RGB). Asumimos que la imagen tiene al menos 3 bandas
                # Si tiene 4 (RGBA), las lee todas.
                num_bands = dataset.count
                bands_to_read = min(num_bands, 4)
                tile_data = dataset.read(tuple(range(1, bands_to_read + 1)), window=window)
                
                # rasterio devuelve los datos en formato (bandas, alto, ancho)
                # Pillow necesita (alto, ancho, bandas)
                import numpy as np
                tile_data_transposed = np.transpose(tile_data, (1, 2, 0))
                
                # Crear la imagen con Pillow y guardarla
                mode = 'RGB' if bands_to_read == 3 else 'RGBA'
                # Si es de 1 banda (escala de grises)
                if bands_to_read == 1:
                    tile_data_transposed = tile_data_transposed[:, :, 0]
                    mode = 'L'
                    
                img = Image.fromarray(tile_data_transposed, mode=mode)
                
                out_tile_path = images_dir / tile_name
                img.save(out_tile_path, "PNG")
                
                print(f"Guardada tesela {tile_name}")

    # Preparar el documento JSON final
    final_json = {
        "orthophoto_metadata": {
            "name": ortho_name,
            "width_px": width_px,
            "height_px": height_px,
            "width_m": metadata["width_m"],
            "height_m": metadata["height_m"],
            "origin_global": metadata["origin_global"],
            "x_sections": x_sections,
            "y_sections": y_sections,
            "metadata_folder_path": str(base_out_dir).replace('\\', '/')
        },
        "tiles": tiles_metadata
    }
    
    # Guardar JSON
    json_path = base_out_dir / f"{y_sections}_{x_sections}_metadata.json"
    write_json(str(json_path), final_json)
    print(f"Metadatos guardados en {json_path}")
