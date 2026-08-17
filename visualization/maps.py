import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path

def plot_trajectory_on_map(ortho_path, trajectory_points, output_path):
    """
    Dibuja la trayectoria sobre una versión de baja resolución de la ortofoto.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(ortho_path) as dataset:
        # Leer a una resolución mucho menor (e.g. 1/20) para no colapsar la RAM
        scale_factor = 1/20
        
        # calculate new dimensions
        new_width = int(dataset.width * scale_factor)
        new_height = int(dataset.height * scale_factor)
        
        data = dataset.read(
            out_shape=(dataset.count, new_height, new_width),
            resampling=Resampling.bilinear
        )
        
        # Transform for the downsampled image
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
        
        bounds = dataset.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
    img = np.transpose(data, (1, 2, 0))
    # Normalize if needed
    if img.max() > 255:
        img = (img / 256).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = img.astype(np.uint8)
        
    # Extraer solo RGB si tiene 4 canales
    if img.shape[2] == 4:
        img = img[:, :, :3]
        
    plt.figure(figsize=(10, 10))
    plt.imshow(img, extent=extent)
    
    # Extraer coordenadas de la trayectoria
    x_coords = [pt["x"] for pt in trajectory_points]
    y_coords = [pt["y"] for pt in trajectory_points]
    
    plt.plot(x_coords, y_coords, color='red', linewidth=2, label="Trayectoria")
    plt.scatter(x_coords, y_coords, color='yellow', s=10, label="Puntos de Disparo")
    
    plt.title("Ruta del Dron sobre Ortofotografía")
    plt.xlabel("Coordenada X Global")
    plt.ylabel("Coordenada Y Global")
    plt.legend()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
