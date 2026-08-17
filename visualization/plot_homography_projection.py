import os
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12

def plot_homography_projection():
    ortho_path = 'e:/TFG/ortophotos/PNOA_MA_OF_ETRS89_HU30_h25_0559_1.tif'
    out_dir = 'e:/TFG/docs/images'
    os.makedirs(out_dir, exist_ok=True)
    
    # Extraer una sola tesela
    col_off, row_off = 28671, 19071
    t_size = 400
    
    with rasterio.open(ortho_path) as src:
        win = Window(col_off + t_size, row_off + t_size, t_size, t_size)
        img = src.read((1, 2, 3), window=win)
        img = np.moveaxis(img, 0, -1)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    
    # Simular la proyección de la imagen (un polígono distorsionado por perspectiva/homografía)
    pts = np.array([
        [50, 80],
        [320, 60],
        [360, 310],
        [80, 350]
    ])
    
    poly = patches.Polygon(pts, closed=True, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='Proyección de Captura')
    ax.add_patch(poly)
    
    # Centro proyectado
    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
    ax.plot(cx, cy, 'ro', markersize=8, label='Centro Proyectado')
    
    # Flechas simulando los ejes de la cámara
    ax.arrow(cx, cy, 50, -10, head_width=10, head_length=15, fc='red', ec='red', linewidth=2)
    ax.arrow(cx, cy, -10, 50, head_width=10, head_length=15, fc='green', ec='green', linewidth=2)
    
    ax.set_title("Proyección mediante Homografía")
    ax.legend(loc='upper right')
    
    # Ocultar ejes para un gráfico más limpio
    ax.axis('off')
    
    out_path = os.path.join(out_dir, 'homography_projection.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {out_path}")

if __name__ == "__main__":
    plot_homography_projection()
