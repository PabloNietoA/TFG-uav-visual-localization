import os
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Configurar fuente para asemejarse a Latin Modern (LaTeX)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12

def plot_search_radius():
    ortho_path = 'e:/TFG/ortophotos/PNOA_MA_OF_ETRS89_HU30_h25_0559_1.tif'
    out_dir = 'e:/TFG/docs/images'
    os.makedirs(out_dir, exist_ok=True)
    
    col_off, row_off = 28671, 19071
    w_size = 2000
    
    with rasterio.open(ortho_path) as src:
        win = Window(col_off, row_off, w_size, w_size)
        img = src.read((1, 2, 3), window=win)
        img = np.moveaxis(img, 0, -1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    tile_size = 400
    num_tiles = w_size // tile_size
    
    # Posición odométrica simulada y radio de búsqueda
    odom_x, odom_y = 1050, 950
    search_radius = 500
    
    # Dibujar teselas y evaluar si caen dentro del radio
    for i in range(num_tiles):
        for j in range(num_tiles):
            x = i * tile_size
            y = j * tile_size
            
            # Centro de la tesela
            cx, cy = x + tile_size/2, y + tile_size/2
            
            # Distancia al centro de odometría
            dist = np.sqrt((cx - odom_x)**2 + (cy - odom_y)**2)
            
            if dist <= search_radius + tile_size*0.7:  # Heurística de intersección simple
                edge_color = 'green'
                face_color = 'green'
                alpha = 0.2
            else:
                edge_color = 'red'
                face_color = 'red'
                alpha = 0.1
                
            rect = patches.Rectangle((x, y), tile_size, tile_size, linewidth=1.5, 
                                     edgecolor=edge_color, facecolor=face_color, alpha=alpha)
            ax.add_patch(rect)
    
    # Dibujar radio de búsqueda y posición
    circle = patches.Circle((odom_x, odom_y), search_radius, edgecolor='blue', facecolor='none', linewidth=2, linestyle='--')
    ax.add_patch(circle)
    ax.plot(odom_x, odom_y, 'bx', markersize=10, markeredgewidth=2, label='Posición Odométrica')
    
    # Leyenda artificial
    ax.plot([], [], color='green', alpha=0.5, linewidth=4, label='Teselas Aceptadas')
    ax.plot([], [], color='red', alpha=0.3, linewidth=4, label='Teselas Descartadas')
    ax.legend(loc='upper right', framealpha=0.9)
    
    ax.set_title("Radio de Búsqueda de Teselas")
    ax.set_xlabel("Píxeles (X)")
    ax.set_ylabel("Píxeles (Y)")
    
    out_path = os.path.join(out_dir, 'search_radius.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {out_path}")

if __name__ == "__main__":
    plot_search_radius()
