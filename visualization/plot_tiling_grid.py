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

def plot_tiling_grid():
    ortho_path = 'e:/TFG/ortophotos/PNOA_MA_OF_ETRS89_HU30_h25_0559_1.tif'
    out_dir = 'e:/TFG/docs/images'
    os.makedirs(out_dir, exist_ok=True)
    
    # Coordenadas de un bloque central (mixto)
    col_off, row_off = 28671, 19071
    w_size = 2000
    
    with rasterio.open(ortho_path) as src:
        win = Window(col_off, row_off, w_size, w_size)
        # Leer como RGB (las bandas suelen ser 1,2,3 para RGB)
        img = src.read((1, 2, 3), window=win)
        img = np.moveaxis(img, 0, -1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    
    # Simular una cuadrícula de teselas
    tile_size = 400
    num_tiles_x = w_size // tile_size
    num_tiles_y = w_size // tile_size
    
    for i in range(num_tiles_x):
        for j in range(num_tiles_y):
            x = i * tile_size
            y = j * tile_size
            
            # Dibujar bordes de tesela
            rect = patches.Rectangle((x, y), tile_size, tile_size, linewidth=1, edgecolor='white', facecolor='none', alpha=0.5)
            ax.add_patch(rect)
            
            # Destacar una tesela específica
            if i == 2 and j == 2:
                highlight = patches.Rectangle((x, y), tile_size, tile_size, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
                ax.add_patch(highlight)
                ax.text(x + tile_size/2, y + tile_size/2, f'T({i},{j})', color='white', weight='bold', 
                        fontsize=14, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=2))
            elif i % 2 == 0 and j % 2 == 0:
                ax.text(x + 10, y + 25, f'{i},{j}', color='white', fontsize=10, alpha=0.7)
                
    ax.set_title("Teselado de la Ortofotografía")
    ax.set_xlabel("Píxeles (X)")
    ax.set_ylabel("Píxeles (Y)")
    
    # Guardar figura
    out_path = os.path.join(out_dir, 'tiling_grid.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {out_path}")

if __name__ == "__main__":
    plot_tiling_grid()
