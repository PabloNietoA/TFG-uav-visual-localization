import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman', 'Computer Modern Roman', 'Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 12

def plot_dbscan_refinement():
    out_dir = 'e:/TFG/docs/images'
    os.makedirs(out_dir, exist_ok=True)
    
    # Generar datos simulados
    np.random.seed(42)
    # Cluster principal (Top-N con buen matching)
    inliers = np.random.normal(loc=[10, 10], scale=1.5, size=(15, 2))
    # Valores aislados (malos matchings)
    outliers = np.random.uniform(low=[0, 0], high=[25, 25], size=(8, 2))
    
    pts = np.vstack((inliers, outliers))
    
    # Aplicar DBSCAN
    db = DBSCAN(eps=3.0, min_samples=3).fit(pts)
    labels = db.labels_
    
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Separar inliers (label != -1) de outliers (label == -1)
    mask_inliers = labels != -1
    mask_outliers = labels == -1
    
    ax.scatter(pts[mask_outliers, 0], pts[mask_outliers, 1], c='red', marker='x', s=60, label='Valores Aislados (Ruido)')
    ax.scatter(pts[mask_inliers, 0], pts[mask_inliers, 1], c='green', marker='o', s=60, alpha=0.6, label='Agrupación Aceptada')
    
    # Calcular centroide (media) de los inliers
    centroid = np.mean(pts[mask_inliers], axis=0)
    ax.plot(centroid[0], centroid[1], marker='*', color='blue', markersize=15, label='Posición Refinada')
    
    # Estimación odométrica previa (algo desviada)
    odom = [7, 13]
    ax.plot(odom[0], odom[1], marker='D', color='gray', markersize=8, label='Estimación Odométrica')
    
    # Dibujar radio eps alrededor de algunos inliers para ilustrar DBSCAN
    for pt in pts[mask_inliers][:3]:
        circle = plt.Circle((pt[0], pt[1]), 3.0, color='green', fill=False, linestyle=':', alpha=0.3)
        ax.add_patch(circle)
        
    ax.set_title("Refinamiento de Posición con DBSCAN")
    ax.set_xlabel("Desplazamiento X (m)")
    ax.set_ylabel("Desplazamiento Y (m)")
    
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    out_path = os.path.join(out_dir, 'dbscan_refinement.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {out_path}")

if __name__ == "__main__":
    plot_dbscan_refinement()
