import json
import matplotlib.pyplot as plt
import os
import numpy as np

def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_refinement_results(trajectory_json_path, results_json_path, output_dir):
    traj_meta = _load_json(trajectory_json_path)
    results_meta = _load_json(results_json_path)
    
    gt_x = []
    gt_y = []
    for img in traj_meta["images"]:
        gt_x.append(img["center_global"][0])
        gt_y.append(img["center_global"][1])
        
    heu_x = []
    heu_y = []
    est_x = []
    est_y = []
    
    for res in results_meta["results"]:
        heu = res.get("estimated_global_pos_odometry")
        if heu and len(heu) == 2:
            heu_x.append(heu[0])
            heu_y.append(heu[1])
            
        est = res.get("refined_global_pos")
        if est and len(est) == 2:
            est_x.append(est[0])
            est_y.append(est[1])
        else:
            # Si no hay estimación robusta, podemos omitirlo o usar la heurística
            if heu and len(heu) == 2:
                est_x.append(heu[0])
                est_y.append(heu[1])

    # Plot 1: Heurística Corregida vs GT
    plt.figure(figsize=(10, 8))
    plt.plot(gt_x, gt_y, 'k--', label="Ground Truth", linewidth=2)
    if heu_x:
        plt.plot(heu_x, heu_y, 'b-', label="Heurística (Odometría Corregida)", alpha=0.7)
    
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.title("Trayectoria: Heurística Corregida vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis() # Invertir eje Y porque en general las coordenadas de imagen crecen hacia abajo
    
    out_heu = os.path.join(output_dir, "heuristic_vs_gt.png")
    plt.savefig(out_heu, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Posiciones Estimadas (Refinadas) vs GT
    plt.figure(figsize=(10, 8))
    plt.plot(gt_x, gt_y, 'k--', label="Ground Truth", linewidth=2)
    if est_x:
        plt.plot(est_x, est_y, 'r-', label="Posición Estimada Robusta", alpha=0.7)
        plt.scatter(est_x, est_y, c='r', s=10, alpha=0.5)
        
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.title("Trayectoria: Posición Estimada Final vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.gca().invert_yaxis()
    
    out_est = os.path.join(output_dir, "estimated_vs_gt.png")
    plt.savefig(out_est, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráficas de refinamiento guardadas en: {output_dir}")
