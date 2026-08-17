import os
import json
import matplotlib.pyplot as plt

def plot_gt_comparison(trajectory_json_path, odometry_json_path, output_path):
    """
    Genera un gráfico 2D comparando el Ground Truth (metadata real) contra el Estimated Path (odometría visual).
    """
    with open(trajectory_json_path, 'r', encoding='utf-8') as f:
        traj_meta = json.load(f)
        
    with open(odometry_json_path, 'r', encoding='utf-8') as f:
        odom_meta = json.load(f)
        
    images = traj_meta["images"]
    
    gt_x = []
    gt_y = []
    for img in images:
        gt_x.append(img["center_global"][0])
        gt_y.append(img["center_global"][1])
        
    pure_path = odom_meta.get("pure_path", [])
    refined_path = odom_meta.get("refined_path", [])
    
    # Fallback to old format if present
    if not pure_path and "path" in odom_meta:
        pure_path = odom_meta["path"]
        refined_path = []
    
    pure_x = [pos[0] for pos in pure_path]
    pure_y = [pos[1] for pos in pure_path]
    
    ref_x = [pos[0] for pos in refined_path]
    ref_y = [pos[1] for pos in refined_path]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    min_len = min(len(gt_x), len(pure_x))
    
    ax.plot(gt_x[:min_len], gt_y[:min_len], label='Ground Truth', color='green', linewidth=2)
    ax.plot(pure_x[:min_len], pure_y[:min_len], label='Odometría Visual Pura', color='red', linestyle='--', linewidth=2)
    
    if refined_path:
        min_len_ref = min(len(gt_x), len(ref_x))
        ax.plot(ref_x[:min_len_ref], ref_y[:min_len_ref], label='Trayectoria Refinada', color='purple', linestyle='-.', linewidth=2)

    # Marcar inicio
    ax.scatter(gt_x[0], gt_y[0], color='blue', s=100, label='Inicio (t=0s)')

    ax.set_xlabel('Global X (m)')
    ax.set_ylabel('Global Y (m)')
    ax.set_title('Comparativa 2D: Ground Truth vs Odometría Visual')

    ax.legend()

    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico 2D guardado en {output_path}")
