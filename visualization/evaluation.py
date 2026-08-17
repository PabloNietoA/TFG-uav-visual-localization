"""
Módulo de Gráficas de Evaluación.

Genera un conjunto de visualizaciones (boxplots, histogramas, scatter plots,
y mapas superpuestos) para evaluar la precisión de la estimación de trayectoria
comparada con el Ground Truth.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
from rasterio.enums import Resampling
from scipy.stats import norm

def generate_evaluation_plots(results: list, stats: dict, trajectory_dir: str, orthophoto_path: str = None, suffix: str = "", method_prefix: str = "", custom_charts_dir: str = None, shift = (0.0, 0.0)):
    """
    Genera y guarda múltiples gráficas para la evaluación de la odometría.

    Args:
        results (list): Lista de diccionarios con resultados de evaluación por frame.
        stats (dict): Diccionario con las estadísticas globales (MSE, RMSE, etc.).
        trajectory_dir (str): Directorio donde guardar las gráficas (carpeta 'charts').
        orthophoto_path (str, optional): Ruta a la ortofoto para superponer. Default None.
        suffix (str, optional): Sufijo para añadir al nombre de los archivos guardados.
        method_prefix (str, optional): Prefijo del método para nombres de archivo (ej. 'sift_').
        custom_charts_dir (str, optional): Ruta de directorio personalizado para guardar las gráficas. Si no se especifica, se crea la carpeta 'charts' dentro de trajectory_dir.
        shift: Tupla (x,y) o string ('mean', 'median') que se aplicó.
    """
    if not orthophoto_path:
        import glob
        import json
        meta_files = glob.glob(os.path.join(trajectory_dir, "*_metadata.json"))
        if meta_files:
            try:
                with open(meta_files[0], 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                ortho_name = meta.get("trajectory_metadata", {}).get("orthophoto_name")
                if ortho_name:
                    candidate = os.path.join("ortophotos", f"{ortho_name}.tif")
                    if os.path.exists(candidate):
                        orthophoto_path = candidate
            except Exception as e:
                print(f"Error al leer la ortofoto desde metadatos en {trajectory_dir}: {e}")

    if custom_charts_dir:
        charts_dir = custom_charts_dir
    else:
        charts_dir = os.path.join(trajectory_dir, "charts")
        
    os.makedirs(charts_dir, exist_ok=True)
    
    # Construir prefijo para nombres de archivo
    prefix = f"{method_prefix}_" if method_prefix else ""
    
    # Extract data
    errors_x = [r["err_x"] for r in results]
    errors_y = [r["err_y"] for r in results]
    errors_dist = [r["err_dist"] for r in results]
    times = [r["time"] for r in results]
    
    gt_x = [r["gt_pos"][0] for r in results]
    gt_y = [r["gt_pos"][1] for r in results]
    est_x = [r["est_pos"][0] for r in results]
    est_y = [r["est_pos"][1] for r in results]

    if isinstance(shift, str):
        shift_title = f" (Shift: {shift.capitalize()})"
    elif shift != (0.0, 0.0):
        shift_title = f" (Shift: {shift})"
    else:
        shift_title = ""

    # 1. Boxplots of error distribution in X, Y, and distance error together
    plt.figure(figsize=(10, 6))
    plt.boxplot([errors_x, errors_y, errors_dist])
    plt.xticks([1, 2, 3], ['Error X', 'Error Y', 'Error Distancia'])
    plt.title(f'Distribución del Error (Boxplots){shift_title}')
    plt.ylabel('Error (m)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(charts_dir, f"{prefix}boxplot_errors{suffix}.png"), dpi=150)
    plt.close()

    # 2 & 3. Boxplots separately in normal distribution format and histogram format with MEE line
    def plot_dist(data, name, mee, filename_prefix):
        # Boxplot on top or bottom? The user asked for boxplot and histogram/normal dist. 
        # Actually it's better to make it two subplots or just overlay.
        # "boxplot ... junto a ... normal... linea con el mee. Haz esto mismo en histograma".
        # Let's create a combined plot with a boxplot on top and histogram below
        
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(8, 6))
        
        ax_box.boxplot(data, vert=False)
        ax_box.axvline(mee, color='r', linestyle='dashed', linewidth=2)
        ax_box.set_yticks([])
        
        # Normal distribution fit
        mu, std = norm.fit(data)
        xmin, xmax = min(data) - std, max(data) + std
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        
        ax_hist.hist(data, bins=30, density=True, alpha=0.6, color='b', label='Histograma')
        ax_hist.plot(x, p, 'k', linewidth=2, label=fr'Ajuste Normal ($\mu={mu:.2f}$)')
        ax_hist.axvline(mee, color='r', linestyle='dashed', linewidth=2, label=f'MEE ({mee:.2f})')
        
        ax_hist.set_title(f'Distribución de Error: {name}{shift_title}')
        ax_hist.set_xlabel('Error (m)')
        ax_hist.set_ylabel('Densidad')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(charts_dir, f"{prefix}{filename_prefix}_dist{suffix}.png"), dpi=150)
        plt.close()

    plot_dist(errors_x, "Eje X", stats["x"]["MEE"], "error_x")
    plot_dist(errors_y, "Eje Y", stats["y"]["MEE"], "error_y")
    plot_dist(errors_dist, "Distancia", stats["distance"]["MEE"], "error_dist")

    # 4. Cumulative error curve with percentiles
    plt.figure(figsize=(8, 6))
    sorted_dist = np.sort(errors_dist)
    p = 1. * np.arange(len(sorted_dist)) / (len(sorted_dist) - 1)
    plt.plot(sorted_dist, p, marker='.', linestyle='none', color='b')
    
    # Percentiles
    perc_50 = np.percentile(sorted_dist, 50) # CEP
    perc_90 = np.percentile(sorted_dist, 90)
    perc_95 = np.percentile(sorted_dist, 95)
    
    plt.axvline(perc_50, color='g', linestyle='--', label=f'50% (CEP): {perc_50:.2f}m')
    plt.axvline(perc_90, color='orange', linestyle='--', label=f'90%: {perc_90:.2f}m')
    plt.axvline(perc_95, color='r', linestyle='--', label=f'95%: {perc_95:.2f}m')
    
    plt.title(f'Curva de Error Acumulado (Distancia){shift_title}')
    plt.xlabel('Error de Distancia (m)')
    plt.ylabel('Probabilidad Acumulada')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(charts_dir, f"{prefix}cumulative_error{suffix}.png"), dpi=150)
    plt.close()

    # 5. Ground truth vs estimated in X and Y without connecting points on orthophoto
    if orthophoto_path and os.path.exists(orthophoto_path):
        with rasterio.open(orthophoto_path) as dataset:
            scale_factor = 1/20
            new_width = max(1, int(dataset.width * scale_factor))
            new_height = max(1, int(dataset.height * scale_factor))
            
            data = dataset.read(
                out_shape=(dataset.count, new_height, new_width),
                resampling=Resampling.bilinear
            )
            bounds = dataset.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
        img = np.transpose(data, (1, 2, 0))
        if img.max() > 255:
            img = (img / 256).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)
        if img.shape[2] == 4:
            img = img[:, :, :3]
            
        plt.figure(figsize=(12, 12))
        plt.imshow(img, extent=extent)
        # Ground Truth scatter
        plt.scatter(gt_x, gt_y, color='lime', marker='o', s=20, label='Ground Truth')
        
        # Calculate percentiles and outliers for heatmap
        e_dist = np.array(errors_dist)
        p25 = np.percentile(e_dist, 25)
        p50 = np.percentile(e_dist, 50)
        p75 = np.percentile(e_dist, 75)
        p90 = np.percentile(e_dist, 90)
        p95 = np.percentile(e_dist, 95)
        iqr = p75 - p25
        upper_bound = p75 + 1.5 * iqr

        colors = []
        
        for i, err in enumerate(e_dist):
            if err > p95:
                colors.append('indigo')
            elif err > p90:
                colors.append('red')
            elif err > p75:
                colors.append('orange')
            elif err > p50:
                colors.append('yellow')
            elif err > p25:
                colors.append('green')
            else:
                colors.append('lightgreen')

        # Estimated scatter (Heatmap)
        plt.scatter(est_x, est_y, c=colors, marker='x', s=20)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Ground Truth', markerfacecolor='lime', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='> P95', markeredgecolor='indigo', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='> P90', markeredgecolor='red', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='> P75', markeredgecolor='orange', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='> P50', markeredgecolor='yellow', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='> P25', markeredgecolor='green', markersize=8),
            Line2D([0], [0], marker='x', color='w', label='<= P25', markeredgecolor='lightgreen', markersize=8)
        ]
        
        # Ajustar límites de los ejes a la trayectoria con algo de margen
        margin = 100
        plt.xlim(min(min(gt_x), min(est_x)) - margin, max(max(gt_x), max(est_x)) + margin)
        plt.ylim(min(min(gt_y), min(est_y)) - margin, max(max(gt_y), max(est_y)) + margin)
        
        plt.title(f'Ground Truth vs Estimado (Mapa de Calor){shift_title}')
        plt.xlabel('X Global')
        plt.ylabel('Y Global')
        plt.legend(handles=legend_elements)
        plt.savefig(os.path.join(charts_dir, f"{prefix}map_gt_vs_est{suffix}.png"), dpi=200, bbox_inches='tight')
        plt.close()
        # Center on the zone with highest density of large errors (> P90)
        high_err_indices = np.where(e_dist > p90)[0]
        if len(high_err_indices) > 0:
            high_x = np.array([est_x[i] for i in high_err_indices])
            high_y = np.array([est_y[i] for i in high_err_indices])
            
            best_count = -1
            best_idx = 0
            # Look for the point with the most high-error neighbors within 200m
            for i, hx in enumerate(high_x):
                hy = high_y[i]
                dists = (high_x - hx)**2 + (high_y - hy)**2
                count = np.sum(dists < 200**2)
                if count > best_count:
                    best_count = count
                    best_idx = i
                    
            center_x = high_x[best_idx]
            center_y = high_y[best_idx]
        else:
            max_idx = np.argmax(e_dist)
            center_x = est_x[max_idx]
            center_y = est_y[max_idx]

        zoom_margin = 250  
        # Read high-res crops for zoomed images (5b)
        from rasterio.windows import from_bounds
        with rasterio.open(orthophoto_path) as dataset:
            win_5b = from_bounds(center_x - zoom_margin, center_y - zoom_margin, center_x + zoom_margin, center_y + zoom_margin, dataset.transform)
            data_5b = dataset.read(window=win_5b, boundless=True, fill_value=0)
            bounds_5b = rasterio.windows.bounds(win_5b, dataset.transform)
            extent_5b = [bounds_5b[0], bounds_5b[2], bounds_5b[1], bounds_5b[3]]
            img_5b = np.transpose(data_5b, (1, 2, 0))
            if img_5b.max() > 255: img_5b = (img_5b / 256).astype(np.uint8)
            elif img_5b.dtype != np.uint8: img_5b = img_5b.astype(np.uint8)
            if img_5b.shape[2] == 4: img_5b = img_5b[:, :, :3]
            
        # 5b. Zoomed in version on the area with highest concentration of errors
        plt.figure(figsize=(12, 12))
        plt.imshow(img_5b, extent=extent_5b)
        plt.scatter(gt_x, gt_y, color='lime', marker='o', s=40, label='Ground Truth')
        plt.scatter(est_x, est_y, c=colors, marker='x', s=40)
        
        plt.xlim(center_x - zoom_margin, center_x + zoom_margin)
        plt.ylim(center_y - zoom_margin, center_y + zoom_margin)
        
        plt.title(f'Ground Truth vs Estimado (Zoom Mayor Concentración de Error){shift_title}')
        plt.xlabel('X Global')
        plt.ylabel('Y Global')
        plt.legend(handles=legend_elements)
        plt.savefig(os.path.join(charts_dir, f"{prefix}map_gt_vs_est_zoomed{suffix}.png"), dpi=200, bbox_inches='tight')
        plt.close()

        # 5c. Zoomed in versions with 1920x1080 (16:9) proportions for multiple zones
        clusters = []
        if len(high_err_indices) > 0:
            for idx in high_err_indices:
                px, py = est_x[idx], est_y[idx]
                added = False
                for c in clusters:
                    cx = np.mean([est_x[i] for i in c])
                    cy = np.mean([est_y[i] for i in c])
                    if (px - cx)**2 + (py - cy)**2 < 300**2:
                        c.append(idx)
                        added = True
                        break
                if not added:
                    clusters.append([idx])
                    
        valid_clusters = [c for c in clusters if len(c) >= 3]
        if valid_clusters:
            # Sort by number of high-error points (outliers) in the cluster and keep top 2
            valid_clusters = sorted(valid_clusters, key=len, reverse=True)[:2]
        else:
            if clusters:
                valid_clusters = [max(clusters, key=len)]
            else:
                valid_clusters = [[np.argmax(e_dist)]]
                
        zoom_margin_y_5c = 350
        zoom_margin_x_5c = zoom_margin_y_5c * (1920 / 1080)
        
        for z_idx, cluster in enumerate(valid_clusters):
            center_x_5c = np.mean([est_x[i] for i in cluster])
            center_y_5c = np.mean([est_y[i] for i in cluster])
            
            x_min_5c = center_x_5c - zoom_margin_x_5c
            x_max_5c = center_x_5c + zoom_margin_x_5c
            y_min_5c = center_y_5c - zoom_margin_y_5c
            y_max_5c = center_y_5c + zoom_margin_y_5c
            
            with rasterio.open(orthophoto_path) as dataset:
                win_5c = from_bounds(x_min_5c, y_min_5c, x_max_5c, y_max_5c, dataset.transform)
                data_5c = dataset.read(window=win_5c, boundless=True, fill_value=0)
                bounds_5c = rasterio.windows.bounds(win_5c, dataset.transform)
                extent_5c = [bounds_5c[0], bounds_5c[2], bounds_5c[1], bounds_5c[3]]
                img_5c = np.transpose(data_5c, (1, 2, 0))
                if img_5c.max() > 255: img_5c = (img_5c / 256).astype(np.uint8)
                elif img_5c.dtype != np.uint8: img_5c = img_5c.astype(np.uint8)
                if img_5c.shape[2] == 4: img_5c = img_5c[:, :, :3]
                
            fig, ax = plt.subplots(figsize=(19.2, 10.8))
            ax.imshow(img_5c, extent=extent_5c)
            ax.scatter(gt_x, gt_y, color='lime', marker='o', s=40, label='Ground Truth')
            ax.scatter(est_x, est_y, c=colors, marker='x', s=40)
            
            ax.set_xlim(x_min_5c, x_max_5c)
            ax.set_ylim(y_min_5c, y_max_5c)
            
            ax.set_title(f'Ground Truth vs Estimado (Zoom 1920x1080 - Zona {z_idx + 1}){shift_title}')
            ax.set_xlabel('X Global')
            ax.set_ylabel('Y Global')
            ax.legend(handles=legend_elements)
            
            ins_ax = ax.inset_axes([0.75, 0.05, 0.23, 0.23])
            ins_ax.imshow(img, extent=extent)
            ins_ax.plot(gt_x, gt_y, color='lime', linewidth=1)
            
            import matplotlib.patches as patches
            rect = patches.Rectangle((x_min_5c, y_min_5c), x_max_5c - x_min_5c, y_max_5c - y_min_5c, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ins_ax.add_patch(rect)
            ins_ax.set_xticks([])
            ins_ax.set_yticks([])
            for spine in ins_ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(2)
                
            plt.savefig(os.path.join(charts_dir, f"{prefix}map_gt_vs_est_zoomed_1920x1080_zone{z_idx + 1}{suffix}.png"), dpi=100)
            plt.close()
    else:
        print(f"Ortofoto no encontrada o no proporcionada: {orthophoto_path}. Omitiendo overlay en mapa.")

    # 6. Scatter plot of error in X vs error in Y
    plt.figure(figsize=(8, 8))
    plt.scatter(errors_x, errors_y, alpha=0.7, color='purple')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.axvline(0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Error en X vs Error en Y{shift_title}')
    plt.xlabel('Error X (m)')
    plt.ylabel('Error Y (m)')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(charts_dir, f"{prefix}scatter_error_xy{suffix}.png"), dpi=150)
    plt.close()

    # 7. Error in X over time and Error in Y over time superimposed
    plt.figure(figsize=(10, 6))
    plt.plot(times, errors_x, marker='o', linestyle='-', color='b', label='Error X', markersize=4)
    plt.plot(times, errors_y, marker='s', linestyle='-', color='r', label='Error Y', markersize=4)
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'Error en X e Y a lo largo del Tiempo{shift_title}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(charts_dir, f"{prefix}error_over_time{suffix}.png"), dpi=150)
    plt.close()

    # 8. Tiempo de procesamiento por imagen
    elapsed_times = [r.get("elapsed_seconds") for r in results if r.get("elapsed_seconds") is not None]
    if elapsed_times:
        # 8a. Histograma de tiempos
        f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(8, 6))
        ax_box.boxplot(elapsed_times, vert=False)
        mean_t = np.mean(elapsed_times)
        ax_box.axvline(mean_t, color='r', linestyle='dashed', linewidth=2)
        ax_box.set_yticks([])
        
        mu_t, std_t = norm.fit(elapsed_times)
        xmin_t, xmax_t = min(elapsed_times) - std_t, max(elapsed_times) + std_t
        x_t = np.linspace(max(0, xmin_t), xmax_t, 100)
        p_t = norm.pdf(x_t, mu_t, std_t)
        
        ax_hist.hist(elapsed_times, bins=30, density=True, alpha=0.6, color='teal', label='Histograma')
        ax_hist.plot(x_t, p_t, 'k', linewidth=2, label=fr'Ajuste Normal ($\mu={mu_t:.3f}$s)')
        ax_hist.axvline(mean_t, color='r', linestyle='dashed', linewidth=2, label=f'Media ({mean_t:.3f}s)')
        
        ax_hist.set_title(f'Distribución del Tiempo de Procesamiento por Imagen{shift_title}')
        ax_hist.set_xlabel('Tiempo (s)')
        ax_hist.set_ylabel('Densidad')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_dir, f"{prefix}timing_distribution{suffix}.png"), dpi=150)
        plt.close()
        
        # 8b. Scatter de tiempo por imagen a lo largo de la trayectoria
        plt.figure(figsize=(10, 6))
        plt.scatter(times[:len(elapsed_times)], elapsed_times, color='teal', marker='o', s=20, alpha=0.7)
        plt.axhline(mean_t, color='r', linestyle='--', alpha=0.7, label=f'Media ({mean_t:.3f}s)')
        plt.title(f'Tiempo de Procesamiento a lo largo de la Trayectoria{shift_title}')
        plt.xlabel('Tiempo de Trayectoria (s)')
        plt.ylabel('Tiempo de Procesamiento (s)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_dir, f"{prefix}timing_over_time{suffix}.png"), dpi=150)
        plt.close()

        # 8c. Scatter de Error vs Tiempo de Procesamiento
        e_dist_valid = [r["err_dist"] for r in results if r.get("elapsed_seconds") is not None]
        plt.figure(figsize=(8, 6))
        plt.scatter(elapsed_times, e_dist_valid, color='darkorange', marker='o', s=20, alpha=0.7)
        plt.title(f'Error de Distancia vs Tiempo de Procesamiento{shift_title}')
        plt.xlabel('Tiempo de Procesamiento (s)')
        plt.ylabel('Error de Distancia (m)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_dir, f"{prefix}scatter_error_vs_timing{suffix}.png"), dpi=150)
        plt.close()


def generate_comparison_plots(method_results: dict, charts_dir: str, suffix: str = ""):
    """
    Genera gráficas comparativas entre métodos de emparejamiento.

    Genera comparativas por parejas (boxplot + CDF) y una comparativa global
    (boxplot agrupado + CDF superpuesta + tabla resumen de métricas).

    Args:
        method_results (dict): Diccionario {method_name: (results_list, stats_dict)}.
        charts_dir (str): Directorio donde guardar las gráficas.
        suffix (str, optional): Sufijo para nombres de archivo.
    """
    os.makedirs(charts_dir, exist_ok=True)
    methods = sorted(method_results.keys())
    
    if len(methods) < 2:
        print("Se necesitan al menos 2 métodos para generar comparativas.")
        return

    # Paleta de colores para los métodos
    color_palette = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4', '#795548']
    method_colors = {m: color_palette[i % len(color_palette)] for i, m in enumerate(methods)}

    # --- Comparativas por parejas ---
    from itertools import combinations
    for m1, m2 in combinations(methods, 2):
        res1, stats1 = method_results[m1]
        res2, stats2 = method_results[m2]
        
        e_dist1 = [r["err_dist"] for r in res1]
        e_dist2 = [r["err_dist"] for r in res2]
        
        pair_label = f"{m1}_vs_{m2}"
        
        # 1. Boxplot enfrentado
        fig, ax = plt.subplots(figsize=(8, 6))
        bp = ax.boxplot([e_dist1, e_dist2], labels=[m1, m2], patch_artist=True)
        bp['boxes'][0].set_facecolor(method_colors[m1])
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(method_colors[m2])
        bp['boxes'][1].set_alpha(0.6)
        ax.set_title(f'Comparativa Error Distancia: {m1} vs {m2}')
        ax.set_ylabel('Error de Distancia (m)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(charts_dir, f"comparison_{pair_label}_boxplot{suffix}.png"), dpi=150)
        plt.close()
        
        # 2. CDF superpuesta
        fig, ax = plt.subplots(figsize=(8, 6))
        for m_name, e_dist, color in [(m1, e_dist1, method_colors[m1]), (m2, e_dist2, method_colors[m2])]:
            sorted_dist = np.sort(e_dist)
            p = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist)
            ax.plot(sorted_dist, p, color=color, linewidth=2, label=m_name)
        ax.set_title(f'CDF Error Distancia: {m1} vs {m2}')
        ax.set_xlabel('Error de Distancia (m)')
        ax.set_ylabel('Probabilidad Acumulada')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(charts_dir, f"comparison_{pair_label}_cdf{suffix}.png"), dpi=150)
        plt.close()

    # --- Comparativa global (todos los métodos) ---
    
    # 1. Boxplot agrupado
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2.5), 6))
    all_errors = [([r["err_dist"] for r in method_results[m][0]]) for m in methods]
    bp = ax.boxplot(all_errors, labels=methods, patch_artist=True)
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(method_colors[methods[i]])
        box.set_alpha(0.6)
    ax.set_title('Comparativa Global: Error de Distancia por Método')
    ax.set_ylabel('Error de Distancia (m)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, f"comparison_all_boxplot{suffix}.png"), dpi=150)
    plt.close()
    
    # 2. CDF superpuesta global
    fig, ax = plt.subplots(figsize=(10, 6))
    for m_name in methods:
        res_list, _ = method_results[m_name]
        e_dist = sorted([r["err_dist"] for r in res_list])
        p = np.arange(1, len(e_dist) + 1) / len(e_dist)
        ax.plot(e_dist, p, color=method_colors[m_name], linewidth=2, label=m_name)
    ax.set_title('CDF Global: Error de Distancia por Método')
    ax.set_xlabel('Error de Distancia (m)')
    ax.set_ylabel('Probabilidad Acumulada')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(charts_dir, f"comparison_all_cdf{suffix}.png"), dpi=150)
    plt.close()
    
    # 3. Tabla resumen de métricas
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2.5), 4))
    ax.axis('off')
    
    col_headers = methods
    row_headers = ['RMSE (m)', 'MAE (m)', 'CEP (m)', 'MEE (m)', 'Nº Muestras']
    
    cell_data = []
    for row_label in row_headers:
        row = []
        for m_name in methods:
            _, stats = method_results[m_name]
            res_list = method_results[m_name][0]
            if row_label == 'RMSE (m)':
                row.append(f"{stats['distance']['RMSE']:.2f}")
            elif row_label == 'MAE (m)':
                row.append(f"{stats['distance']['MAE']:.2f}")
            elif row_label == 'CEP (m)':
                row.append(f"{stats['distance'].get('CEP', 0):.2f}")
            elif row_label == 'MEE (m)':
                row.append(f"{stats['distance']['MEE']:.2f}")
            elif row_label == 'Nº Muestras':
                row.append(str(len(res_list)))
        cell_data.append(row)
    
    table = ax.table(
        cellText=cell_data,
        rowLabels=row_headers,
        colLabels=col_headers,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Colorear encabezados de columna
    for j, m_name in enumerate(methods):
        cell = table[0, j]
        cell.set_facecolor(method_colors[m_name])
        cell.set_alpha(0.3)
    
    ax.set_title('Resumen Comparativo de Métricas', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, f"comparison_all_metrics_table{suffix}.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Comparativa de tiempos de procesamiento (boxplot)
    timing_data = {}
    for m_name in methods:
        res_list, _ = method_results[m_name]
        t_list = [r.get("elapsed_seconds") for r in res_list if r.get("elapsed_seconds") is not None]
        if t_list:
            timing_data[m_name] = t_list
    
    if len(timing_data) >= 2:
        timing_methods = sorted(timing_data.keys())
        fig, ax = plt.subplots(figsize=(max(8, len(timing_methods) * 2.5), 6))
        bp = ax.boxplot([timing_data[m] for m in timing_methods], labels=timing_methods, patch_artist=True)
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(method_colors[timing_methods[i]])
            box.set_alpha(0.6)
        ax.set_title('Comparativa: Tiempo de Procesamiento por Método')
        ax.set_ylabel('Tiempo por Imagen (s)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"comparison_all_timing_boxplot{suffix}.png"), dpi=150)
        plt.close()

        # 5. Tabla resumen de tiempos
        fig, ax = plt.subplots(figsize=(max(8, len(timing_methods) * 2.5), 3.5))
        ax.axis('off')
        
        t_row_headers = ['Media (s)', 'Mediana (s)', 'Std (s)', 'Mín (s)', 'Máx (s)', 'Total (s)']
        t_cell_data = []
        for row_label in t_row_headers:
            row = []
            for m_name in timing_methods:
                t_arr = np.array(timing_data[m_name])
                if row_label == 'Media (s)':
                    row.append(f"{np.mean(t_arr):.3f}")
                elif row_label == 'Mediana (s)':
                    row.append(f"{np.median(t_arr):.3f}")
                elif row_label == 'Std (s)':
                    row.append(f"{np.std(t_arr):.3f}")
                elif row_label == 'Mín (s)':
                    row.append(f"{np.min(t_arr):.3f}")
                elif row_label == 'Máx (s)':
                    row.append(f"{np.max(t_arr):.3f}")
                elif row_label == 'Total (s)':
                    row.append(f"{np.sum(t_arr):.1f}")
            t_cell_data.append(row)
        
        t_table = ax.table(
            cellText=t_cell_data,
            rowLabels=t_row_headers,
            colLabels=timing_methods,
            cellLoc='center',
            loc='center'
        )
        t_table.auto_set_font_size(False)
        t_table.set_fontsize(10)
        t_table.scale(1.2, 1.5)
        for j, m_name in enumerate(timing_methods):
            cell = t_table[0, j]
            cell.set_facecolor(method_colors[m_name])
            cell.set_alpha(0.3)
        ax.set_title('Comparativa: Estadísticas de Tiempo', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"comparison_all_timing_table{suffix}.png"), dpi=150, bbox_inches='tight')
        plt.close()
