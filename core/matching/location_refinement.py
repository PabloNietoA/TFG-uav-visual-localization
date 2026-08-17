import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class LocationRefiner:
    def __init__(self, threshold_m=100.0):
        """
        threshold_m: Distancia máxima en metros para considerar una estimación de posición como válida (inlier) respecto a la heurística.
        """
        self.threshold_m = threshold_m

    def estimate_position_from_tile(self, src_pts, dst_pts, tile_info, img_shape, scale_m_per_px):
        """
        Calcula la posición global de la imagen usando la homografía hacia una tesela.
        src_pts: puntos en la imagen del dron.
        dst_pts: puntos en la imagen de la tesela.
        """
        if len(src_pts) < 4 or len(dst_pts) < 4:
            return None
            
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None
            
        # Centro de la imagen del dron
        cx = img_shape[1] / 2.0
        cy = img_shape[0] / 2.0
        
        center_pt = np.array([[[cx, cy]]], dtype=np.float32)
        
        try:
            # Proyectar el centro de la imagen del dron a coordenadas de pixel de la tesela
            tile_pt = cv2.perspectiveTransform(center_pt, H)[0][0]
        except Exception:
            return None
            
        px, py = tile_pt[0], tile_pt[1]
        
        # Validar que el punto proyectado no sea disparatado (ej. infinito o NaN)
        if not np.isfinite(px) or not np.isfinite(py):
            return None
            
        # Convertir a coordenadas globales
        tile_origin_x = tile_info["origin_global"][0]
        tile_origin_y = tile_info["origin_global"][1]
        
        # La tesela crece hacia la derecha en X y hacia abajo en pixel Y (pero Y global disminuye hacia el sur)
        global_x = tile_origin_x + px * scale_m_per_px
        global_y = tile_origin_y - py * scale_m_per_px
        
        return [float(global_x), float(global_y)]
        
    def filter_outliers_and_average(self, candidate_positions, candidate_scores, heuristic_pos):
        """
        Elimina outliers usando clustering DBSCAN dando peso a la heurística y calcula la media ponderada.
        """
        if not candidate_positions:
            return None
            
        pts = list(candidate_positions)
        scores = list(candidate_scores)
        
        # Añadir la heurística múltiples veces para darle peso extra y asegurar que forme un cluster
        heuristic_weight = max(3, len(candidate_positions))
        for _ in range(heuristic_weight):
            pts.append(heuristic_pos)
            
        pts = np.array(pts)
        
        # Aplicar DBSCAN
        clustering = DBSCAN(eps=self.threshold_m, min_samples=2).fit(pts)
        labels = clustering.labels_
        
        # Identificar el clúster de la heurística (los últimos heuristic_weight elementos)
        heuristic_label = labels[-1]
        
        if heuristic_label == -1:
            # Si la heurística se catalogó como ruido (no debería pasar por las repeticiones, pero por seguridad)
            return None
            
        valid_positions = []
        valid_scores = []
        
        # Extraer los candidatos que pertenecen al cluster de la heurística
        for i in range(len(candidate_positions)):
            if labels[i] == heuristic_label:
                valid_positions.append(candidate_positions[i])
                valid_scores.append(candidate_scores[i])
                
        if not valid_positions:
            return None
            
        # Media ponderada
        valid_positions = np.array(valid_positions)
        valid_scores = np.array(valid_scores)
        
        # Normalizar pesos
        weights = valid_scores / np.sum(valid_scores)
        
        robust_x = np.sum(valid_positions[:, 0] * weights)
        robust_y = np.sum(valid_positions[:, 1] * weights)
        
        return [float(robust_x), float(robust_y)]
