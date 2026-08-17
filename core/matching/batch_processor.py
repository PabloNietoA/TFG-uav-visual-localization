"""
Módulo Batch Matcher.

Este módulo se encarga del cruce principal entre una trayectoria (imágenes de un dron) 
y una base de datos de teselas (ortofotos troceadas). Implementa odometría visual, 
filtrado heurístico por radio, emparejamiento de características y refinamiento en vivo.
"""

import json
import os
import time
import cv2
import numpy as np
import pickle
import torch
from tqdm import tqdm

from core.matching.extractors import SIFTExtractor, ORBExtractor
from core.matching.matchers import DeepExtractorMatcher, ClassicalMatcher
from .odometry import VisualOdometry
from .location_refinement import LocationRefiner
from utils.io import serialize_keypoints, deserialize_keypoints

class BatchMatcher:
    """
    Clase orquestadora para el emparejamiento masivo de imágenes de un vuelo contra
    una base de datos de teselas georreferenciadas.
    
    Attributes:
        method (str): Método de extracción y emparejamiento (ej. 'sift', 'superpoint_lightglue').
        threshold_m (float): Radio estricto en metros para clustering DBSCAN en la consolidación.
        tiles_metadata (dict): Metadatos completos de la base de datos de teselas.
        scale_m_per_px (float): Escala espacial global (metros por píxel).
    """

    def __init__(self, tiles_json_path: str, method: str = "sift", threshold_m: float = 100.0):
        """
        Inicializa el emparejador cargando los metadatos de teselas y los extractores.

        Args:
            tiles_json_path (str): Ruta al archivo JSON con los metadatos de las teselas.
            method (str): Nombre del algoritmo a usar ("sift", "orb", "superpoint_lightglue").
            threshold_m (float): Radio para agrupar candidatos (clustering DBSCAN).
        """
        self.method = method
        self.threshold_m = threshold_m
        self.tiles_metadata = self._load_json(tiles_json_path)
        self.tiles_base_dir = os.path.dirname(tiles_json_path)
        
        ortho_meta = self.tiles_metadata["orthophoto_metadata"]
        self.scale_m_per_px = ortho_meta["width_m"] / ortho_meta["width_px"]
        self.tiles_info = self.tiles_metadata["tiles"]
        
        # Inicializar extractores de características con caché en disco
        extractor_path = os.path.join(self.tiles_base_dir, "keypoints", f"{method}_extractor.pkl")
        
        loaded = False
        if os.path.exists(extractor_path):
            try:
                with open(extractor_path, 'rb') as f:
                    data = pickle.load(f)
                self.extractor = data.get("extractor")
                self.matcher = data.get("matcher")
                self.extractor_matcher = data.get("extractor_matcher")
                
                # Si es un dict tipo Kornia, debemos reinicializarlo
                if isinstance(self.extractor_matcher, dict) and self.extractor_matcher.get("type") == "kornia_lightglue_superpoint":
                    self.extractor_matcher = DeepExtractorMatcher()
                    
                loaded = True
            except Exception as e:
                print(f"Aviso: Error cargando extractor caché {extractor_path}, se reinicializará. Error: {e}")
        
        if not loaded:
            if method == "sift":
                self.extractor = SIFTExtractor()
                self.matcher = ClassicalMatcher("sift")
            elif method == "orb":
                self.extractor = ORBExtractor()
                self.matcher = ClassicalMatcher("orb")
            elif method == "superpoint_lightglue":
                self.extractor_matcher = DeepExtractorMatcher()
                self.extractor = None
                self.matcher = None
            else:
                raise ValueError(f"Método de emparejamiento desconocido: {method}")
            
            # Guardar extractor en caché
            os.makedirs(os.path.dirname(extractor_path), exist_ok=True)
            try:
                # Para kornia guardamos un dict descriptivo porque picklear un modelo PyTorch directamente puede dar problemas
                extractor_matcher_to_save = self.extractor_matcher
                if method == "superpoint_lightglue":
                    extractor_matcher_to_save = {"type": "kornia_lightglue_superpoint"}
                    
                with open(extractor_path, 'wb') as f:
                    pickle.dump({
                        "extractor": getattr(self, "extractor", None),
                        "matcher": getattr(self, "matcher", None),
                        "extractor_matcher": extractor_matcher_to_save
                    }, f)
            except Exception as e:
                print(f"Aviso: No se pudo guardar extractor caché en {extractor_path}. Error: {e}")

    def _load_json(self, path: str) -> dict:
        """Carga y parsea un archivo JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, path: str, data: dict):
        """Serializa y guarda un diccionario en formato JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def _get_or_compute_keypoints(self, img_path: str, save_dir: str, base_name: str) -> tuple:
        """
        Extrae los puntos de interés. Si es una tesela (base de datos), 
        usa o crea un caché en disco. Si es imagen de trayectoria, los calcula al vuelo.
        """
        is_database_tile = (save_dir == self.tiles_base_dir)
        kp_dir = os.path.join(save_dir, "keypoints")
        cache_file = os.path.join(kp_dir, f"{base_name}_{self.method}_keypoints.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Intentar cargar desde caché si es tesela
        if is_database_tile and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return deserialize_keypoints(data["keypoints"], device=device)
            except Exception as e:
                print(f"Aviso: Error cargando caché {cache_file}, se recalculará. Error: {e}")
        
        # Extraer características
        curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if curr_img is None:
            return None, None
            
        if self.method != "superpoint_lightglue":
            kp, des = self.extractor.extract(curr_img)
        else:
            kp, des = self.extractor_matcher.extract(curr_img)
            
        # Guardar en caché si es tesela
        if is_database_tile:
            os.makedirs(kp_dir, exist_ok=True)
            data_to_save = {
                "keypoints": serialize_keypoints(kp, des)
            }
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_to_save, f)
            except Exception as e:
                print(f"Aviso: No se pudo guardar caché en {cache_file}. Error: {e}")
                
        return kp, des

    def filter_tiles_by_radius(self, estimated_pos_global: tuple, radius_m: float = 3000.0) -> list:
        """
        Filtra las teselas base, retornando solo las que intersecten con un radio de 
        seguridad trazado desde la posición actual estimada.

        Args:
            estimated_pos_global (tuple): Posición actual (X, Y) estimada.
            radius_m (float): Radio de búsqueda en metros.

        Returns:
            list: Lista de diccionarios de teselas candidatas.
        """
        candidates = []
        for tile in self.tiles_info:
            tile_w_m = tile["width_px"] * self.scale_m_per_px
            tile_h_m = tile["height_px"] * self.scale_m_per_px
            
            # Y global disminuye hacia el sur
            cx = tile["origin_global"][0] + tile_w_m / 2
            cy = tile["origin_global"][1] - tile_h_m / 2
            
            dist = np.sqrt((estimated_pos_global[0] - cx)**2 + (estimated_pos_global[1] - cy)**2)
            
            if dist < radius_m + max(tile_w_m, tile_h_m) / 2:
                candidates.append(tile)
                
        # Fallback: si por deriva brutal nos salimos de todas, no filtrar
        return candidates if candidates else self.tiles_info

    def match_trajectory(self, trajectory_json_path: str) -> tuple:
        """
        Ejecuta el pipeline completo de emparejamiento para una trayectoria.

        Args:
            trajectory_json_path (str): Ruta al JSON de la trayectoria a procesar.

        Returns:
            tuple: (ruta_resultados_json, ruta_odometria_json)
        """
        traj_meta = self._load_json(trajectory_json_path)
        traj_dir = os.path.dirname(trajectory_json_path)
        
        images_info = traj_meta.get("images", [])
        if not images_info:
            print("No hay imágenes en la trayectoria.")
            return None, None
            
        # 1. Inicialización (Ground Truth solo en t=0)
        start_pos = images_info[0]["center_global"]
        start_heading = images_info[0]["direction_vector"]
        
        vo_pure = VisualOdometry(start_pos, start_heading, self.scale_m_per_px)
        vo_refined = VisualOdometry(start_pos, start_heading, self.scale_m_per_px)
        refiner = LocationRefiner(threshold_m=self.threshold_m)
        
        prev_img = prev_kp = prev_des = None
        
        results = []
        pure_odometry_path = [list(start_pos)]
        refined_path_log = [list(start_pos)]
        
        # 2. Iterar imágenes
        pbar = tqdm(images_info, desc="Emparejando trayectoria", unit="img")
        for i, img_info in enumerate(pbar):
            img_start_time = time.perf_counter()
            img_name_no_ext = os.path.splitext(img_info["name"])[0]
            img_path = os.path.join(traj_dir, "images", img_info["name"])
            
            curr_kp, curr_des = self._get_or_compute_keypoints(img_path, traj_dir, img_name_no_ext)
            curr_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if curr_img is None:
                continue
                
            # A. Odometría Visual (t-1 -> t)
            if i > 0:
                if self.method != "superpoint_lightglue":
                    pure_pos = vo_pure.update(prev_kp, prev_des, curr_kp, curr_des, self.matcher.matcher)
                    heuristic_pos = vo_refined.update(prev_kp, prev_des, curr_kp, curr_des, self.matcher.matcher)
                else:
                    sift = cv2.SIFT_create()
                    k1, d1 = sift.detectAndCompute(prev_img, None)
                    k2, d2 = sift.detectAndCompute(curr_img, None)
                    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                    pure_pos = vo_pure.update(k1, d1, k2, d2, flann)
                    heuristic_pos = vo_refined.update(k1, d1, k2, d2, flann)
            else:
                heuristic_pos = start_pos
                pure_pos = start_pos
                
            if i > 0:
                pure_odometry_path.append(list(pure_pos))
            
            # B. Filtrado Espacial y Matching
            candidates = self.filter_tiles_by_radius(heuristic_pos, radius_m=3000)
            pbar.set_postfix({"Est. Pos": f"[{heuristic_pos[0]:.0f}, {heuristic_pos[1]:.0f}]", "Teselas": len(candidates)})
            
            match_scores = []
            for tile in candidates:
                tile_name_no_ext = os.path.splitext(tile["name"])[0]
                tile_path = os.path.join(self.tiles_base_dir, "images", tile["name"])
                
                tile_kp, tile_des = self._get_or_compute_keypoints(tile_path, self.tiles_base_dir, tile_name_no_ext)
                if tile_kp is None:
                    continue
                    
                if self.method != "superpoint_lightglue":
                    res = self.matcher.match(curr_kp, curr_des, tile_kp, tile_des, curr_img.shape)
                else:
                    res = self.extractor_matcher.match_keypoints(curr_kp, curr_des, tile_kp, tile_des, curr_img.shape)
                    
                match_scores.append({
                    "tile_name": tile["name"],
                    "tile_info": tile,
                    "score": float(res.score),
                    "inliers": int(res.inliers),
                    "spatial_coverage": float(res.spatial_coverage),
                    "src_pts": res.inlier_src_pts.tolist() if res.inlier_src_pts is not None else [],
                    "dst_pts": res.inlier_dst_pts.tolist() if res.inlier_dst_pts is not None else [],
                    "estimated_pos": []
                })
                
            match_scores = [m for m in match_scores if m["score"] > 0]
            match_scores.sort(key=lambda x: x["score"], reverse=True)
            top_5 = match_scores[:5]
            
            # C. Refinamiento Global (Homografía + DBSCAN)
            candidate_positions, candidate_scores = [], []
            for m in top_5:
                pos = None
                src_pts, dst_pts = m.get("src_pts", []), m.get("dst_pts", [])
                
                if len(src_pts) >= 4 and len(dst_pts) >= 4:
                    pos = refiner.estimate_position_from_tile(
                        np.float32(src_pts), np.float32(dst_pts), 
                        m["tile_info"], curr_img.shape, self.scale_m_per_px
                    )
                
                m["estimated_pos"] = pos if pos is not None else []
                if pos is not None:
                    candidate_positions.append(pos)
                    candidate_scores.append(m["score"])
                    
            robust_pos = None
            if candidate_positions:
                robust_pos = refiner.filter_outliers_and_average(candidate_positions, candidate_scores, heuristic_pos)
                
            if robust_pos is not None:
                final_pos = robust_pos
                vo_refined.current_pos = np.array(robust_pos, dtype=np.float64)
            else:
                final_pos = heuristic_pos
                
            if i > 0:
                refined_path_log.append(list(final_pos))
                
            clean_top_5 = []
            for m in top_5:
                clean_m = m.copy()
                clean_m.pop("src_pts", None)
                clean_m.pop("dst_pts", None)
                clean_top_5.append(clean_m)

            img_elapsed = time.perf_counter() - img_start_time

            results.append({
                "query_image": img_info["name"],
                "elapsed_seconds": round(img_elapsed, 4),
                "estimated_global_pos_odometry": [float(x) for x in heuristic_pos],
                "pure_odometry_pos": [float(x) for x in pure_pos],
                "refined_global_pos": [float(x) for x in final_pos] if robust_pos is not None else [],
                "top_5_matches": clean_top_5
            })
            
            prev_img, prev_kp, prev_des = curr_img, curr_kp, curr_des
            
        # 3. Guardar resultados (multi-método: merge con existente)
        method_data = {
            "trajectory_name": traj_meta.get("trajectory_metadata", {}).get("name", "unknown"),
            "matching_method": self.method,
            "results": results
        }
        
        out_path = os.path.join(traj_dir, "matching_results.json")
        existing = {}
        if os.path.exists(out_path):
            try:
                existing = self._load_json(out_path)
                # Retrocompatibilidad: si el JSON existente tiene formato antiguo
                # (con clave "results" en la raíz), migrar al nuevo formato
                if "results" in existing and "matching_method" in existing:
                    old_method = existing["matching_method"]
                    existing = {old_method: {
                        "trajectory_name": existing.get("trajectory_name", "unknown"),
                        "matching_method": old_method,
                        "results": existing["results"],
                        **{k: v for k, v in existing.items() 
                           if k not in ("trajectory_name", "matching_method", "results")}
                    }}
            except Exception:
                existing = {}
        
        existing[self.method] = method_data
        self._save_json(out_path, existing)
        
        odom_path_json = os.path.join(traj_dir, "odometry_path.json")
        odom_existing = {}
        if os.path.exists(odom_path_json):
            try:
                odom_existing = self._load_json(odom_path_json)
                # Retrocompatibilidad: formato antiguo con "pure_path" en la raíz
                if "pure_path" in odom_existing:
                    odom_existing = {"_legacy": odom_existing}
            except Exception:
                odom_existing = {}
        
        odom_existing[self.method] = {
            "pure_path": pure_odometry_path,
            "refined_path": refined_path_log
        }
        self._save_json(odom_path_json, odom_existing)
        
        return out_path, odom_path_json
