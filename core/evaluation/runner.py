"""
Módulo Evaluador.

Este módulo contiene la lógica para calcular las métricas estadísticas de precisión
del algoritmo de Odometría Visual y Refinamiento comparando las estimaciones contra
el Ground Truth generado durante la simulación.
Permite evaluar trayectorias individuales, familias de trayectorias y conjuntos globales,
con soporte para extracción robusta de outliers usando rangos intercuartílicos (IQR).
Soporta múltiples métodos de emparejamiento almacenados en el mismo matching_results.json.
"""

import json
from core.evaluation.metrics import compute_stats, filter_outliers, compute_robust_shift
import numpy as np
import os
import glob

class Evaluator:
    """
    Clase encargada de evaluar los resultados del pipeline de emparejamiento.
    
    Attributes:
        base_dir (str): Directorio base donde se encuentran las trayectorias a evaluar.
    """

    def __init__(self, base_dir: str):
        """
        Inicializa el evaluador.

        Args:
            base_dir (str): Ruta al directorio base (puede ser de una trayectoria única,
                una familia, o la carpeta global 'trajectories').
        """
        self.base_dir = base_dir

    def _load_json(self, path: str) -> dict:
        """Carga un archivo JSON."""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, data: dict, path: str):
        """Guarda un diccionario en formato JSON."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def _is_legacy_format(self, data: dict) -> bool:
        """Detecta si un matching_results.json usa el formato antiguo (plano)."""
        return "results" in data and "matching_method" in data

    def _get_method_data(self, results_meta: dict, method: str) -> dict:
        """
        Obtiene los datos de un método concreto, soportando ambos formatos.

        Args:
            results_meta (dict): Contenido del matching_results.json.
            method (str): Nombre del método a buscar.

        Returns:
            dict: Datos del método (con clave "results"), o dict vacío si no existe.
        """
        if self._is_legacy_format(results_meta):
            # Formato antiguo: solo hay un método
            if results_meta.get("matching_method") == method:
                return results_meta
            return {}
        # Formato nuevo: claves por método
        return results_meta.get(method, {})

    def get_available_methods(self, traj_dir: str = None) -> list:
        """
        Lista los métodos de emparejamiento disponibles en un directorio.
        
        Si se proporciona un directorio de trayectoria individual, busca en su
        matching_results.json. Si no se proporciona, busca recursivamente en base_dir.

        Args:
            traj_dir (str, optional): Directorio de una trayectoria individual.
                Si es None, se usa self.base_dir.

        Returns:
            list: Lista de nombres de métodos disponibles.
        """
        search_dir = traj_dir or self.base_dir
        methods = set()
        
        # Buscar en el propio directorio
        results_path = os.path.join(search_dir, "matching_results.json")
        if os.path.exists(results_path):
            try:
                data = self._load_json(results_path)
                if self._is_legacy_format(data):
                    methods.add(data["matching_method"])
                else:
                    methods.update(data.keys())
            except Exception:
                pass
        
        # Buscar recursivamente en subdirectorios
        if not methods or traj_dir is None:
            for root, dirs, files in os.walk(search_dir):
                if "matching_results.json" in files:
                    rp = os.path.join(root, "matching_results.json")
                    try:
                        data = self._load_json(rp)
                        if self._is_legacy_format(data):
                            methods.add(data["matching_method"])
                        else:
                            methods.update(data.keys())
                    except Exception:
                        pass
        
        return sorted(methods)

    def _extract_results_from_dir(self, traj_dir: str, method: str = None, shift = (0.0, 0.0), mad_threshold: float = 3.5) -> tuple:
        """
        Extrae los resultados calculados previamente en un directorio de trayectoria
        sin sobrescribir ni re-evaluar datos.

        Args:
            traj_dir (str): Ruta a la trayectoria individual.
            method (str, optional): Método de emparejamiento a extraer. Si es None,
                se usa el primer método disponible (retrocompatibilidad).
            shift: Tupla (x,y) o string ('mean', 'median').

        Returns:
            tuple: (Lista de diccionarios con resultados, shift_aplicado)
        """
        results_path = os.path.join(traj_dir, "matching_results.json")
        traj_name = os.path.basename(os.path.normpath(traj_dir))
        metadata_path = os.path.join(traj_dir, f"{traj_name}_metadata.json")
        
        if not os.path.exists(results_path) or not os.path.exists(metadata_path):
            return [], (0.0, 0.0)

        try:
            results_meta = self._load_json(results_path)
            traj_meta = self._load_json(metadata_path)
        except json.JSONDecodeError as e:
            return [], (0.0, 0.0)

        # Obtener datos del método solicitado
        if method:
            method_data = self._get_method_data(results_meta, method)
        elif self._is_legacy_format(results_meta):
            method_data = results_meta
        else:
            keys = list(results_meta.keys())
            method_data = results_meta[keys[0]] if keys else {}

        if not method_data or "results" not in method_data:
            return [], (0.0, 0.0)

        gt_map = {img["name"]: img["center_global"] for img in traj_meta.get("images", [])}
        raw_results = []

        for res in method_data.get("results", []):
            img_name = res.get("query_image")
            if img_name not in gt_map:
                continue

            gt_pos = gt_map[img_name]
            est = res.get("refined_global_pos") or res.get("estimated_global_pos_odometry")
            
            if not est or len(est) != 2:
                continue

            err_x_raw = est[0] - gt_pos[0]
            err_y_raw = est[1] - gt_pos[1]
            
            time_val = res.get("timestamp", 0.0)
            if time_val == 0.0:
                try:
                    time_val = float(img_name.split('s_')[0])
                except ValueError:
                    pass

            raw_results.append({
                "time": time_val,
                "img_name": img_name,
                "gt_pos": gt_pos,
                "est_pos": [est[0], est[1]],
                "err_x_raw": err_x_raw,
                "err_y_raw": err_y_raw,
                "elapsed_seconds": res.get("elapsed_seconds", None)
            })

        if not raw_results:
            return [], (0.0, 0.0)
            
        if isinstance(shift, str):
            errs_x = [r["err_x_raw"] for r in raw_results]
            errs_y = [r["err_y_raw"] for r in raw_results]
            applied_shift = compute_robust_shift(errs_x, errs_y, shift_type=shift, mad_threshold=mad_threshold)
        else:
            applied_shift = (float(shift[0]), float(shift[1]))

        valid_results = []
        for r in raw_results:
            r["est_pos"][0] = r["est_pos"][0] - applied_shift[0]
            r["est_pos"][1] = r["est_pos"][1] - applied_shift[1]
            
            r["err_x"] = r["est_pos"][0] - r["gt_pos"][0]
            r["err_y"] = r["est_pos"][1] - r["gt_pos"][1]
            r["err_dist"] = np.sqrt(r["err_x"]**2 + r["err_y"]**2)
            valid_results.append(r)

        valid_results.sort(key=lambda x: x["time"])
        return valid_results, applied_shift

    def _save_stats_to_metadata(self, traj_dir: str, method: str, outlier_filter: str, stats: dict, shift = (0.0, 0.0)):
        """Guarda las estadísticas globales de la evaluación en los metadatos de la trayectoria."""
        traj_name = os.path.basename(os.path.normpath(traj_dir))
        metadata_path = os.path.join(traj_dir, f"{traj_name}_metadata.json")
        if not os.path.exists(metadata_path):
            return
        try:
            meta = self._load_json(metadata_path)
            stats_key = f"statistics_outliers_{outlier_filter}" if outlier_filter != "none" else "statistics"
            if shift != (0.0, 0.0):
                stats_key += "_shifted"
            target_method = method or "default"
            
            if "evaluation_stats" not in meta:
                meta["evaluation_stats"] = {}
                
            if target_method not in meta["evaluation_stats"]:
                meta["evaluation_stats"][target_method] = {}
                
            meta["evaluation_stats"][target_method][stats_key] = stats
            
            self._save_json(meta, metadata_path)
        except Exception as e:
            pass  # print en {metadata_path}: {e}")

    def evaluate_single(self, method: str = None, outlier_filter: str = "none", iqr_multiplier: float = 1.5, mad_threshold: float = 3.5, shift = (0.0, 0.0)) -> tuple:
        """
        Evalúa una única trayectoria localizada en `self.base_dir`.

        Args:
            method (str, optional): Método de emparejamiento a evaluar. Si None, 
                se usa el primer método disponible.
            outlier_filter (str): Tipo de filtro ('none', 'iqr', 'mad').
            iqr_multiplier (float): Multiplicador IQR para determinar outliers si filtro es iqr.
            mad_threshold (float): Umbral de Z-Score modificado si filtro es mad.

        Returns:
            tuple: (filtered_results, stats) o None si falla.
        """
        valid_results, applied_shift = self._extract_results_from_dir(self.base_dir, method=method, shift=shift, mad_threshold=mad_threshold)
        if not valid_results:
            method_str = f" (método: {method})" if method else ""
            # print en {self.base_dir}{method_str}")
            return None

        filtered_results = filter_outliers(valid_results, outlier_filter=outlier_filter, iqr_multiplier=iqr_multiplier, mad_threshold=mad_threshold)

        if not filtered_results:
            # print outliers en {self.base_dir}")
            return None

        e_x = [r["err_x"] for r in filtered_results]
        e_y = [r["err_y"] for r in filtered_results]
        e_dist_filtered = [r["err_dist"] for r in filtered_results]
        timing = [r["elapsed_seconds"] for r in filtered_results if r.get("elapsed_seconds") is not None]

        stats = compute_stats(e_x, e_y, e_dist_filtered, timing_seconds=timing or None)
        stats["shift_applied"] = shift if isinstance(shift, str) else list(shift)
        stats["shift_tuple"] = list(applied_shift)

        # Guardar en archivo resumen sin modificar matching_results.json
        target_name = os.path.basename(os.path.normpath(self.base_dir))
        suffix = f"_outliers_{outlier_filter}" if outlier_filter != "none" else ""
        if isinstance(shift, str):
            suffix += f"_shifted_{shift}"
        elif shift != (0.0, 0.0):
            suffix += "_shifted_custom"
        method_suffix = f"_{method}" if method else ""
        
        summary = {
            "method": method or "default",
            "trajectory_count": 1,
            "images_total": len(valid_results),
            "images_filtered": len(filtered_results),
            "outliers_removed": len(valid_results) - len(filtered_results),
            "outlier_filter": outlier_filter,
            "statistics": stats
        }
        out_file = os.path.join(self.base_dir, f"{target_name}_stats_summary{method_suffix}{suffix}.json")
        try:
            self._save_json(summary, out_file)
        except Exception:
            pass

        # Guardar también en los metadatos de la trayectoria (JSON global)
        self._save_stats_to_metadata(self.base_dir, method, outlier_filter, stats, shift=shift)

        return filtered_results, stats

    def _guess_trajectory_type(self, traj_name, images):
        name_lower = traj_name.lower()
        if 'lemn' in name_lower:
            return 'lemniscate'
        if 'straight' in name_lower:
            return 'straight'
        if 'curve' in name_lower:
            return 'curve'
        
        # Mathematical fallback
        if not images:
            return 'curve'
            
        try:
            dirs = [(img['direction_vector'][0], img['direction_vector'][1]) for img in images if 'direction_vector' in img]
            if not dirs:
                return 'curve'
                
            dir_x = dirs[0][0]
            dir_y = dirs[0][1]
            is_straight = all(abs(d[0] - dir_x) < 1e-4 and abs(d[1] - dir_y) < 1e-4 for d in dirs)
            
            if is_straight:
                return 'straight'
        except Exception:
            pass
            
        return 'curve'

    def _core_evaluate_batch(self, method: str = None, outlier_filter: str = "none", iqr_multiplier: float = 1.5, mad_threshold: float = 3.5, shift = (0.0, 0.0)) -> tuple:
        """
        Lógica central que evalúa múltiples trayectorias, devuelve res_filtrados, stats, ortho, y grouped_data.
        """
        traj_dirs = []
        for root, dirs, files in os.walk(self.base_dir):
            if "matching_results.json" in files:
                traj_dirs.append(root)

        if not traj_dirs:
            # print en {self.base_dir}")
            return None, None, None, None

        all_raw_results = []
        time_offset = 0.0
        ortho_path = None
        
        traj_data_list = []

        for t_dir in traj_dirs:
            traj_type = "curve" # default
            # Intentar deducir la ortofoto y extraer traj_type
            meta_files = glob.glob(os.path.join(t_dir, "*_metadata.json"))
            if meta_files:
                try:
                    meta = self._load_json(meta_files[0])
                    traj_meta = meta.get("trajectory_metadata", {})
                    
                    if ortho_path is None:
                        ortho_name = traj_meta.get("orthophoto_name")
                        if ortho_name:
                            candidate = os.path.join("ortophotos", f"{ortho_name}.tif")
                            if os.path.exists(candidate):
                                ortho_path = candidate
                                
                    traj_type = traj_meta.get("traj_type")
                    if not traj_type:
                        traj_name = traj_meta.get("name", os.path.basename(os.path.normpath(t_dir)))
                        traj_type = self._guess_trajectory_type(traj_name, meta.get("images", []))
                        # Save deduced type
                        meta.setdefault("trajectory_metadata", {})["traj_type"] = traj_type
                        self._save_json(meta, meta_files[0])
                except Exception:
                    pass

            traj_name = os.path.relpath(t_dir, self.base_dir)
            # Extraemos resultados sin aplicar shift todavía (se aplica globalmente)
            res, _ = self._extract_results_from_dir(t_dir, method=method, shift=(0.0, 0.0), mad_threshold=mad_threshold)
            if res:
                traj_data_list.append((t_dir, traj_name, traj_type, res))
                all_raw_results.extend(res)

        if not all_raw_results:
            return None, None, None, None

        # Calcular el shift global sin outliers de MAD
        if isinstance(shift, str):
            errs_x = [r["err_x_raw"] for r in all_raw_results]
            errs_y = [r["err_y_raw"] for r in all_raw_results]
            global_applied_shift = compute_robust_shift(errs_x, errs_y, shift_type=shift, mad_threshold=mad_threshold)
        else:
            global_applied_shift = (float(shift[0]), float(shift[1]))

        all_results = []
        for t_dir, traj_name, traj_type, res in traj_data_list:
            # Aplicar el shift global a esta trayectoria
            for r in res:
                r["est_pos"][0] = r["est_pos"][0] - global_applied_shift[0]
                r["est_pos"][1] = r["est_pos"][1] - global_applied_shift[1]
                r["err_x"] = r["est_pos"][0] - r["gt_pos"][0]
                r["err_y"] = r["est_pos"][1] - r["gt_pos"][1]
                r["err_dist"] = np.sqrt(r["err_x"]**2 + r["err_y"]**2)

            # Calcular y guardar estadísticas individuales para esta trayectoria
            t_filtered = filter_outliers(res, outlier_filter=outlier_filter, iqr_multiplier=iqr_multiplier, mad_threshold=mad_threshold)
            if t_filtered:
                t_e_x = [r["err_x"] for r in t_filtered]
                t_e_y = [r["err_y"] for r in t_filtered]
                t_e_d = [r["err_dist"] for r in t_filtered]
                t_timing = [r["elapsed_seconds"] for r in t_filtered if r.get("elapsed_seconds") is not None]
                t_stats = compute_stats(t_e_x, t_e_y, t_e_d, timing_seconds=t_timing or None)
                t_stats["shift_applied"] = shift if isinstance(shift, str) else list(shift)
                t_stats["shift_tuple"] = list(global_applied_shift)
                self._save_stats_to_metadata(t_dir, method, outlier_filter, t_stats, shift=shift)

            max_t = 0.0
            for r in res:
                r_copy = dict(r)
                r_copy["time"] += time_offset
                r_copy["traj_name"] = traj_name
                r_copy["traj_type"] = traj_type
                all_results.append(r_copy)
                if r["time"] > max_t:
                    max_t = r["time"]
            time_offset += max_t + 1.0

        if not all_results:
            return None, None, None, None

        filtered_results = filter_outliers(all_results, outlier_filter=outlier_filter, iqr_multiplier=iqr_multiplier, mad_threshold=mad_threshold)

        if not filtered_results:
            return None, None, None, None

        e_x = [r["err_x"] for r in filtered_results]
        e_y = [r["err_y"] for r in filtered_results]
        e_d = [r["err_dist"] for r in filtered_results]
        timing = [r["elapsed_seconds"] for r in filtered_results if r.get("elapsed_seconds") is not None]

        stats = compute_stats(e_x, e_y, e_d, timing_seconds=timing or None)
        stats["shift_applied"] = shift if isinstance(shift, str) else list(shift)
        stats["shift_tuple"] = list(global_applied_shift)
        
        # Agrupar por tipo (todos y filtrados) para conteos
        grouped_all = {}
        for r in all_results:
            ttype = r.get("traj_type", "curve")
            if ttype not in grouped_all:
                grouped_all[ttype] = []
            grouped_all[ttype].append(r)
            
        grouped_filtered = {}
        for r in filtered_results:
            ttype = r.get("traj_type", "curve")
            if ttype not in grouped_filtered:
                grouped_filtered[ttype] = []
            grouped_filtered[ttype].append(r)

        subsets_stats = {}
        grouped_data = {}
        for ttype, t_results in grouped_filtered.items():
            t_e_x = [r["err_x"] for r in t_results]
            t_e_y = [r["err_y"] for r in t_results]
            t_e_d = [r["err_dist"] for r in t_results]
            t_timing = [r["elapsed_seconds"] for r in t_results if r.get("elapsed_seconds") is not None]
            
            stats_dict = compute_stats(t_e_x, t_e_y, t_e_d, timing_seconds=t_timing or None)
            
            t_all = grouped_all.get(ttype, [])
            traj_names = set(r.get("traj_name") for r in t_all if r.get("traj_name"))
            
            subsets_stats[ttype] = {
                "trajectory_count": len(traj_names),
                "images_total": len(t_all),
                "images_filtered": len(t_results),
                "outliers_removed": len(t_all) - len(t_results),
                "statistics": stats_dict
            }
            # Store in grouped_data as tuple (results, stats)
            grouped_data[ttype] = (t_results, stats_dict)

        # Calculate worst trajectories
        traj_errors = {}
        for r in filtered_results:
            t_name = r.get("traj_name")
            if t_name:
                if t_name not in traj_errors:
                    traj_errors[t_name] = []
                traj_errors[t_name].append(r["err_dist"])
        
        traj_avg_errors = []
        for t_name, errors in traj_errors.items():
            avg_error = float(np.mean(errors))
            traj_avg_errors.append({"trajectory": t_name, "avg_error": avg_error})
        
        traj_avg_errors.sort(key=lambda x: x["avg_error"], reverse=True)
        top_worst_trajectories = traj_avg_errors[:3]

        # Resumen informativo
        method_label = method or "all"
        summary = {
            "method": method_label,
            "trajectory_count": len(traj_dirs),
            "images_total": len(all_results),
            "images_filtered": len(filtered_results),
            "outliers_removed": len(all_results) - len(filtered_results),
            "outlier_filter": outlier_filter,
            "statistics": stats,
            "top_worst_trajectories": top_worst_trajectories,
            "subsets": subsets_stats
        }

        # Guardar archivo resumen en la base
        target_name = os.path.basename(os.path.normpath(self.base_dir))
        if target_name == "trajectories":
            target_name = "global"
        
        suffix = f"_outliers_{outlier_filter}" if outlier_filter != "none" else ""
        if isinstance(shift, str):
            suffix += f"_shifted_{shift}"
        elif shift != (0.0, 0.0):
            suffix += "_shifted_custom"
        method_suffix = f"_{method}" if method else ""
        out_file = os.path.join(self.base_dir, f"{target_name}_stats_summary{method_suffix}{suffix}.json")
        self._save_json(summary, out_file)
        
        return filtered_results, stats, ortho_path, grouped_data

    def evaluate_batch(self, method: str = None, outlier_filter: str = "none", iqr_multiplier: float = 1.5, mad_threshold: float = 3.5, shift = (0.0, 0.0)) -> tuple:
        """
        Evalúa múltiples trayectorias (familia o global) de forma recursiva.
        
        Devuelve:
            tuple: (filtered_results, stats_dict, ortho_path_guess)
        """
        res = self._core_evaluate_batch(method=method, outlier_filter=outlier_filter, iqr_multiplier=iqr_multiplier, mad_threshold=mad_threshold, shift=shift)
        if res[0] is None:
            return None, None, None
        return res[0], res[1], res[2]
        
    def evaluate_batch_with_subsets(self, method: str = None, outlier_filter: str = "none", iqr_multiplier: float = 1.5, mad_threshold: float = 3.5, shift = (0.0, 0.0)) -> tuple:
        """
        Evalúa múltiples trayectorias de forma recursiva, agrupando resultados por tipo.
        
        Devuelve:
            tuple: (filtered_results, stats_dict, ortho_path_guess, grouped_data)
        """
        return self._core_evaluate_batch(method=method, outlier_filter=outlier_filter, iqr_multiplier=iqr_multiplier, mad_threshold=mad_threshold, shift=shift)
