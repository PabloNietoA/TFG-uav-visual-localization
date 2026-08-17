import os
import json
from core.matching.batch_processor import BatchMatcher

def add_match_parser(subparsers):
    """Agrega el parser para el comando match."""
    p = subparsers.add_parser("match", help="Empareja imágenes simuladas con la base de datos de teselas")
    p.add_argument("-t", "--target", required=True, help="Directorio de la trayectoria (familia) o archivo JSON de metadatos")
    p.add_argument("-d", "--tiles", required=True, help="Ruta al JSON de metadatos de las teselas")
    p.add_argument("-m", "--method", choices=["sift", "orb", "superpoint_lightglue"], default="sift", help="Algoritmo de matching a usar (default: sift)")
    p.add_argument("-th", "--threshold", type=float, default=100.0, help="Distancia umbral para filtrar teselas candidatas en metros (default: 100.0)")
    p.add_argument("-f", "--force", action="store_true", help="Si se especifica, fuerza el emparejamiento sobreescribiendo resultados anteriores")
    p.set_defaults(func=handle_match)

def _find_trajectory_metadata_files(directory: str) -> list:
    """Busca recursivamente archivos *_metadata.json en un directorio."""
    metadata_files = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith("_metadata.json"):
                metadata_files.append(os.path.join(root, f))
    return metadata_files

def handle_match(args):
    """Manejador del comando match."""
    target = args.target
    if not os.path.exists(target):
        print(f"Error: No se encontró el target '{target}'")
        return
    if not os.path.exists(args.tiles):
        print(f"Error: No se encontró la base de datos de teselas '{args.tiles}'")
        return

    print(f"Iniciando proceso de matching con el algoritmo '{args.method}'...")
    matcher = BatchMatcher(args.tiles, method=args.method, threshold_m=args.threshold)

    def is_already_matched(trajectory_dir):
        if args.force:
            return False
        results_file = os.path.join(trajectory_dir, "matching_results.json")
        if not os.path.exists(results_file):
            return False
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                res_data = json.load(f)
            # Detectar ambos formatos
            if args.method in res_data:
                return True
            if "matching_method" in res_data and res_data["matching_method"] == args.method:
                return True
        except Exception:
            pass
        return False

    if os.path.isdir(target):
        # Target es un directorio: buscar múltiples trayectorias
        metadata_files = _find_trajectory_metadata_files(target)
        if not metadata_files:
            print(f"Error: No se encontraron trayectorias (archivos *_metadata.json) en '{target}'")
            return
        
        print(f"Se encontraron {len(metadata_files)} trayectoria(s) en '{target}'.")
        for i, meta_path in enumerate(metadata_files, 1):
            traj_dir = os.path.dirname(meta_path)
            traj_name = os.path.basename(traj_dir)
            
            if is_already_matched(traj_dir):
                print(f"[{i}/{len(metadata_files)}] Omitiendo '{traj_name}': ya fue procesado con '{args.method}' (usa --force para reescribir)")
                continue
                
            print(f"[{i}/{len(metadata_files)}] Procesando trayectoria: '{traj_name}'...")
            results_path, _ = matcher.match_trajectory(meta_path)
            if results_path:
                print(f"  -> Resultados guardados en '{results_path}'")
    else:
        # Target es directamente un JSON de metadatos
        traj_dir = os.path.dirname(target)
        if is_already_matched(traj_dir):
            print(f"Omitiendo matching: la trayectoria ya fue procesada con el algoritmo '{args.method}'. Usa --force para reescribir.")
            return
            
        results_path, _ = matcher.match_trajectory(target)
        if results_path:
            print(f"Matching finalizado. Resultados en '{results_path}'")
