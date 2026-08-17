import os
import random
import shutil
import json
import argparse
from cli.commands.simulate import handle_simulate
from cli.commands.match import handle_match
from cli.commands.evaluate import handle_evaluate

def add_pipeline_parser(subparsers):
    """Agrega el parser para el comando pipeline."""
    p = subparsers.add_parser("pipeline", help="Ejecuta el pipeline completo de forma automatizada (simulación, matching y evaluación)")
    p.add_argument("-n", "--name", default=None, help="Nombre de la trayectoria o familia de trayectorias (default: auto-generado)")
    p.add_argument("-o", "--ortho", default=None, help="Nombre de la ortofoto (sin .tif) en 'ortophotos/'. Si no se indica, se elige una al azar.")
    p.add_argument("-db", "--tile-db", default="10_10", help="Base de datos de teselas a usar (default: 10_10)")
    p.add_argument("-t", "--type", choices=["straight", "curve", "lemniscate"], default=None, help="Tipo de ruta. Si no se indica, es aleatorio en modo familia o 'curve' en run único.")
    p.add_argument("-m", "--method", choices=["sift", "orb", "superpoint_lightglue"], default="sift", help="Algoritmo de matching (default: sift)")
    p.add_argument("--runs", type=int, default=1, help="Número de trayectorias a generar y evaluar (crea una familia) (default: 1)")
    p.add_argument("-s", "--speed", type=float, default=15.0, help="Velocidad del dron en m/s (default: 15.0)")
    p.add_argument("-f", "--freq", type=float, default=1.0, help="Frecuencia de captura en Hz (default: 1.0)")
    p.add_argument("-a", "--angle", type=float, default=0.0, help="Ángulo de rotación inicial de la trayectoria en grados")
    p.set_defaults(func=handle_pipeline)

def get_random_ortho():
    import glob
    tifs = glob.glob(os.path.join("ortophotos", "*.tif"))
    if not tifs:
        return None
    return os.path.splitext(os.path.basename(random.choice(tifs)))[0]

def handle_pipeline(args):
    """Manejador del comando pipeline."""
    ortho_name = args.ortho or get_random_ortho()
    
    if not ortho_name:
        print("Error: No se encontró ninguna ortofoto en el directorio 'ortophotos/'.")
        return
        
    ortho_path = os.path.join("ortophotos", f"{ortho_name}.tif")
    tiles_json = os.path.join("ortophotos", "sections", ortho_name, args.tile_db, f"{args.tile_db}_metadata.json")
    if not os.path.exists(tiles_json):
        # Fallback a directorio antiguo
        alt = os.path.join("ortophotos", "sections", args.tile_db, f"{args.tile_db}_metadata.json")
        if os.path.exists(alt):
            tiles_json = alt
            
    if not os.path.exists(ortho_path) or not os.path.exists(tiles_json):
        print(f"Error: Falta la ortofoto '{ortho_path}' o el JSON de teselas '{tiles_json}'.")
        return

    print(f"--- INICIANDO PIPELINE AUTOMÁTICO ---")
    print(f"Ortofoto: '{ortho_name}', Runs: {args.runs}, Algoritmo: '{args.method}'")
    
    if args.runs == 1:
        traj_name = args.name or "test_run"
        sim_args = argparse.Namespace(
            ortho=ortho_path, tiles_json=tiles_json, name=traj_name, type=args.type or "curve",
            start_x=None, start_y=None, end_x=None, end_y=None, speed=args.speed, freq=args.freq,
            out_w=1920, out_h=1080, laps=1,
            angle=args.angle
        )
        print("\n[Paso 1/3] Simulación")
        handle_simulate(sim_args)
        
        meta_path = os.path.join("trajectories", traj_name, f"{traj_name}_metadata.json")
        match_args = argparse.Namespace(target=meta_path, tiles=tiles_json, method=args.method, threshold=100.0, force=True)
        print("\n[Paso 2/3] Emparejamiento")
        handle_match(match_args)
        
        eval_args = argparse.Namespace(target=os.path.join("trajectories", traj_name), mode="single", outlier_filter="none", iqr=1.5, mad_threshold=3.5, shift="(0,0)", method=args.method)
        print("\n[Paso 3/3] Evaluación")
        handle_evaluate(eval_args)
    else:
        family_name = args.name or "family_run"
        family_dir = os.path.join("trajectories", family_name)
        os.makedirs(family_dir, exist_ok=True)
        
        for i in range(args.runs):
            print(f"\n--- Ejecución {i+1}/{args.runs} ---")
            traj_id = str(i)
            final_dir = os.path.join(family_dir, traj_id)
            if os.path.exists(final_dir): shutil.rmtree(final_dir)
            
            sim_args = argparse.Namespace(
                ortho=ortho_path, tiles_json=tiles_json, name=traj_id, type=args.type or random.choice(["straight", "curve", "lemniscate"]),
                start_x=None, start_y=None, end_x=None, end_y=None, speed=15.0, freq=1.0,
                out_w=1920, out_h=1080, laps=1,
                angle=args.angle or random.uniform(0, 360),
                out_dir=final_dir
            )
            handle_simulate(sim_args)
            
            meta_path = os.path.join(final_dir, f"{traj_id}_metadata.json")
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            meta["trajectory_metadata"]["name"] = f"{traj_id}_{family_name}"
            meta["trajectory_metadata"]["database_path"] = final_dir.replace('\\', '/')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=4)
                
            match_args = argparse.Namespace(target=meta_path, tiles=tiles_json, method=args.method, threshold=100.0, force=True)
            handle_match(match_args)
            
            eval_single = argparse.Namespace(target=final_dir, mode="single", outlier_filter="none", iqr=1.5, mad_threshold=3.5, shift="(0,0)", method=args.method)
            handle_evaluate(eval_single)
            
        print("\n--- Evaluación global de Familia ---")
        eval_batch = argparse.Namespace(target=family_dir, mode="batch", outlier_filter="iqr", iqr=1.5, mad_threshold=3.5, shift="(0,0)", method=args.method)
        handle_evaluate(eval_batch)
        print("Pipeline de familia completado con éxito.")
