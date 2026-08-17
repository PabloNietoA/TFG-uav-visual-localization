import os
from core.evaluation.runner import Evaluator
from visualization.evaluation import generate_evaluation_plots, generate_comparison_plots

def add_evaluate_parser(subparsers):
    """Agrega el parser para el comando evaluate."""
    p = subparsers.add_parser("evaluate", help="Evalúa los resultados de matching y genera gráficas")
    p.add_argument("-t", "--target", default=None, help="Directorio objetivo (ej. trajectories/mi_vuelo). Si no se indica, evalúa todo en 'trajectories/'")
    p.add_argument("-mod", "--mode", choices=["single", "batch"], default="single", help="Modo de evaluación: 'single' (trayectoria única) o 'batch' (familia/todas) (default: single)")
    p.add_argument("-m", "--method", choices=["sift", "orb", "superpoint_lightglue"], default=None, help="Método específico a evaluar. Si no se indica, evalúa y compara todos los disponibles.")
    p.add_argument("-of", "--outlier-filter", choices=["none", "iqr", "mad"], default="none", help="Método estadístico robusto de filtrado de outliers: 'none' (sin filtrar), 'iqr' (Rango Intercuartílico), o 'mad' (Z-Score Modificado con MAD) (default: none)")
    p.add_argument("--iqr", type=float, default=1.5, help="Multiplicador IQR para la detección de outliers si el filtro es iqr (default: 1.5)")
    p.add_argument("--mad-threshold", type=float, default=3.5, help="Umbral para el Z-Score modificado con MAD si el filtro es mad (default: 3.5)")
    p.add_argument("-s", "--shift", type=str, default="(0,0)", help="Desplazamiento global a restar a la estimación (x,y), o 'mean'/'median'. Ej: (0,1) (default: (0,0))")
    p.set_defaults(func=handle_evaluate)

def handle_evaluate(args):
    """Manejador del comando evaluate."""
    if args.shift.lower() in ["mean", "median"]:
        shift = args.shift.lower()
    else:
        import ast
        try:
            shift_tuple = ast.literal_eval(args.shift)
            if not isinstance(shift_tuple, tuple) or len(shift_tuple) != 2:
                raise ValueError
            shift = (float(shift_tuple[0]), float(shift_tuple[1]))
        except (ValueError, SyntaxError):
            print("Error: --shift debe ser una tupla de dos números (ej: (0,1)) o 'mean'/'median'.")
            return

    target = args.target if args.target else "trajectories"
    print(f"Iniciando evaluación sobre: '{target}' (Modo: {args.mode}, Filtro de outliers: {args.outlier_filter}, Shift: {shift})")
    
    ev = Evaluator(target)
    
    available_methods = ev.get_available_methods()
    if args.method:
        if args.method not in available_methods:
            print(f"Error: Método '{args.method}' no encontrado. Disponibles: {available_methods}")
            return
        methods_to_eval = [args.method]
    else:
        methods_to_eval = available_methods if available_methods else [None]
    
    print(f"Métodos a evaluar: {methods_to_eval}")
    
    method_results = {}
    
    for method in methods_to_eval:
        method_label = method or "default"
        print(f"\n--- Evaluando método: {method_label} ---")
        
        if args.mode == "single":
            res = ev.evaluate_single(
                method=method,
                outlier_filter=args.outlier_filter,
                iqr_multiplier=args.iqr,
                mad_threshold=args.mad_threshold,
                shift=shift
            )
            if res:
                valid_results, stats = res
                suffix = f"_outliers_{args.outlier_filter}" if args.outlier_filter != "none" else ""
                if isinstance(shift, str):
                    suffix += f"_shifted_{shift}"
                elif shift != (0.0, 0.0):
                    suffix += f"_shifted_custom"
                generate_evaluation_plots(valid_results, stats, target, suffix=suffix, method_prefix=method_label)
                method_results[method_label] = (valid_results, stats)
                print(f"Evaluación 'single' ({method_label}) completada.")
        else:
            res = ev.evaluate_batch_with_subsets(
                method=method,
                outlier_filter=args.outlier_filter,
                iqr_multiplier=args.iqr,
                mad_threshold=args.mad_threshold,
                shift=shift
            )
            if res and res[1]:  # stats exists
                valid_results, stats, ortho_guess, grouped_data = res
                suffix = f"_outliers_{args.outlier_filter}" if args.outlier_filter != "none" else ""
                if isinstance(shift, str):
                    suffix += f"_shifted_{shift}"
                elif shift != (0.0, 0.0):
                    suffix += f"_shifted_custom"
                generate_evaluation_plots(valid_results, stats, target, orthophoto_path=ortho_guess, suffix=suffix, method_prefix=method_label)
                
                # Generar gráficas por subgrupo (tipo de trayectoria)
                if grouped_data:
                    for traj_type, (t_res, t_stats) in grouped_data.items():
                        custom_dir = os.path.join(target, "charts", traj_type)
                        generate_evaluation_plots(t_res, t_stats, target, orthophoto_path=ortho_guess, suffix=suffix, method_prefix=method_label, custom_charts_dir=custom_dir)
                        
                method_results[method_label] = (valid_results, stats)
                print(f"Evaluación 'batch' ({method_label}) completada.")
    
    # Generar gráficas comparativas si se evaluó más de un método
    if len(method_results) >= 2:
        print("\n--- Generando gráficas comparativas ---")
        charts_dir = os.path.join(target, "charts")
        suffix = f"_outliers_{args.outlier_filter}" if args.outlier_filter != "none" else ""
        if isinstance(shift, str):
            suffix += f"_shifted_{shift}"
        elif shift != (0.0, 0.0):
            suffix += f"_shifted_custom"
        generate_comparison_plots(method_results, charts_dir, suffix=suffix)
        print("Comparativas generadas con éxito.")
