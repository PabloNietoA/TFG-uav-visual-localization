import os
from core.simulation.flight import simulate_drone_flight

def add_simulate_parser(subparsers):
    """Agrega el parser para el comando simulate."""
    p = subparsers.add_parser("simulate", help="Simula el vuelo de un dron sobre una ortofoto y genera imágenes")
    p.add_argument("-o", "--ortho", required=True, help="Ruta de la ortofoto original (GeoTIFF)")
    p.add_argument("-j", "--tiles-json", required=True, help="Ruta al JSON de metadatos de las teselas")
    p.add_argument("-n", "--name", default="noname_flight", help="Nombre descriptivo de la trayectoria (default: noname_flight)")
    p.add_argument("-t", "--type", choices=["straight", "curve", "lemniscate"], default="curve", help="Tipo de ruta a generar (default: curve)")
    
    # Coordenadas
    p.add_argument("--start-x", type=float, default=None, help="Punto de inicio en X (coordenadas globales)")
    p.add_argument("--start-y", type=float, default=None, help="Punto de inicio en Y (coordenadas globales)")
    p.add_argument("--end-x", type=float, default=None, help="Punto de fin en X (coordenadas globales)")
    p.add_argument("--end-y", type=float, default=None, help="Punto de fin en Y (coordenadas globales)")
    
    # Parámetros de vuelo
    p.add_argument("-s", "--speed", type=float, default=15.0, help="Velocidad del dron en m/s (default: 15.0)")
    p.add_argument("-f", "--freq", type=float, default=1.0, help="Frecuencia de captura en Hz (default: 1.0)")
    p.add_argument("--out-w", type=int, default=1280, help="Ancho de la imagen generada (default: 1280)")
    p.add_argument("--out-h", type=int, default=720, help="Alto de la imagen generada (default: 720)")
    p.add_argument("-l", "--laps", type=int, default=1, help="Número de vueltas (útil en trayectorias cerradas como lemniscata)")
    p.add_argument("-a", "--angle", type=float, default=0.0, help="Ángulo de rotación inicial de la trayectoria en grados")
    p.add_argument("--out-dir", type=str, default=None, help="Directorio de salida (si es diferente de trajectories/name)")
    
    p.set_defaults(func=handle_simulate)

def handle_simulate(args):
    """Manejador del comando simulate."""
    if not os.path.exists(args.ortho):
        print(f"Error: No se encontró la ortofoto '{args.ortho}'")
        return
    if not os.path.exists(args.tiles_json):
        print(f"Error: No se encontró el JSON de teselas '{args.tiles_json}'")
        return

    start = (args.start_x, args.start_y) if args.start_x is not None and args.start_y is not None else None
    end = (args.end_x, args.end_y) if args.end_x is not None and args.end_y is not None else None

    print(f"Iniciando simulación de vuelo: {args.name} ({args.type})...")
    simulate_drone_flight(
        ortho_path=args.ortho,
        tiles_json_path=args.tiles_json,
        trajectory_name=args.name,
        traj_type=args.type,
        start=start,
        end=end,
        speed_m_s=args.speed,
        freq_hz=args.freq,
        out_w=args.out_w,
        out_h=args.out_h,
        laps=args.laps,
        angle_deg=args.angle,
        output_dir=getattr(args, 'out_dir', None)
    )
    print("Simulación completada.")
