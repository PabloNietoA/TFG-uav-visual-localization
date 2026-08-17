import os
from core.processing.tiling import split_orthophoto_and_save

def add_tile_parser(subparsers):
    """Agrega el parser para el comando tile."""
    p = subparsers.add_parser("tile", help="Divide ortofotografías grandes en teselas más pequeñas")
    p.add_argument("-i", "--input", required=True, help="Ruta de la ortofoto (GeoTIFF) de entrada")
    p.add_argument("-o", "--output", default="ortophotos/sections", help="Directorio base de salida (default: ortophotos/sections)")
    p.add_argument("-x", "--tiles-x", type=int, default=10, help="Número de secciones en el eje X (default: 10)")
    p.add_argument("-y", "--tiles-y", type=int, default=10, help="Número de secciones en el eje Y (default: 10)")
    p.set_defaults(func=handle_tile)

def handle_tile(args):
    """Manejador del comando tile."""
    if not os.path.exists(args.input):
        print(f"Error: El archivo de entrada '{args.input}' no existe.")
        return
        
    print(f"Procesando ortofoto: {args.input} -> {args.tiles_x}x{args.tiles_y} teselas...")
    split_orthophoto_and_save(args.input, args.tiles_x, args.tiles_y, args.output)
    print("Teselas generadas con éxito.")
