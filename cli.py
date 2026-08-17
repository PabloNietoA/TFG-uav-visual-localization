import argparse
from cli.commands.tile import add_tile_parser
from cli.commands.simulate import add_simulate_parser
from cli.commands.match import add_match_parser
from cli.commands.evaluate import add_evaluate_parser
from cli.commands.pipeline import add_pipeline_parser

def main():
    parser = argparse.ArgumentParser(
        description="Herramienta unificada para Odometría Visual y Matching",
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, title="Comandos disponibles")

    # Registrar subcomandos
    add_tile_parser(subparsers)
    add_simulate_parser(subparsers)
    add_match_parser(subparsers)
    add_evaluate_parser(subparsers)
    add_pipeline_parser(subparsers)

    args = parser.parse_args()

    # Ejecutar el handler correspondiente
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
