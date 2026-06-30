"""Unified command-line interface: ``td <command>``.

One CLI surface for the whole project keeps operational commands discoverable
and testable. Sprints add subcommands here (train, serve, evaluate, drift...).
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from threat_detection.config import get_config
from threat_detection.data.generate import generate_raw_data
from threat_detection.logging_utils import configure_logging

app = typer.Typer(
    add_completion=False,
    help="Threat Detection ML Pipeline CLI.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Configure logging for every invocation."""
    configure_logging(level="DEBUG" if verbose else "INFO")


@app.command()
def generate(force: bool = typer.Option(False, help="Regenerate even if files exist")) -> None:
    """Generate the raw data zone (synthetic CICIDS2017 + text corpus)."""
    paths = generate_raw_data(force=force)
    table = Table(title="Raw data generated")
    table.add_column("Dataset")
    table.add_column("Path")
    for name, path in paths.items():
        table.add_row(name, str(path))
    console.print(table)


@app.command("show-config")
def show_config() -> None:
    """Print the validated configuration loaded from params.yaml."""
    cfg = get_config()
    console.print_json(cfg.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
