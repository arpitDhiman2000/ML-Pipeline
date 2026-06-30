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
from threat_detection.features.build import run_preprocessing
from threat_detection.logging_utils import configure_logging
from threat_detection.tracking import configure_mlflow, get_tracking_uri, is_remote, load_env

app = typer.Typer(
    add_completion=False,
    help="Threat Detection ML Pipeline CLI.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    """Configure logging and load .env for every invocation."""
    configure_logging(level="DEBUG" if verbose else "INFO")
    load_env()


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


@app.command()
def preprocess() -> None:
    """Clean, split, and fit the preprocessing artifacts (tabular + text)."""
    result = run_preprocessing()
    table = Table(title="Preprocessing complete")
    table.add_column("Modality")
    table.add_column("Split")
    table.add_column("Rows", justify="right")
    for modality, splits in result.items():
        for split, rows in splits.items():
            table.add_row(modality, split, str(rows))
    console.print(table)


@app.command("train-tabular")
def train_tabular_cmd(
    no_register: bool = typer.Option(False, help="Skip MLflow model-registry registration"),
) -> None:
    """Train Isolation Forest + XGBoost, evaluate, and log to MLflow."""
    from threat_detection.training.train_tabular import train

    metrics = train(register=not no_register)
    table = Table(title="Tabular model — test metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.to_flat_dict().items():
        table.add_row(key, f"{value:.4f}")
    console.print(table)


@app.command("train-text")
def train_text_cmd(
    no_register: bool = typer.Option(False, help="Skip MLflow model-registry registration"),
) -> None:
    """Train the LSTM text classifier and bake it off vs the TF-IDF baseline."""
    from threat_detection.training.train_text import train

    metrics = train(register=not no_register)
    table = Table(title="Text model — test metrics (LSTM vs baseline)")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}")
    console.print(table)


@app.command()
def promote(
    model: str = typer.Argument(
        None, help="Registered model name; omit to promote all project models"
    ),
    version: str = typer.Option(None, help="Specific version; default = latest"),
    alias: str = typer.Option("production", help="Alias to point at the version"),
) -> None:
    """Point an alias (default 'production') at a model version in the registry."""
    from threat_detection.registry import ALL_MODELS
    from threat_detection.registry import promote as do_promote

    targets = [model] if model else list(ALL_MODELS)
    table = Table(title="Registry promotion")
    table.add_column("Model")
    table.add_column("Alias")
    table.add_column("Version", justify="right")
    for name in targets:
        try:
            promoted = do_promote(name, version=version, alias=alias)
            table.add_row(name, alias, promoted)
        except Exception as exc:  # surface backend/registry errors clearly
            table.add_row(name, alias, f"[red]FAILED: {exc}[/red]")
    console.print(table)


@app.command("show-config")
def show_config() -> None:
    """Print the validated configuration loaded from params.yaml."""
    cfg = get_config()
    console.print_json(cfg.model_dump_json(indent=2))


@app.command("mlflow-check")
def mlflow_check() -> None:
    """Diagnose MLflow/DagsHub auth: show what creds will be sent (masked)."""
    import os
    from pathlib import Path

    load_env()

    def mask(val: str | None) -> str:
        if not val:
            return "[red]<not set>[/red]"
        if len(val) <= 6:
            return f"[yellow]set (len={len(val)})[/yellow]"
        return f"[green]{val[:3]}…{val[-2:]} (len={len(val)})[/green]"

    env_path = Path(".env")
    console.print(f".env file present: {env_path.exists()}  ({env_path.resolve()})")
    console.print(f"MLFLOW_TRACKING_URI      = {os.environ.get('MLFLOW_TRACKING_URI')!r}")
    console.print(f"MLFLOW_TRACKING_USERNAME = {mask(os.environ.get('MLFLOW_TRACKING_USERNAME'))}")
    console.print(f"MLFLOW_TRACKING_PASSWORD = {mask(os.environ.get('MLFLOW_TRACKING_PASSWORD'))}")
    console.print(f"MLFLOW_TRACKING_TOKEN    = {mask(os.environ.get('MLFLOW_TRACKING_TOKEN'))}")

    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    has_basic = bool(
        os.environ.get("MLFLOW_TRACKING_USERNAME") and os.environ.get("MLFLOW_TRACKING_PASSWORD")
    )
    has_token = bool(os.environ.get("MLFLOW_TRACKING_TOKEN"))
    if is_remote(uri) and not (has_basic or has_token):
        console.print(
            "[red]No credentials set[/red] — set USERNAME+PASSWORD (token as password) "
            "or MLFLOW_TRACKING_TOKEN in .env."
        )
    elif "<" in (os.environ.get("MLFLOW_TRACKING_PASSWORD") or ""):
        console.print(
            "[red]Password still contains a placeholder ('<...>')[/red] — paste your real token."
        )
    else:
        console.print(
            "[green]Credentials present.[/green] If you still get 401, the token is wrong/expired."
        )


@app.command("mlflow-smoke")
def mlflow_smoke() -> None:
    """Log a tiny MLflow run to verify tracking (e.g. the DagsHub server)."""
    import mlflow

    uri = get_tracking_uri()
    console.print(f"[bold]MLflow tracking URI:[/bold] {uri}  (remote={is_remote(uri)})")
    configure_mlflow(experiment="smoke-test")
    with mlflow.start_run(run_name="connectivity-check") as run:
        mlflow.log_param("hello", "dagshub")
        mlflow.log_metric("answer", 42)
        console.print(f"[green]Logged run[/green] id={run.info.run_id}")
    if is_remote(uri):
        console.print("Open your DagsHub repo -> [bold]Experiments[/bold] tab to see it.")


if __name__ == "__main__":
    app()
