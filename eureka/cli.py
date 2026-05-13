"""Eureka-EML command-line interface."""

import click
import numpy as np
import pandas as pd


@click.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option("--target", "-t", required=True, help="Name of the target column")
@click.option("--population", "-p", default=30, help="Population size")
@click.option("--generations", "-g", default=50, help="Number of generations")
@click.option("--max-depth", "-d", default=6, help="Maximum tree depth")
@click.option("--timeout", default=120.0, help="Timeout in seconds")
@click.option("--latex", is_flag=True, help="Output LaTeX format")
@click.option("--quiet", "-q", is_flag=True, help="Suppress progress output")
def main(csv_path, target, population, generations, max_depth, timeout, latex, quiet):
    """Discover equations from CSV data.

    Example: eureka data.csv --target price
    """
    df = pd.read_csv(csv_path)

    if target not in df.columns:
        raise click.BadParameter(
            f"Column '{target}' not found. Available: {list(df.columns)}"
        )

    y = df[target].values.astype(np.float32)
    feature_cols = df.drop(columns=[target]).select_dtypes(include=[np.number])
    X = feature_cols.values.astype(np.float32)
    feature_names = list(feature_cols.columns)

    if X.shape[1] == 0:
        raise click.BadParameter("No numeric feature columns found")

    click.echo(f"Features: {feature_names}")
    click.echo(f"Samples: {X.shape[0]}, Variables: {X.shape[1]}")
    click.echo()

    from eureka.regression import discover
    from eureka.simplify import simplify_tree, to_latex as to_latex_fn

    result = discover(
        X, y,
        population=population,
        generations=generations,
        max_depth=max_depth,
        timeout=timeout,
        verbose=not quiet,
    )

    equation = to_latex_fn(result.tree) if latex else simplify_tree(result.tree)

    click.echo()
    click.echo("=" * 50)
    click.echo("Discovered equation:")
    click.echo(f"  {target} = {equation}")
    click.echo(f"MSE: {result.mse:.6e}")
    click.echo(f"Complexity: {result.complexity} nodes")
    click.echo("=" * 50)


if __name__ == "__main__":
    main()
