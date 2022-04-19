#!/usr/bin/env python
import click
from pathlib import Path
from torchvision import datasets


@click.command()
@click.option("--output-path", type=Path, default=Path("/opt/ml/processing/output"))
def main(output_path: Path):
    """Download MNIST dataset."""
    output_path.mkdir(parents=True, exist_ok=True)
    datasets.MNIST(output_path, download=True)
    for p in output_path.rglob("**/*.gz"):
        p.unlink()


if __name__ == "__main__":
    main()
