#!/usr/bin/env python
import click
import json
import os
import tarfile
import torch
from pathlib import Path
from torchvision import datasets, transforms

from model import Net


@click.command()
@click.option(
    "--model-path", type=Path, default=Path("/opt/ml/processing/model/mnist_cnn.pt")
)
@click.option("--input-path", type=Path, default=Path("/opt/ml/processing/input/"))
@click.option(
    "--output-path",
    type=Path,
    default=Path("/opt/ml/processing/output/evaluation.json"),
)
@click.option("--batch-size", type=int, default=1000)
def main(model_path: Path, input_path: Path, output_path: Path, batch_size: int):
    if model_path.suffixes == [".tar", ".gz"]:
        with tarfile.open(model_path) as t:
            t.extractall(model_path.parent)
        model_path = Path(model_path.parent / "mnist_cnn.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_kwargs = {"batch_size": batch_size}
    if torch.cuda.is_available():
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset = datasets.MNIST(str(input_path), train=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    evaluation = {
        "accuracy": correct / len(data_loader.dataset),
    }

    os.makedirs(output_path.parent, exist_ok=True)
    json.dump(evaluation, open(output_path, "w"))


if __name__ == "__main__":
    main()
