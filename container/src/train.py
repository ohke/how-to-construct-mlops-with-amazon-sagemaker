#!/usr/bin/env python
# https://github.com/pytorch/examples/blob/main/mnist/main.py

from __future__ import annotations, print_function
import click
import json
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from glob import glob
from torchvision import datasets, transforms
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR

from model import Net


@dataclass
class Hyperparameter:
    epochs: int = 2
    batch_size: int = 64
    lr: float = 1.0

    @classmethod
    def from_str_dict(cls, d: dict[str, str]) -> Hyperparameter:
        return Hyperparameter(
            epochs=int(d["epochs"]),
            batch_size=int(d["batch_size"]),
            lr=float(d["lr"]),
        )


@dataclass
class Checkpoint:
    epoch: int
    model: Net
    optimizer: Optimizer
    scheduler: StepLR

    def save(self, checkpoint_path: str) -> str:
        os.makedirs(checkpoint_path, exist_ok=True)

        path = os.path.join(checkpoint_path, "checkpoint-{:04d}.tar".format(self.epoch))
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

        return path

    @classmethod
    def load(
        cls, checkpoint_path: str, model: Net, optimizer: Optimizer, schduler: StepLR
    ) -> Checkpoint:
        tars = glob(os.path.join(checkpoint_path, "checkpoint-*.tar"))
        if len(tars) == 0:
            return Checkpoint(0, model, optimizer, schduler)

        c = torch.load(tars[-1])
        model.load_state_dict(c["model_state_dict"])
        optimizer.load_state_dict(c["optimizer_state_dict"])
        schduler.load_state_dict(c["scheduler_state_dict"])

        return Checkpoint(c["epoch"], model, optimizer, schduler)


def train(model, device, train_loader, optimizer, epoch, log_interval, dry_run):
    model.train()

    train_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if dry_run:
                break

    print("\ntrain_loss: {:.4f}\n".format(train_loss))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\ntest_loss: {:.4f} test_accuracy: {:.4f} ({}/{})\n".format(
            test_loss,
            correct / len(test_loader.dataset),
            correct,
            len(test_loader.dataset),
        )
    )


@click.command()
@click.option(
    "--test-batch-size",
    type=int,
    default=1000,
    help="input batch size for testing (default: 1000)",
)
@click.option(
    "--gamma", type=float, default=7, help="Learning rate step gamma (default: 0.7)"
)
@click.option(
    "--no-cuda",
    is_flag=True,
    show_default=True,
    default=False,
    help="disables CUDA training",
)
@click.option(
    "--dry-run",
    is_flag=True,
    show_default=True,
    default=False,
    help="quickly check a single pass",
)
@click.option("--seed", type=int, default=1, help="random seed (default: 1)")
@click.option(
    "--log-interval",
    type=int,
    default=10,
    help="how many batches to wait before logging training status",
)
@click.option(
    "--input-path",
    type=str,
    default="/opt/ml/input/data/train",
    help="path to load data",
)
@click.option(
    "--output-path",
    type=str,
    default="/opt/ml/model",
    help="path to save trained model",
)
@click.option(
    "--parameter-path",
    type=str,
    default="/opt/ml/input/config/hyperparameters.json",
    help="path to load hyperparameters.json",
)
@click.option(
    "--checkpoint-path",
    type=str,
    default="/opt/ml/checkpoints",
    help="path to save checkpoint file",
)
def main(
    test_batch_size: int,
    gamma: float,
    no_cuda: bool,
    dry_run: bool,
    seed: int,
    log_interval: int,
    input_path: str,
    output_path: str,
    parameter_path: str,
    checkpoint_path: str,
):
    hyperparameter = Hyperparameter.from_str_dict(json.load(open(parameter_path)))

    use_cuda = not no_cuda and torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {"batch_size": hyperparameter.batch_size}
    test_kwargs = {"batch_size": test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    dataset1 = datasets.MNIST(input_path, train=True, transform=transform)
    dataset2 = datasets.MNIST(input_path, train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=hyperparameter.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    checkpoint = Checkpoint.load(checkpoint_path, model, optimizer, scheduler)
    for epoch in range(checkpoint.epoch + 1, hyperparameter.epochs + 1):
        train(
            checkpoint.model,
            device,
            train_loader,
            checkpoint.optimizer,
            epoch,
            log_interval,
            dry_run,
        )
        test(checkpoint.model, device, test_loader)
        checkpoint.scheduler.step()

        checkpoint.epoch = epoch
        saved_path = checkpoint.save(checkpoint_path)
        print(f"Checkpoint saved: {saved_path}")

    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, "mnist_cnn.pt"))


if __name__ == "__main__":
    main()
