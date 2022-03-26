import argparse
import os
from glob import glob
from torchvision import datasets


def main(output_path: str):
    parser = argparse.ArgumentParser(description="Preprocess MNIST Example")
    parser.add_argument(
        "--output-path",
        type=str,
        default="./data/original",
    )
    args = parser.parse_args()

    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    datasets.MNIST(output_path, download=True)

    for p in glob(os.path.join(output_path, "**/*.gz"), recursive=True):
        os.remove(p)


if __name__ == "__main__":
    main()
