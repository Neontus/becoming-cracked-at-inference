"""Confirm that the local learning package is installed correctly."""

from inference_lab import __version__


def main() -> None:
    print(f"inference-lab {__version__} is ready")


if __name__ == "__main__":
    main()

