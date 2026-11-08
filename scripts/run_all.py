"""Run phase A: metrics + figures."""
import subprocess
import sys


def main():
    csv = sys.argv[1]
    py = sys.executable
    for mod in ("scripts.train_classifiers", "scripts.reproduce_figures"):
        subprocess.check_call([py, "-m", mod, csv])


if __name__ == "__main__":
    main()
