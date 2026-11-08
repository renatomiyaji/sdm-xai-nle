"""Deprecated entry point — use: python -m scripts.run_all study_matrix.csv"""
import subprocess
import sys


def main():
    subprocess.check_call([sys.executable, "-m", "scripts.run_all", sys.argv[1]])


if __name__ == "__main__":
    main()
