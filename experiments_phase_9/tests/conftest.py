from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch


def pytest_sessionstart(session):
    torch.set_num_threads(2)
