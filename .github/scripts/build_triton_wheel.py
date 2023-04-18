#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Optional

def build_triton() -> Path:
    triton_pythondir = Path("/triton") / "python"
    check_call([sys.executable, "setup.py", "bdist_wheel"], cwd=triton_pythondir)
    whl_path = list((triton_pythondir / "dist").glob("*.whl"))[0]
    shutil.copy(whl_path, Path.cwd())

def main() -> None:
    build_triton()
    
if __name__ == "__main__":
    main()

