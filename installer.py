#!/usr/bin/env python3
import os
import sys
import signal
import shutil
import subprocess
from argparse import ArgumentParser, HelpFormatter
from types import FrameType

from facefusion import metadata, wording
from facefusion.common_helper import is_linux

# Only CUDA on Linux is supported:
ONNXRUNTIME_PKG = "onnxruntime-gpu"
ONNXRUNTIME_VERSION = "1.22.0"


def main():
    signal.signal(signal.SIGINT, handle_sigint)

    parser = ArgumentParser(
        description="Install project dependencies (Linux + CUDA only)",
        formatter_class=lambda *args, **kwargs: HelpFormatter(*args, max_help_position=50, **kwargs),
    )
    parser.add_argument(
        "--skip-conda",
        help=wording.get("help.skip_conda"),
        action="store_true",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"{metadata.get('name')} {metadata.get('version')}"
    )

    args = parser.parse_args()

    if not is_linux():
        sys.stderr.write("This installer only supports Linux + CUDA.\n")
        sys.exit(1)

    # If they want Conda-based install, require CONDA_PREFIX
    has_conda = "CONDA_PREFIX" in os.environ
    if not args.skip_conda and not has_conda:
        sys.stderr.write(wording.get("conda_not_activated") + os.linesep)
        sys.exit(1)

    pip = shutil.which("pip")
    if pip is None:
        sys.stderr.write("Error: pip not found in PATH.\n")
        sys.exit(1)

    # 1) Install all other requirements
    with open("requirements.txt", encoding="utf-8") as reqs:
        for line in reqs:
            pkg = line.strip()
            if not pkg or pkg.startswith("#"):
                continue
            if pkg.startswith("onnxruntime"):
                continue
            subprocess.check_call([pip, "install", pkg, "--force-reinstall"])

    # 2) Install onnxruntime-gpu
    gpu_pkg = f"{ONNXRUNTIME_PKG}=={ONNXRUNTIME_VERSION}"
    subprocess.check_call([pip, "install", gpu_pkg, "--force-reinstall"])

    # 3) If using Conda, update LD_LIBRARY_PATH so TensorRT libs are picked up
    if has_conda:
        prefix = os.environ["CONDA_PREFIX"]
        ld_paths = os.getenv("LD_LIBRARY_PATH", "").split(os.pathsep) if os.getenv("LD_LIBRARY_PATH") else []
        python_dir = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidates = [
            os.path.join(prefix, "lib"),
            os.path.join(prefix, "lib", python_dir, "site-packages", "tensorrt_libs"),
        ]
        # keep only existing, unique paths
        ld_paths = list(dict.fromkeys(p for p in ld_paths + candidates if os.path.isdir(p)))
        conda = shutil.which("conda")
        if conda is None:
            sys.stderr.write("Warning: conda not found; skipping LD_LIBRARY_PATH setup.\n")
        else:
            path_str = os.pathsep.join(ld_paths)
            subprocess.check_call([conda, "env", "config", "vars", "set", f"LD_LIBRARY_PATH={path_str}"])

    print("✔️ Installation complete!")


def handle_sigint(signum: int, frame: FrameType) -> None:
    sys.exit(0)


if __name__ == "__main__":
    main()
