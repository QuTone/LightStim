"""
Build script for the C++ GF(2) RREF extension module.

Usage:
    python src/utils/cpp/build.py

Produces _gf2_rref_cpp.<platform>.so in src/utils/cpp/ which is importable
from Python as: from src.utils.cpp._gf2_rref_cpp import row_echelon
"""
import os
import subprocess
import sys
import sysconfig

def build():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    cpp_file = os.path.join(src_dir, "gf2_rref.cpp")

    # Get pybind11 and Python include paths
    import pybind11
    pybind11_include = pybind11.get_include()
    python_include = sysconfig.get_path("include")

    # Output shared library name
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    output = os.path.join(src_dir, f"_gf2_rref_cpp{ext_suffix}")

    cmd = [
        "g++",
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        f"-I{pybind11_include}",
        f"-I{python_include}",
        cpp_file,
        "-o", output,
    ]

    print(f"Building: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Build FAILED:\n{result.stderr}")
        sys.exit(1)

    print(f"Built: {output}")
    return output


if __name__ == "__main__":
    build()
