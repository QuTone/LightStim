"""C++ extensions, built by ``pip install``; package metadata is in pyproject.toml.

* ``lightstim.utils.cpp._gf2_rref_cpp`` -- bit-packed GF(2) row echelon.
* ``lightstim.record_tableau.cpp._record_tableau_cpp`` -- the record-tableau
  detector engine, compiled against stim's C++ source (the ``third_party/stim``
  git submodule, pinned to the stim version in pyproject.toml).

Both are optional: without a C++ compiler (or the submodule) the install still
succeeds; the RREF falls back to pure Python and the record_tableau backend
raises an ImportError saying what is missing.

LIGHTSTIM_NATIVE_ARCH sets -march for the record-tableau engine (default
"native": stim then uses the widest SIMD words of the build machine; set a
portable target such as "x86-64-v3", or "" for none, when building wheels).
LIGHTSTIM_BUILD_JOBS sets the number of parallel compile jobs.
"""
import os
from pathlib import Path

from pybind11.setup_helpers import (ParallelCompile, Pybind11Extension, build_ext, has_flag,
                                    naive_recompile)
from setuptools import setup

HERE = Path(__file__).resolve().parent
STIM_SRC = Path("third_party") / "stim" / "src"

# stim translation units behind the header templates the engine uses
# (Tableau, TableauTransposedRaii, TableauSimulator::do_gate, GATE_DATA).
STIM_SOURCES = [
    "stim/mem/simd_util.cc",
    "stim/mem/bit_ref.cc",
    "stim/gates/gates.cc",
    "stim/circuit/circuit.cc",
    "stim/circuit/circuit_instruction.cc",
    "stim/circuit/gate_target.cc",
    "stim/circuit/gate_decomposition.cc",
    "stim/io/measure_record.cc",
    "stim/util_bot/arg_parse.cc",
    "stim/util_bot/probability_util.cc",
]


def _extensions():
    extensions = [
        Pybind11Extension(
            "lightstim.utils.cpp._gf2_rref_cpp",
            ["lightstim/utils/cpp/gf2_rref.cpp"],
            cxx_std=17,
            optional=True,
        ),
    ]
    if (HERE / STIM_SRC / "stim" / "stabilizers" / "tableau.h").exists():
        gate_data = sorted((HERE / STIM_SRC / "stim" / "gates").glob("gate_data_*.cc"))
        extensions.append(Pybind11Extension(
            "lightstim.record_tableau.cpp._record_tableau_cpp",
            ["lightstim/record_tableau/cpp/record_tableau.cc"]
            + [str(STIM_SRC / s) for s in STIM_SOURCES]
            + [str(p.relative_to(HERE)) for p in gate_data],
            include_dirs=[str(STIM_SRC)],
            cxx_std=20,
            optional=True,
        ))
    else:
        print("lightstim: third_party/stim is missing (run `git submodule update --init`); "
              "skipping the record_tableau C++ engine.")
    return extensions


class BuildExt(build_ext):
    """Optimisation flags, plus -march for the record-tableau engine."""

    def build_extensions(self):
        if self.compiler.compiler_type == "unix":
            arch = os.environ.get("LIGHTSTIM_NATIVE_ARCH", "native")
            arch_flag = f"-march={arch}" if arch else None
            if arch_flag and not has_flag(self.compiler, arch_flag):
                arch_flag = None
            for ext in self.extensions:
                ext.extra_compile_args += ["-O3"]
                if ext.name.endswith("_record_tableau_cpp"):
                    ext.extra_compile_args += ["-fno-strict-aliasing"] + ([arch_flag] if arch_flag else [])
        super().build_extensions()


ParallelCompile("LIGHTSTIM_BUILD_JOBS", needs_recompile=naive_recompile).install()

setup(ext_modules=_extensions(), cmdclass={"build_ext": BuildExt})
