"""Optional native algebra, adapted from Hou Yuewen's PR #110 build setup.

The inverse-tableau core uses the pinned Stim source submodule. Public Python
Stim remains a normal dependency; no old detector/observable policy is loaded.
Set LIGHTSTIM_BUILD_JOBS for parallel compilation. The default instruction set
is portable; LIGHTSTIM_NATIVE_ARCH=native is an explicit local-build opt-in.
"""
import os
from pathlib import Path

from pybind11.setup_helpers import (
    ParallelCompile, Pybind11Extension, build_ext, has_flag, naive_recompile,
)
from setuptools import setup

ROOT = Path(__file__).resolve().parent
STIM_SRC = Path("third_party/stim/src")
STIM_SOURCES = [
    "stim/mem/simd_util.cc", "stim/mem/bit_ref.cc", "stim/gates/gates.cc",
    "stim/circuit/circuit.cc", "stim/circuit/circuit_instruction.cc",
    "stim/circuit/gate_target.cc", "stim/circuit/gate_decomposition.cc",
    "stim/io/measure_record.cc", "stim/util_bot/arg_parse.cc",
    "stim/util_bot/probability_util.cc",
]


def extensions():
    result = [Pybind11Extension(
        "lightstim.utils.cpp._gf2_rref_cpp",
        ["lightstim/utils/cpp/gf2_rref.cpp"], cxx_std=17, optional=True,
    )]
    if (ROOT / STIM_SRC / "stim/stabilizers/tableau.h").exists():
        gates = sorted((ROOT / STIM_SRC / "stim/gates").glob("gate_data_*.cc"))
        result.append(Pybind11Extension(
            "lightstim.utils.cpp._pauli_backend_cpp",
            ["lightstim/utils/cpp/pauli_backend.cc"]
            + [str(STIM_SRC / path) for path in STIM_SOURCES]
            + [str(path.relative_to(ROOT)) for path in gates],
            include_dirs=[str(STIM_SRC)],
            depends=["lightstim/utils/cpp/pauli_propagation.h",
                     "lightstim/utils/cpp/persistent_record_state.h"],
            cxx_std=20, optional=True,
        ))
    else:
        print("lightstim: optional inverse-tableau extension skipped; "
              "run `git submodule update --init third_party/stim` to build it.")
    return result


class BuildExt(build_ext):
    def build_extensions(self):
        if self.compiler.compiler_type == "unix":
            arch = os.environ.get("LIGHTSTIM_NATIVE_ARCH", "")
            arch_flag = f"-march={arch}" if arch else None
            if arch_flag and not has_flag(self.compiler, arch_flag):
                raise RuntimeError(f"Compiler does not support {arch_flag}")
            for ext in self.extensions:
                ext.extra_compile_args += ["-O3"]
                if ext.name.endswith("_pauli_backend_cpp"):
                    ext.extra_compile_args += ["-fno-strict-aliasing"]
                    if arch_flag:
                        ext.extra_compile_args.append(arch_flag)
        super().build_extensions()


# pybind11's naive_recompile checks only .cc mtimes. The state engine is in a
# header, so an editable rebuild must also invalidate objects after header or
# build-configuration changes (including changes in the pinned Stim checkout).
_header_mtime = max(
    [Path(__file__).stat().st_mtime]
    + [p.stat().st_mtime
       for base in (ROOT / "lightstim/utils/cpp", ROOT / STIM_SRC)
       for pattern in ("*.h", "*.inl") for p in base.rglob(pattern)]
)


def needs_recompile(obj, src):
    return naive_recompile(obj, src) or Path(obj).stat().st_mtime < _header_mtime


ParallelCompile("LIGHTSTIM_BUILD_JOBS", default=2, needs_recompile=needs_recompile).install()
setup(ext_modules=extensions(), cmdclass={"build_ext": BuildExt})
