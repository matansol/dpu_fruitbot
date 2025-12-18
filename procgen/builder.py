import threading
import os
import contextlib
import subprocess as sp
import shutil
import json
import sys
import platform
import multiprocessing as mp

import gym3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


global_build_lock = threading.Lock()
global_builds = set()


class RunFailure(Exception):
    pass


@contextlib.contextmanager
def nullcontext():
    # this is here for python 3.6 support
    yield


@contextlib.contextmanager
def chdir(newdir):
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(curdir)


def run(cmd):
    return sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf8")


def check(proc, verbose):
    if proc.returncode != 0:
        print(f"RUN FAILED {proc.args}:\n{proc.stdout}")
        raise RunFailure("failed to build procgen from source")
    if verbose:
        print(f"RUN {proc.args}:\n{proc.stdout}")


def _windows_detect_generator():
    """
    Best-effort detection of an appropriate Visual Studio CMake generator on Windows.
    Precedence:
    1) Respect PROCGEN_CMAKE_GENERATOR/PROCGEN_CMAKE_ARCH if provided
    2) Use vswhere (if available) to pick VS 2022 or VS 2019
    3) Parse `cmake --help` to see supported VS generators
    4) Fallback to VS 2019
    Returns (generator, arch)
    """
    # 1) Environment overrides
    gen_env = os.environ.get("PROCGEN_CMAKE_GENERATOR")
    arch_env = os.environ.get("PROCGEN_CMAKE_ARCH", "x64")
    if gen_env:
        return gen_env, arch_env

    # 2) Try vswhere to detect installed VS
    try:
        vswhere_path = os.path.join(
            os.environ.get("ProgramFiles(x86)", r"C:\\Program Files (x86)"),
            "Microsoft Visual Studio",
            "Installer",
            "vswhere.exe",
        )
        if os.path.exists(vswhere_path):
            # Query for latest VS with C++ tools; get installationVersion
            proc = sp.run(
                [
                    vswhere_path,
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationVersion",
                    "-format",
                    "value",
                ],
                stdout=sp.PIPE,
                stderr=sp.STDOUT,
                encoding="utf8",
            )
            if proc.returncode == 0:
                ver = (proc.stdout or "").strip()
                # installationVersion like 17.10.2... => VS 2022
                if ver.startswith("17."):
                    return "Visual Studio 17 2022", "x64"
                if ver.startswith("16."):
                    return "Visual Studio 16 2019", "x64"
    except Exception:
        pass

    # 3) Parse cmake --help for supported generators
    try:
        proc = sp.run(["cmake", "--help"], stdout=sp.PIPE, stderr=sp.STDOUT, encoding="utf8")
        if proc.returncode == 0:
            help_text = proc.stdout or ""
            if "Visual Studio 17 2022" in help_text:
                return "Visual Studio 17 2022", "x64"
            if "Visual Studio 16 2019" in help_text:
                return "Visual Studio 16 2019", "x64"
    except Exception:
        pass

    # 4) Fallback
    return "Visual Studio 16 2019", "x64"


def _attempt_configure(build_type, package):
    if "PROCGEN_CMAKE_PREFIX_PATH" in os.environ:
        cmake_prefix_paths = [os.environ["PROCGEN_CMAKE_PREFIX_PATH"]]
    else:
        # guess some common qt cmake paths, it's unclear why cmake can't find qt without this
        cmake_prefix_paths = ["/usr/local/opt/qt5/lib/cmake"]
        conda_exe = shutil.which("conda")
        if conda_exe is not None:
            conda_info = json.loads(
                sp.run(["conda", "info", "--json"], stdout=sp.PIPE).stdout
            )
            conda_prefix = conda_info["active_prefix"]
            if conda_prefix is None:
                conda_prefix = conda_info["conda_prefix"]
            if platform.system() == "Windows":
                conda_prefix = os.path.join(conda_prefix, "library")
            conda_cmake_path = os.path.join(conda_prefix, "lib", "cmake", "Qt5")
            # prepend this qt since it's likely to be loaded already by the python process
            cmake_prefix_paths.insert(0, conda_cmake_path)

    generator = "Unix Makefiles"
    extra_configure_options = []
    if platform.system() == "Windows":
        # Auto-detect a suitable Visual Studio generator (or respect env overrides)
        generator, arch = _windows_detect_generator()
        extra_configure_options.extend(["-A", arch])
    configure_cmd = [
        "cmake",
        "-G",
        generator,
        *extra_configure_options,
        "-DCMAKE_PREFIX_PATH=" + ";".join(cmake_prefix_paths),
        f"-DLIBENV_DIR={gym3.libenv.get_header_dir()}",
        "../..",
    ]
    if package:
        configure_cmd.append("-DPROCGEN_PACKAGE=ON")
    if platform.system() != "Windows":
        # this is not used on windows, the option needs to be passed to cmake --build instead
        configure_cmd.append(f"-DCMAKE_BUILD_TYPE={build_type}")

    check(run(configure_cmd), verbose=package)


def build(package=False, debug=False):
    """
    Build procgen from source.
    Set PROCGEN_NO_BUILD=1 environment variable to skip auto-build (only if DLL exists).
    """
    build_dir = os.path.join(SCRIPT_DIR, ".build")
    build_type = "relwithdebinfo"
    if debug:
        build_type = "debug"

    # Ensure the build directory exists before trying to chdir into it
    os.makedirs(build_dir, exist_ok=True)

    with chdir(build_dir), global_build_lock:
        # check if we have built yet in this process
        if build_type not in global_builds:
            if package:
                # avoid the filelock dependency when building from setup.py
                lock_ctx = nullcontext()
            else:
                # prevent multiple processes from trying to build at the same time
                import filelock

                lock_ctx = filelock.FileLock(".build-lock")
            with lock_ctx:
                sys.stdout.write("building procgen...")
                sys.stdout.flush()
                try:
                    os.makedirs(build_type, exist_ok=True)
                    with chdir(build_type):
                        _attempt_configure(build_type, package)
                except RunFailure:
                    # cmake can get into a weird state, so nuke the build directory and retry once
                    sys.stdout.write("retrying configure due to failure...")
                    sys.stdout.flush()
                    shutil.rmtree(build_type)
                    os.makedirs(build_type, exist_ok=True)
                    with chdir(build_type):
                        _attempt_configure(build_type, package)

                if "MAKEFLAGS" not in os.environ:
                    os.environ["MAKEFLAGS"] = f"-j{mp.cpu_count()}"

                with chdir(build_type):
                    build_cmd = ["cmake", "--build", ".", "--config", build_type]
                    check(run(build_cmd), verbose=package)
                print("done")

            global_builds.add(build_type)

    lib_dir = os.path.join(build_dir, build_type)
    if platform.system() == "Windows":
        # the built library is in a different location on windows
        lib_dir = os.path.join(lib_dir, build_type)
    return lib_dir
