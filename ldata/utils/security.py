import faulthandler
import platform


def reliability_guard(maximum_memory_bytes=None):
    """
    Source: https://github.com/openai/human-eval.
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    ### WARNING
    -----------
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    try:
        builtins.exit = None
        builtins.quit = None
    except TypeError:
        pass

    import os

    try:
        os.environ["OMP_NUM_THREADS"] = "1"

        os.kill = None
        os.system = None
        os.putenv = None
        os.remove = None
        os.removedirs = None
        os.rmdir = None
        os.fchdir = None
        os.setuid = None
        os.fork = None
        os.forkpty = None
        os.killpg = None
        os.rename = None
        os.renames = None
        os.truncate = None
        os.replace = None
        os.unlink = None
        os.fchmod = None
        os.fchown = None
        os.chmod = None
        os.chown = None
        os.chroot = None
        os.fchdir = None
        os.lchflags = None
        os.lchmod = None
        os.lchown = None
        os.getcwd = None
        os.chdir = None
    except TypeError:
        pass

    import shutil

    try:
        shutil.rmtree = None
        shutil.move = None
        shutil.chown = None
    except TypeError:
        pass

    import subprocess

    try:
        subprocess.Popen = None  # type: ignore
        __builtins__["help"] = None
    except TypeError:
        pass

    import sys

    try:
        sys.modules["ipdb"] = None  # type: ignore[reportArgumentType]
        sys.modules["joblib"] = None  # type: ignore[reportArgumentType]
        sys.modules["resource"] = None  # type: ignore[reportArgumentType]
        sys.modules["psutil"] = None  # type: ignore[reportArgumentType]
        sys.modules["tkinter"] = None  # type: ignore[reportArgumentType]
    except TypeError:
        pass
