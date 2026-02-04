"""Public package API surface."""

from .replay import replay_bk2
from .metadata import collect_bk2_files, create_sidecar_dict

# Lazy import to avoid RuntimeWarning when running as __main__
def generate_aligned_recording(*args, **kwargs):
    from .generate_run_recording import generate_aligned_recording as _func
    return _func(*args, **kwargs)

__all__ = [
    "replay_bk2",
    "collect_bk2_files",
    "create_sidecar_dict",
    "generate_aligned_recording",
]
