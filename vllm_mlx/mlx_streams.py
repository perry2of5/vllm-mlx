# SPDX-License-Identifier: Apache-2.0
"""Helpers for binding MLX generation streams to worker threads."""

import importlib
from collections.abc import Iterable

import mlx.core as mx

# Sentinel used to record that a module failed to import so we do not
# re-attempt (and re-acquire the import lock) on every subsequent call.
_IMPORT_FAILED = object()

# Cache of already-imported modules keyed by module name.  Entries are either
# a real module object or _IMPORT_FAILED.  Concurrent writes from multiple
# worker threads are safe under Python's GIL: the import machinery is itself
# thread-safe, and all writers will set the same value, so no explicit lock is
# needed here.
_module_cache: dict[str, object] = {}


def bind_generation_streams(
    module_names: Iterable[str] = ("mlx_lm.generate", "mlx_vlm.generate"),
) -> object:
    """Bind mlx-lm/mlx-vlm generation streams to the current thread.

    MLX streams are thread-local. If a model is loaded on one thread and
    generation runs on another, module-level generation streams created during
    import can point at a stream that does not exist in the worker thread.

    Modules are imported at most once and then cached so that repeated calls
    do not acquire the Python import lock unnecessarily.
    """
    default_stream = mx.new_stream(mx.default_device())
    mx.set_default_stream(default_stream)
    for module_name in module_names:
        if module_name not in _module_cache:
            try:
                _module_cache[module_name] = importlib.import_module(module_name)
            except ImportError:
                _module_cache[module_name] = _IMPORT_FAILED
                continue
        module = _module_cache[module_name]
        if module is _IMPORT_FAILED:
            continue
        if hasattr(module, "generation_stream"):
            module.generation_stream = default_stream
    return default_stream
