import importlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import ModuleType
from typing import Any, Callable, List

from tqdm import tqdm

import modules.globals


FRAME_PROCESSORS_INTERFACE = [
    "pre_check",
    "pre_start",
    "process_frame",
    "process_image",
    "process_video",
]

# Cache modules by processor name, not by one sticky global list order.
_FRAME_PROCESSOR_CACHE: dict[str, ModuleType] = {}


def load_frame_processor_module(frame_processor: str) -> ModuleType:
    try:
        frame_processor_module = importlib.import_module(
            f"modules.processors.frame.{frame_processor}"
        )
    except ImportError:
        print(f"Frame processor {frame_processor} not found")
        sys.exit(1)

    for method_name in FRAME_PROCESSORS_INTERFACE:
        if not hasattr(frame_processor_module, method_name):
            print(
                f"Frame processor {frame_processor} is missing required method: "
                f"{method_name}"
            )
            sys.exit(1)

    return frame_processor_module


def _get_or_load_processor(frame_processor: str) -> ModuleType:
    module = _FRAME_PROCESSOR_CACHE.get(frame_processor)
    if module is None:
        module = load_frame_processor_module(frame_processor)
        _FRAME_PROCESSOR_CACHE[frame_processor] = module
    return module


def _dedupe_preserve_order(names: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _resolve_requested_processors(frame_processors: List[str]) -> List[str]:
    requested = list(frame_processors or [])

    # UI can enable extra processors, but should not silently remove explicitly
    # requested processors here.
    for frame_processor, state in modules.globals.fp_ui.items():
        if state is True:
            requested.append(frame_processor)

    requested = _dedupe_preserve_order(requested)

    # Keep globals in sync with the resolved runtime list.
    modules.globals.frame_processors = list(requested)
    return requested


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    resolved_processors = _resolve_requested_processors(frame_processors)
    return [_get_or_load_processor(name) for name in resolved_processors]


def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    """
    Backwards-compatible shim.

    Older call sites may still call this function. We no longer mutate a sticky
    module list here; instead we normalize globals.frame_processors from the
    current request + UI state.
    """
    _resolve_requested_processors(frame_processors)


def multi_process_frame(
    source_path: str,
    temp_frame_paths: List[str],
    process_frames: Callable[[str, List[str], Any], None],
    progress: Any = None,
) -> None:
    """Process frames in parallel with bounded fan-out and deterministic waits."""
    max_workers = modules.globals.execution_threads or 1
    max_workers = max(1, int(max_workers))

    if not temp_frame_paths:
        return

    # Submit one frame-path per task. The per-processor implementations already
    # accept a list[str], so keep that contract but pass one item for isolation.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_frames, source_path, [path], progress)
            for path in temp_frame_paths
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame: {e}")


def process_video(
    source_path: str,
    frame_paths: list[str],
    process_frames: Callable[[str, List[str], Any], None],
) -> None:
    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    total = len(frame_paths)

    with tqdm(
        total=total,
        desc="Processing",
        unit="frame",
        dynamic_ncols=True,
        bar_format=progress_bar_format,
    ) as progress:
        progress.set_postfix(
            {
                "execution_providers": modules.globals.execution_providers,
                "execution_threads": modules.globals.execution_threads,
                "max_memory": modules.globals.max_memory,
            }
        )
        multi_process_frame(source_path, frame_paths, process_frames, progress)
