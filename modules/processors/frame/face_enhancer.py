from typing import Any, List
import os
import platform
import threading
import sys
import types

import cv2
import torch
import torchvision.transforms.functional as tv_functional

# Compatibility shim for older BasicSR / GFPGAN stacks that still import
# torchvision.transforms.functional_tensor.
if "torchvision.transforms.functional_tensor" not in sys.modules:
    functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    functional_tensor.rgb_to_grayscale = tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor

import gfpgan

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.typing import Frame, Face
from modules.utilities import conditional_download, is_image, is_video

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def _ensure_model_file() -> str:
    model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
    if not os.path.exists(model_path):
        conditional_download(
            models_dir,
            [
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
            ],
        )
    return model_path


def pre_check() -> bool:
    _ensure_model_file()
    return True


def pre_start() -> bool:
    if modules.globals.target_path and not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = _ensure_model_file()
            device = None
            try:
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    print(f"{NAME}: Using CUDA device.")
                elif platform.system() == "Darwin" and torch.backends.mps.is_available():
                    device = torch.device("mps")
                    print(f"{NAME}: Using MPS device.")
                else:
                    device = torch.device("cpu")
                    print(f"{NAME}: Using CPU device.")

                FACE_ENHANCER = gfpgan.GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=device,
                )
                print(f"{NAME}: GFPGANer initialized successfully on {device}.")
            except Exception as e:
                print(f"{NAME}: Error initializing GFPGANer: {e}")
                if device is not None and device.type != "cpu":
                    print(f"{NAME}: Falling back to CPU due to error.")
                    try:
                        FACE_ENHANCER = gfpgan.GFPGANer(
                            model_path=model_path,
                            upscale=1,
                            arch="clean",
                            channel_multiplier=2,
                            bg_upsampler=None,
                            device=torch.device("cpu"),
                        )
                        print(f"{NAME}: GFPGANer initialized successfully on CPU after fallback.")
                    except Exception as fallback_e:
                        print(f"{NAME}: FATAL: Could not initialize GFPGANer even on CPU: {fallback_e}")
                        FACE_ENHANCER = None
                else:
                    print(f"{NAME}: FATAL: Could not initialize GFPGANer on CPU: {e}")
                    FACE_ENHANCER = None

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    enhancer = get_face_enhancer()
    if enhancer is None:
        return temp_frame

    try:
        with THREAD_SEMAPHORE:
            _, _, restored_img = enhancer.enhance(
                temp_frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
        if restored_img is None:
            return temp_frame
        return restored_img
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    return enhance_face(temp_frame)


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            if progress:
                progress.update(1)
            continue

        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue

        result_frame = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
