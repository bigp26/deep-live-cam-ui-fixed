from __future__ import annotations

"""
Explicit CPU/GPU image-processing helpers.

Design goals:
- Use OpenCV CUDA only when it is genuinely usable on this machine.
- Avoid misleading "GPU available" state when cv2.cuda exists but kernels/devices do not.
- Preserve drop-in helper API for the rest of the codebase.
"""

from typing import Tuple

import cv2
import numpy as np


CUDA_AVAILABLE: bool = False
CUDA_REASON: str = "uninitialized"


def _detect_cuda_support() -> tuple[bool, str]:
    try:
        if not hasattr(cv2, "cuda") or not hasattr(cv2.cuda, "GpuMat"):
            return False, "cv2.cuda module unavailable"

        device_count = int(cv2.cuda.getCudaEnabledDeviceCount())
        if device_count <= 0:
            return False, "no CUDA-enabled OpenCV device available"

        required = [
            "createGaussianFilter",
            "resize",
            "cvtColor",
            "flip",
            "addWeighted",
        ]
        missing = [name for name in required if not hasattr(cv2.cuda, name)]
        if missing:
            return False, f"missing CUDA functions: {', '.join(missing)}"

        try:
            test = np.zeros((2, 2, 3), dtype=np.uint8)
            gpu = cv2.cuda.GpuMat()
            gpu.upload(test)
            _ = gpu.download()
        except Exception as e:
            return False, f"GpuMat upload/download failed: {e}"

        return True, f"OpenCV CUDA active ({device_count} device(s))"
    except Exception as e:
        return False, f"CUDA detection failed: {e}"


CUDA_AVAILABLE, CUDA_REASON = _detect_cuda_support()
if CUDA_AVAILABLE:
    print(f"[gpu_processing] {CUDA_REASON}")
else:
    print(f"[gpu_processing] CPU mode: {CUDA_REASON}")


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def _ksize_odd(ksize: Tuple[int, int]) -> Tuple[int, int]:
    kw = max(1, ksize[0] // 2 * 2 + 1) if ksize[0] > 0 else 0
    kh = max(1, ksize[1] // 2 * 2 + 1) if ksize[1] > 0 else 0
    return kw, kh


def _cv_type_for(img: np.ndarray) -> int:
    channels = 1 if img.ndim == 2 else img.shape[2]
    if channels == 1:
        return cv2.CV_8UC1
    if channels == 3:
        return cv2.CV_8UC3
    if channels == 4:
        return cv2.CV_8UC4
    return cv2.CV_8UC3


_INTERP_MAP = {
    cv2.INTER_NEAREST: cv2.INTER_NEAREST,
    cv2.INTER_LINEAR: cv2.INTER_LINEAR,
    cv2.INTER_CUBIC: cv2.INTER_CUBIC,
    cv2.INTER_AREA: cv2.INTER_AREA,
    cv2.INTER_LANCZOS4: cv2.INTER_LANCZOS4,
}


def gpu_gaussian_blur(
    src: np.ndarray,
    ksize: Tuple[int, int],
    sigma_x: float,
    sigma_y: float = 0,
) -> np.ndarray:
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            cv_type = _cv_type_for(src_u8)
            ks = _ksize_odd(ksize) if ksize != (0, 0) else ksize

            gauss = cv2.cuda.createGaussianFilter(
                cv_type,
                cv_type,
                ks,
                sigma_x,
                sigma_y,
            )
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_dst = gauss.apply(gpu_src)
            return gpu_dst.download()
        except Exception:
            pass

    return cv2.GaussianBlur(src, ksize, sigma_x, sigmaY=sigma_y)


def gpu_add_weighted(
    src1: np.ndarray,
    alpha: float,
    src2: np.ndarray,
    beta: float,
    gamma: float,
) -> np.ndarray:
    if CUDA_AVAILABLE:
        try:
            s1 = _ensure_uint8(src1)
            s2 = _ensure_uint8(src2)
            g1 = cv2.cuda.GpuMat()
            g2 = cv2.cuda.GpuMat()
            g1.upload(s1)
            g2.upload(s2)
            gpu_dst = cv2.cuda.addWeighted(g1, alpha, g2, beta, gamma)
            return gpu_dst.download()
        except Exception:
            pass

    return cv2.addWeighted(src1, alpha, src2, beta, gamma)


def gpu_sharpen(
    src: np.ndarray,
    strength: float,
    sigma: float = 3,
) -> np.ndarray:
    if strength <= 0:
        return src

    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            cv_type = _cv_type_for(src_u8)
            gauss = cv2.cuda.createGaussianFilter(cv_type, cv_type, (0, 0), sigma)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_blurred = gauss.apply(gpu_src)
            gpu_sharp = cv2.cuda.addWeighted(
                gpu_src,
                1.0 + strength,
                gpu_blurred,
                -strength,
                0,
            )
            result = gpu_sharp.download()
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception:
            pass

    blurred = cv2.GaussianBlur(src, (0, 0), sigma)
    sharpened = cv2.addWeighted(src, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def gpu_resize(
    src: np.ndarray,
    dsize: Tuple[int, int],
    fx: float = 0,
    fy: float = 0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            interp = _INTERP_MAP.get(interpolation, cv2.INTER_LINEAR)

            if dsize and dsize[0] > 0 and dsize[1] > 0:
                gpu_dst = cv2.cuda.resize(gpu_src, dsize, interpolation=interp)
            else:
                gpu_dst = cv2.cuda.resize(
                    gpu_src,
                    (0, 0),
                    fx=fx,
                    fy=fy,
                    interpolation=interp,
                )
            return gpu_dst.download()
        except Exception:
            pass

    return cv2.resize(src, dsize, fx=fx, fy=fy, interpolation=interpolation)


def gpu_cvt_color(
    src: np.ndarray,
    code: int,
) -> np.ndarray:
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_dst = cv2.cuda.cvtColor(gpu_src, code)
            return gpu_dst.download()
        except Exception:
            pass

    return cv2.cvtColor(src, code)


def gpu_flip(
    src: np.ndarray,
    flip_code: int,
) -> np.ndarray:
    if CUDA_AVAILABLE:
        try:
            src_u8 = _ensure_uint8(src)
            gpu_src = cv2.cuda.GpuMat()
            gpu_src.upload(src_u8)
            gpu_dst = cv2.cuda.flip(gpu_src, flip_code)
            return gpu_dst.download()
        except Exception:
            pass

    return cv2.flip(src, flip_code)


def is_gpu_accelerated() -> bool:
    return CUDA_AVAILABLE


def get_gpu_status() -> str:
    return CUDA_REASON
