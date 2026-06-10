"""Deterministic severity-graded ESP32-CAM degradation for TEST-2 / tuning (T6, T9).

``degrade_image`` replays the deployment path on a clean image, in physical
order: motion blur (optics) -> resolution round trip toward 800x600-SVGA
equivalent (sensor readout) -> ISO/Gaussian noise (AGC, ~40 dB SNR at
severity 1) -> Planckian white-balance drift (AWB) -> JPEG re-encode
(on-chip codec, quality 85 -> 50). All randomness derives from
``(seed, severity)``: identical inputs produce byte-identical output.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import numpy.typing as npt

MIN_SEVERITY = 1
MAX_SEVERITY = 5

_SVGA_PIXELS = 800 * 600
# Effective resolution sits below nominal SVGA (Bayer demosaic + fixed-focus lens, F7).
_EFFECTIVE_RESOLUTION_FLOOR = 0.75
_JPEG_QUALITY_MAX = 85
_JPEG_QUALITY_MIN = 50
# 40 dB SNR ~ sigma 2.5/255 at full signal; severity 1 matches the measured camera.
_NOISE_SIGMA_BASE = 1.0
_NOISE_SIGMA_PER_SEVERITY = 1.5
_CHROMA_NOISE_RATIO = 0.5
# AWB green/warm drift: per-channel gain delta, up to 10% at severity 5.
_WB_DELTA_PER_SEVERITY = 0.02
_WB_GREEN_RATIO = 0.25
_BLUR_KSIZE: dict[int, int] = {1: 3, 2: 3, 3: 5, 4: 5, 5: 7}

_U8 = npt.NDArray[np.uint8]


def _validate(img: npt.NDArray[np.uint8], severity: int, seed: int) -> None:
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError(f"img must be a uint8 numpy array, got {type(img).__name__}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be HxWx3, got shape {img.shape}")
    if isinstance(severity, bool) or not isinstance(severity, int):
        raise ValueError(f"severity must be an int, got {type(severity).__name__}")
    if not MIN_SEVERITY <= severity <= MAX_SEVERITY:
        raise ValueError(f"severity must be in [{MIN_SEVERITY}, {MAX_SEVERITY}], got {severity}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")


def _motion_blur(img: _U8, severity: int, rng: np.random.Generator) -> _U8:
    ksize = _BLUR_KSIZE[severity]
    angle = math.radians(float(rng.uniform(0.0, 180.0)))
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = (ksize - 1) / 2.0
    dx, dy = math.cos(angle) * center, math.sin(angle) * center
    p0 = (round(center - dx), round(center - dy))
    p1 = (round(center + dx), round(center + dy))
    cv2.line(kernel, p0, p1, 1.0, thickness=1)
    kernel /= float(kernel.sum())
    return np.asarray(cv2.filter2D(img, -1, kernel), dtype=np.uint8)


def _resolution_roundtrip(img: _U8, severity: int) -> _U8:
    """Downscale toward the 800x600-equivalent pixel count and back up."""
    height, width = img.shape[:2]
    svga_factor = min(1.0, math.sqrt(_SVGA_PIXELS / (height * width)))
    full_loss = svga_factor * _EFFECTIVE_RESOLUTION_FLOOR
    scale = 1.0 - (1.0 - full_loss) * (severity / MAX_SEVERITY)
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = cv2.resize(img, small_size, interpolation=cv2.INTER_AREA)
    return np.asarray(
        cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR), dtype=np.uint8
    )


def _noise_and_white_balance(img: _U8, severity: int, rng: np.random.Generator) -> _U8:
    """Sensor AGC noise plus Planckian AWB drift, in one float pass."""
    height, width = img.shape[:2]
    sigma = _NOISE_SIGMA_BASE + _NOISE_SIGMA_PER_SEVERITY * severity
    out = img.astype(np.float32)
    out += rng.normal(0.0, sigma, size=(height, width, 1)).astype(np.float32)
    out += rng.normal(0.0, sigma * _CHROMA_NOISE_RATIO, size=(height, width, 3)).astype(
        np.float32
    )
    # Warm (+1) vs cool/green (-1) drift; magnitude grows with severity.
    direction = 1.0 if rng.integers(0, 2) == 1 else -1.0
    delta = float(rng.uniform(0.5, 1.0)) * _WB_DELTA_PER_SEVERITY * severity
    gains = np.array(
        [1.0 + direction * delta, 1.0 + _WB_GREEN_RATIO * delta, 1.0 - direction * delta],
        dtype=np.float32,
    )
    out *= gains  # RGB channel gains
    return np.asarray(np.clip(out, 0.0, 255.0), dtype=np.uint8)


def _jpeg_roundtrip(img: _U8, severity: int) -> _U8:
    quality = round(
        _JPEG_QUALITY_MAX
        - (severity - MIN_SEVERITY)
        * (_JPEG_QUALITY_MAX - _JPEG_QUALITY_MIN)
        / (MAX_SEVERITY - MIN_SEVERITY)
    )
    ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for JPEG quality {quality}")
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("cv2.imdecode failed on re-encoded JPEG buffer")
    return np.asarray(decoded, dtype=np.uint8)


def degrade_image(img: npt.NDArray[np.uint8], severity: int, seed: int) -> npt.NDArray[np.uint8]:
    """Apply the deterministic ESP32-CAM degradation pipeline at a severity level.

    Args:
        img: HxWx3 uint8 image (channel order is preserved; gains are applied
            assuming RGB -- pass RGB for physically-signed white-balance drift).
        severity: Degradation level in [1, 5]; 1 approximates the measured
            OV2640 output, 5 is the worst plausible deployment frame.
        seed: Non-negative seed; all randomness derives from ``(seed, severity)``.

    Returns:
        Degraded HxWx3 uint8 image, same shape as the input. Byte-identical
        across calls with identical ``(img, severity, seed)``.
    """
    _validate(img, severity, seed)
    rng = np.random.default_rng(np.random.SeedSequence([seed, severity]))
    out = _motion_blur(img, severity, rng)
    out = _resolution_roundtrip(out, severity)
    out = _noise_and_white_balance(out, severity, rng)
    return _jpeg_roundtrip(out, severity)
