"""
image_validator.py
-------------------
Lightweight, rule-based gatekeeper that runs BEFORE any scan model
(CT / MRI / Ultrasound) is invoked. It rejects clearly non-medical
images (selfies, landscapes, documents, logos, animals, screenshots)
so the app never returns a confident "No Clot Detected" on a photo
of a cat.

Design goals:
- No extra training data or model download required (works offline,
  instantly, on CPU).
- Conservative: only reject images that are CONFIDENTLY non-medical.
  When unsure, let the real model run and rely on its own confidence
  thresholding instead of blocking a legitimate borderline scan.
- Cheap: pure NumPy / PIL, a few milliseconds per image.

Medical grayscale-style scans (CT, MRI, X-ray, most ultrasound
exports) share three statistical fingerprints that ordinary photos
almost never share all at once:
  1. Near-grayscale color channels, even when stored as RGB/JPEG.
  2. A dark or near-black background with no busy multi-color scene.
  3. Skin-tone / vivid-hue pixel ratio close to zero.

We combine several independent signals into one confidence score
and only block the image when multiple signals agree.
"""

from __future__ import annotations
import numpy as np
from PIL import Image


# ── Tunable thresholds ─────────────────────────────────────────────
SATURATION_MAX_MEDICAL   = 28     # mean saturation (0-255) - real photos run much higher
CHANNEL_DIFF_MAX_MEDICAL = 14     # mean |R-G| + |G-B| + |R-B| per pixel
SKIN_TONE_MAX_RATIO      = 0.06   # fraction of pixels in skin-tone hue band
COLORFUL_PIXEL_MAX_RATIO = 0.12   # fraction of highly-saturated pixels
MIN_DARK_BACKGROUND_FRAC = 0.10   # scans usually have a meaningful dark/black region

MIN_EDGE_DENSITY         = 0.015  # scans have real anatomical texture, not flat fields
MIN_INTENSITY_STD        = 12.0   # near-uniform images (blank/solid) lack tissue variance
MAX_WHITE_BG_FRAC        = 0.55   # documents/logos are dominated by white background
MIN_UNIQUE_GRAY_LEVELS   = 20     # flat logos/docs use very few distinct gray tones

REJECT_VOTE_THRESHOLD = 3   # need at least this many red flags to reject


def _to_rgb_array(image_input) -> np.ndarray:
    """Accepts PIL Image, file path, or numpy array -> returns HxWx3 uint8 RGB array."""
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img = Image.fromarray(image_input).convert('RGB')
        else:
            img = Image.fromarray(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')

    # Downscale for speed - statistics are stable at small size
    img = img.resize((160, 160))
    return np.asarray(img, dtype=np.float32)


def _saturation_stats(rgb: np.ndarray) -> dict:
    """HSV-based saturation and hue stats without needing OpenCV."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    # Saturation (0-255 scale). Pixels that are near-black carry no reliable
    # hue/saturation signal and must be excluded, otherwise tiny channel
    # noise on a near-zero maxc explodes the ratio (a real risk for medical
    # scans, which are mostly black background).
    valid_px = maxc > 10  # ignore near-black pixels entirely
    sat = np.zeros_like(maxc)
    sat[valid_px] = (delta[valid_px] / maxc[valid_px]) * 255.0
    mean_sat = float(sat[valid_px].mean()) if valid_px.any() else 0.0

    # Fraction of pixels that are strongly colorful (real-world photo signal),
    # measured only among non-black pixels so a scan's black border can't
    # dilute or a stray bright artifact can't dominate the denominator.
    colorful_frac = float((sat[valid_px] > 60).mean()) if valid_px.any() else 0.0

    # Channel divergence - grayscale-like scans have R≈G≈B everywhere.
    # Also restricted to non-black pixels for the same reason as above.
    if valid_px.any():
        channel_diff = (np.abs(r - g) + np.abs(g - b) + np.abs(r - b))[valid_px].mean()
    else:
        channel_diff = 0.0

    # Crude skin-tone band in RGB space (common in selfies / people photos)
    skin_mask = (
        (r > 95) & (g > 40) & (b > 20) &
        (r > g) & (r > b) &
        ((r - g) > 15) &
        ((maxc - minc) > 15)
    )
    skin_frac = float(skin_mask.mean())

    # Dark background fraction (scans are usually black-bordered)
    gray = rgb.mean(axis=-1)
    dark_frac = float((gray < 25).mean())

    return {
        'mean_saturation': mean_sat,
        'colorful_frac':   colorful_frac,
        'channel_diff':    float(channel_diff),
        'skin_frac':       skin_frac,
        'dark_frac':       dark_frac,
    }


def _structure_stats(rgb: np.ndarray) -> dict:
    """
    Texture / structure statistics to catch content that color-only checks
    miss entirely: scanned documents, flat logos, and blank/solid images.
    Real medical scans have continuous anatomical texture (soft tissue,
    bone, organ boundaries) - very different from crisp text edges,
    flat vector-like logo shapes, or a uniform blank field.
    """
    gray = rgb.mean(axis=-1)

    # Simple gradient-based edge density (Sobel-lite via finite differences)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    edge_density = float(((gx > 25).mean() + (gy > 25).mean()) / 2.0)

    # Overall intensity variance - blank / solid images collapse to ~0
    intensity_std = float(gray.std())

    # White/near-white background fraction - dominant in scanned documents
    white_bg_frac = float((gray > 235).mean())

    # Number of distinct gray levels actually used (rounded to nearest 4).
    # Flat logos and documents use very few discrete tones; real scans use
    # a continuous gradient of hundreds of levels.
    quantized = (gray // 4).astype(np.int32)
    unique_levels = int(np.unique(quantized).size)

    return {
        'edge_density':  edge_density,
        'intensity_std': intensity_std,
        'white_bg_frac': white_bg_frac,
        'unique_levels': unique_levels,
    }


def validate_medical_image(image_input, scan_type: str = "medical scan") -> dict:
    """
    Run heuristic checks to decide whether an uploaded image plausibly
    looks like a medical scan (CT / MRI / Ultrasound) at all.

    Returns
    -------
    dict with keys:
      is_valid        : bool  - False means "block this image, do not run the model"
      confidence      : float - 0-100, how confident we are in the verdict
      reasons         : list[str] - human-readable red flags found
      stats           : dict  - raw statistics, useful for debugging/logging
    """
    try:
        rgb = _to_rgb_array(image_input)
    except Exception as e:
        return {
            'is_valid': False,
            'confidence': 0.0,
            'reasons': [f"Could not read image file: {e}"],
            'stats': {},
        }

    stats = _saturation_stats(rgb)
    struct = _structure_stats(rgb)
    stats.update(struct)
    reasons = []
    votes = 0

    if stats['mean_saturation'] > SATURATION_MAX_MEDICAL:
        votes += 1
        reasons.append(
            f"Image has high color saturation ({stats['mean_saturation']:.0f}), "
            f"typical of natural photos rather than grayscale medical scans."
        )

    if stats['channel_diff'] > CHANNEL_DIFF_MAX_MEDICAL:
        votes += 1
        reasons.append(
            "Strong color-channel divergence detected — medical scans are "
            "almost always near-grayscale even when saved in color format."
        )

    if stats['skin_frac'] > SKIN_TONE_MAX_RATIO:
        votes += 1
        reasons.append(
            f"Detected a significant skin-tone colored region "
            f"({stats['skin_frac']*100:.1f}% of pixels), suggesting a photo "
            f"of a person rather than a scan."
        )

    if stats['colorful_frac'] > COLORFUL_PIXEL_MAX_RATIO:
        votes += 1
        reasons.append(
            f"{stats['colorful_frac']*100:.1f}% of pixels are vividly colored — "
            f"inconsistent with typical {scan_type} imagery."
        )

    if stats['dark_frac'] < MIN_DARK_BACKGROUND_FRAC and stats['mean_saturation'] > 15:
        votes += 1
        reasons.append(
            "No significant dark background region found — most clinical scans "
            "have a black or near-black border/background."
        )

    if stats['edge_density'] < MIN_EDGE_DENSITY and stats['intensity_std'] < MIN_INTENSITY_STD:
        votes += 1
        reasons.append(
            "Image is nearly flat/uniform with almost no internal texture — "
            "real scans always show continuous anatomical detail."
        )

    if stats['white_bg_frac'] > MAX_WHITE_BG_FRAC:
        votes += 1
        reasons.append(
            f"{stats['white_bg_frac']*100:.0f}% of the image is a plain white "
            f"background — this pattern is typical of scanned documents, "
            f"logos, or screenshots, not medical imaging."
        )

    if stats['unique_levels'] < MIN_UNIQUE_GRAY_LEVELS:
        votes += 1
        reasons.append(
            "Very few distinct tones detected — the image looks like flat "
            "graphic content (a logo, icon, or document) rather than a "
            "continuous-tone medical scan."
        )

    is_valid = votes < REJECT_VOTE_THRESHOLD

    # ── Hard overrides ──────────────────────────────────────────────
    # Some signals are unambiguous on their own and should not depend on
    # accumulating enough votes alongside unrelated checks.
    hard_block_reasons = []

    if stats['intensity_std'] < 4.0:
        hard_block_reasons.append(
            "Image is essentially blank or a single solid color — no scan "
            "contains zero internal variation."
        )

    if stats['white_bg_frac'] > 0.80 and stats['edge_density'] < 0.04:
        hard_block_reasons.append(
            "Image is dominated by plain white background with sharp, sparse "
            "marks — consistent with a scanned document or printed page, "
            "not a medical scan."
        )

    if hard_block_reasons:
        is_valid = False
        for r in hard_block_reasons:
            if r not in reasons:
                reasons.append(r)
        votes = max(votes, REJECT_VOTE_THRESHOLD)
    # Confidence framed as "how sure are we this verdict is correct"
    confidence = min(95.0, 55.0 + votes * 12.0) if not is_valid else max(55.0, 90.0 - votes * 12.0)

    return {
        'is_valid':   is_valid,
        'confidence': round(confidence, 1),
        'reasons':    reasons,
        'stats':      stats,
        'votes':      votes,
    }


def get_rejection_message(scan_type: str, validation_result: dict) -> str:
    """Build a clear, non-technical message to show the user when an image is rejected."""
    reasons_text = " ".join(validation_result['reasons'][:2])
    return (
        f"This doesn't look like a valid {scan_type} image. {reasons_text} "
        f"Please upload an actual {scan_type} in grayscale/DICOM-exported format."
    )