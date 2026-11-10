"""ROI utilities for phantom statistical analysis.

Provides binary tissue mask loading from the MATLAB ``IntMask.mat`` file,
with a per-phantom CSV fallback. Used to restrict μ statistics to tissue
pixels for the phantom validation experiments.
"""
from __future__ import annotations

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IntMask loaders
# ---------------------------------------------------------------------------

def load_intmask_from_mat(
    mat_path: str,
    phantom_index: int,
    bscan_index: int | None = None,
) -> np.ndarray:
    """Load a binary tissue mask from the MATLAB IntMask.mat (HDF5) file.

    The HDF5 dataset ``"phantomMask"`` has shape
    ``(10, 512, 500, 500)`` = (phantoms, bscans, alines, depth) in HDF5
    row-major storage.  Each (alines, depth) slice is transposed to
    (depth, alines) to match the project's (depth, a_lines) convention.

    Args:
        mat_path: Absolute or relative path to ``IntMask.mat``.
        phantom_index: Tomogram index in [0, 9] selecting the phantom.
        bscan_index: B-scan index in [0, 511].  If ``None``, return the
            full 3-D array for all 512 B-scans.

    Returns:
        Boolean ndarray of shape ``(500, 500)`` when ``bscan_index`` is
        given, or ``(512, 500, 500)`` when ``bscan_index`` is ``None``.

    Raises:
        FileNotFoundError: If ``mat_path`` does not exist.
        ImportError: If ``h5py`` is not installed.
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required to load IntMask.mat. "
            "Install it with: pip install h5py"
        ) from exc

    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"IntMask.mat not found at: {mat_path}")

    if not 0 <= phantom_index <= 9:
        raise ValueError(f"phantom_index must be in [0, 9], got {phantom_index}")
    if bscan_index is not None and not 0 <= bscan_index <= 511:
        raise ValueError(f"bscan_index must be in [0, 511], got {bscan_index}")

    with h5py.File(mat_path, "r") as f:
        dataset = f["phantomMask"]
        if bscan_index is None:
            # Load all bscans for this phantom: shape (512, alines, depth)
            raw = dataset[phantom_index, :, :, :]  # (512, 500, 500) alines, depth
            # Transpose each slice from (alines, depth) -> (depth, alines)
            mask = raw.transpose(0, 2, 1).astype(bool)
        else:
            # Load single bscan: shape (alines, depth)
            raw = dataset[phantom_index, bscan_index, :, :]  # (500, 500)
            mask = raw.T.astype(bool)  # (depth, alines)

    return mask


def load_intmask_from_csv(csv_path: str) -> np.ndarray:
    """Load a binary tissue mask from a CSV export of the IntMask.

    The CSV file has 500 rows x 500 columns of comma-separated 0/1
    integers with no header, already in (depth, alines) shape.

    Args:
        csv_path: Absolute or relative path to the CSV file.

    Returns:
        Boolean ndarray of shape ``(500, 500)``.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"IntMask CSV not found at: {csv_path}")

    mask = np.loadtxt(csv_path, delimiter=",", dtype=np.uint8).astype(bool)
    return mask


def load_intmask(
    phantom_index: int,
    bscan_index: int,
    mat_dir: str | None = None,
) -> np.ndarray:
    """Load a binary tissue mask, preferring .mat over CSV fallback.

    Tries to load from ``{mat_dir}/IntMask.mat`` via
    :func:`load_intmask_from_mat`.  Falls back to the per-phantom CSV
    export ``{mat_dir}/IntMask_phantom{phantom_index:02d}_bscan256.csv``
    if h5py is unavailable or the .mat file does not exist.

    Args:
        phantom_index: Tomogram index in [0, 9].
        bscan_index: B-scan index in [0, 511].
        mat_dir: Directory containing the IntMask files.  Defaults to
            ``"matlab_CF_scripts_results"`` relative to the project root.

    Returns:
        Boolean ndarray of shape ``(500, 500)``.
    """
    if mat_dir is None:
        mat_dir = "matlab_CF_scripts_results"

    if not 0 <= phantom_index <= 9:
        raise ValueError(f"phantom_index must be in [0, 9], got {phantom_index}")
    if not 0 <= bscan_index <= 511:
        raise ValueError(f"bscan_index must be in [0, 511], got {bscan_index}")

    mat_path = os.path.join(mat_dir, "IntMask.mat")
    try:
        mask = load_intmask_from_mat(mat_path, phantom_index, bscan_index)
        logger.info("Loaded IntMask from .mat: phantom %d, bscan %d", phantom_index, bscan_index)
        return mask
    except (FileNotFoundError, ImportError) as exc:
        logger.warning("Falling back to CSV IntMask: %s", exc)

    csv_path = os.path.join(mat_dir, f"IntMask_phantom{phantom_index:02d}_bscan256.csv")
    mask = load_intmask_from_csv(csv_path)
    logger.info("Loaded IntMask from CSV: %s", csv_path)
    return mask
