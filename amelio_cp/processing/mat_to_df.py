import pandas as pd
import numpy as np
import os
import re
from scipy.io import loadmat
from typing import Union, Sequence, Optional, List, Tuple
from pathlib import Path


# RREAD .mat  FILES IN PYTHON
def load_mat_data(file):

    data = loadmat(file)
    print("--------------------------------")
    print("The data is now loaded!")

    return data


# Organize mat files in the order of PreLokomat and then PostLokomat.
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split("(\d+)", s)]


def organized(directory):
    pre_files = [f for f in os.listdir(directory) if f.endswith("eLokomat.mat")]
    post_files = [f for f in os.listdir(directory) if f.endswith("stLokomat.mat")]
    mat_files_pre_sorted = sorted(pre_files, key=natural_sort_key)
    mat_files_post_sorted = sorted(post_files, key=natural_sort_key)
    mat_files_sorted = mat_files_pre_sorted + mat_files_post_sorted
    print("The files are sorted in the order pre and post training!")
    return mat_files_sorted


def access_struct(data: Union[dict, np.ndarray], structs: Sequence[str]) -> np.ndarray:
    """
    Walk through nested MATLAB structs loaded via SciPy.

    Parameters
    ----------
    data : dict or ndarray
        The root loadmat() dictionary or a nested struct array.
    structs : list of field names to traverse (e.g., ['c', 'results', 'Right', 'angAtFullCycle'])

    Returns
    -------
    np.ndarray
        The final nested object (usually an ndarray/structured-array).
    """
    for struct in structs:
        if isinstance(data, np.ndarray) and data.dtype.names is not None:
            # MATLAB struct represented as 1x1 numpy structured array
            data = data[0, 0][struct]
        else:
            # Root dict access
            data = data[struct]
    return data


# --- HELPERS --- #
def _ensure_column_vector(x: Union[np.ndarray, float, int]) -> np.ndarray:
    """Return x as a (n, 1) float64 column vector."""
    a = np.asarray(x, dtype=float).reshape(-1, 1)
    return a


def _scalar_from_mat_field(arr: np.ndarray) -> float:
    """
    Robustly extract a scalar from a MATLAB field that may be shaped (1,1), (1,), etc.
    """
    return float(np.asarray(arr).squeeze())


def _vector_from_mat_field(arr: np.ndarray, length: Optional[int] = None) -> np.ndarray:
    """
    Robustly extract a 1D vector from a MATLAB field and optionally enforce length.
    """
    v = np.asarray(arr, dtype=float).ravel()
    if length is not None and v.size != length:
        raise ValueError(f"Expected vector of length {length}, got {v.size}")
    return v


def _stack_timeseries_and_scalars(
    ts_cols: List[np.ndarray], scalar_values: List[float], target_len: int = 100
) -> np.ndarray:
    """
    Build a (target_len, n_cols) matrix by horizontally stacking:
      - time-series columns (already shape (target_len, k))
      - scalar columns expanded to (target_len, 1) with 99 NaNs appended below the first value
    This mirrors the original behaviour where non-time-series values were padded to 100 rows.
    """
    mats = []
    if ts_cols:
        mats.append(np.hstack(ts_cols))  # (target_len, ?)
    for val in scalar_values:
        # First row = scalar, remaining rows = NaN to reach target_len
        col = np.vstack([[[val]], np.full((target_len - 1, 1), np.nan)])
        mats.append(col)
    if not mats:
        return np.empty((target_len, 0))
    return np.hstack(mats)


def _joint_headers(
    prefix: str, joint: str, directions: Tuple[str, str, str] = ("flx/ext", "abd/add", "int/ext rot")
) -> List[str]:
    """
    Build 3 headers for a joint (x/y/z) with a prefix.
    Example: prefix='Min_R', joint='Hip' -> ['Min_RHip_flx/ext', 'Min_RHip_abd/add', 'Min_RHip_int/ext rot']
    """
    return [f"{prefix}{joint}_{d}" for d in directions]


def _side_char(side_struct: str) -> str:
    """Return 'R' for 'Right', 'L' for 'Left'."""
    return side_struct[0]


# --- EXTRACTORS --- #
def _extract_ang_full_cycle_timeseries(all_data: np.ndarray, joint_names: Sequence[str], side_char: str) -> np.ndarray:
    """
    Extract 100x3 per joint time series for angAtFullCycle, horizontally stacked.
    """
    cols = []
    for joint in joint_names:
        key = side_char + joint  # e.g., 'RHip'
        joint_kin = all_data[0, 0][key][0][0]  # MATLAB field
        joint_kin = np.reshape(joint_kin, (100, 3), order="F")  # (100,3)
        cols.append(joint_kin)
    return np.hstack(cols) if cols else np.empty((100, 0))


def _extract_ang_full_cycle_mean(all_data: np.ndarray, joint_names: Sequence[str], side_char: str) -> np.ndarray:
    """
    Extract mean (over 100 samples) per direction for angAtFullCycle: 3 numbers per joint.
    Returns shape (3 * len(joint_names),).
    """
    parts = []
    for joint in joint_names:
        key = side_char + joint  # e.g., 'RHip'
        joint_kin = all_data[0, 0][key][0][0]
        joint_kin = np.reshape(joint_kin, (100, 3), order="F")  # (100,3)
        parts.append(np.mean(joint_kin, axis=0))  # (3,)
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _extract_minmax_vector(all_data: np.ndarray, joint_names: Sequence[str], side_char: str) -> np.ndarray:
    """
    Extract a 3-vector per joint from e.g., angMinAtFullStance or angMaxAtFullStance.
    Shape: (3 * len(joint_names),).
    """
    parts = []
    for joint in joint_names:
        key = side_char + joint
        vec = all_data[0, 0][key]  # shape varies: use robust ravel
        parts.append(_vector_from_mat_field(vec, length=3))
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def MinMax_feature_extractor(
    *,
    directory: Union[str, Path],
    measurements: List[str],
    separate_legs: bool,
    joint_names: Sequence[str] = ("Pelvis", "Hip", "Knee", "Ankle", "FootProgress"),
):
    """
    Compute min and max values for joint DoFs during stance and return them.

    Notes
    -----
    - No files are written. `output_dir` is ignored (kept only for compatibility).
    - Ensures 'angMaxAtFullStance' and 'angMinAtFullStance' are included in `measurements`.
    - For 'angMinAtFullStance' and 'angMaxAtFullStance': extracts per-joint 3-vectors (x,y,z).
    - For 'baseSustentation': extracts side-specific 'maxPreMoyenne'.
    - Other measurements are treated as scalars.
    - Returns a single table aggregating all subjects (and both legs if `separate_legs=False`).

    Returns
    -------
    pandas.DataFrame (default) or np.ndarray if `output_shape != pd.DataFrame`
    """
    directory = Path(directory)

    measurements = list(measurements)
    if "angMaxAtFullStance" not in measurements:
        measurements.insert(0, "angMaxAtFullStance")
    if "angMinAtFullStance" not in measurements:
        measurements.insert(0, "angMinAtFullStance")

    joint_names = list(joint_names)
    side_structs = ["Right", "Left"]
    prepost_files = organized(directory)

    rows: List[np.ndarray] = []
    headers: Optional[List[str]] = None

    for idx, fname in enumerate(prepost_files, start=1):
        data = load_mat_data(directory / fname)

        if separate_legs:
            for side in side_structs:
                side_c = _side_char(side)
                row_parts: List[np.ndarray] = []
                header_parts: List[str] = []

                for measurement in measurements:
                    all_data = access_struct(data, ["c", "results", side, measurement])

                    if measurement == "angMinAtFullStance":
                        vecs = _extract_minmax_vector(all_data, joint_names, side_c)
                        row_parts.append(vecs)
                        for j in joint_names:
                            header_parts += _joint_headers(prefix=f"Min_{side_c}", joint=j)

                    elif measurement == "angMaxAtFullStance":
                        vecs = _extract_minmax_vector(all_data, joint_names, side_c)
                        row_parts.append(vecs)
                        for j in joint_names:
                            header_parts += _joint_headers(prefix=f"Max_{side_c}", joint=j)

                    elif measurement == "baseSustentation":
                        val = _scalar_from_mat_field(all_data[0, 0]["maxPreMoyenne"][0])
                        row_parts.append(np.array([val]))
                        header_parts.append(f"Max_{side}_{'BOS'}")

                    else:
                        val = _scalar_from_mat_field(all_data[0][0])
                        row_parts.append(np.array([val]))
                        header_parts.append(measurement)

                row = np.concatenate(row_parts)[None, :]
                rows.append(row)
                if headers is None:
                    headers = header_parts.copy()

        else:
            row_parts: List[np.ndarray] = []
            header_parts: List[str] = []

            for side in side_structs:
                side_c = _side_char(side)
                for measurement in measurements:
                    all_data = access_struct(data, ["c", "results", side, measurement])

                    if measurement == "angMinAtFullStance":
                        vecs = _extract_minmax_vector(all_data, joint_names, side_c)
                        row_parts.append(vecs)
                        for j in joint_names:
                            header_parts += _joint_headers(prefix=f"Min_{side_c}", joint=j)

                    elif measurement == "angMaxAtFullStance":
                        vecs = _extract_minmax_vector(all_data, joint_names, side_c)
                        row_parts.append(vecs)
                        for j in joint_names:
                            header_parts += _joint_headers(prefix=f"Max_{side_c}", joint=j)

                    elif measurement == "baseSustentation":
                        val = _scalar_from_mat_field(all_data[0, 0]["maxPreMoyenne"][0])
                        row_parts.append(np.array([val]))
                        header_parts.append(f"Max_{side}_{'BOS'}")

                    else:
                        val = _scalar_from_mat_field(all_data[0][0])
                        row_parts.append(np.array([val]))
                        header_parts.append(measurement)

            row = np.concatenate(row_parts)[None, :]
            rows.append(row)
            if headers is None:
                headers = header_parts.copy()

    # Aggregate all subjects
    all_mat = np.vstack(rows) if rows else np.empty((0, 0))
    headers = headers or []
    all_df = pd.DataFrame(all_mat, columns=headers)

    return all_df
