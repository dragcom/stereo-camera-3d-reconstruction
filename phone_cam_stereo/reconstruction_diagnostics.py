"""Quality checks for a rectified stereo reconstruction."""

from pathlib import Path

import cv2
import numpy as np


def _status(value, excellent, acceptable):
    if value >= excellent:
        return "GOOD"
    if value >= acceptable:
        return "MARGINAL"
    return "FAIL"


def _right_matcher(left_matcher):
    """Create the matching right-to-left SGBM matcher for consistency testing."""
    disparities = left_matcher.getNumDisparities()
    block_size = left_matcher.getBlockSize()
    return cv2.StereoSGBM_create(
        minDisparity=-disparities,
        numDisparities=disparities,
        blockSize=block_size,
        P1=left_matcher.getP1(),
        P2=left_matcher.getP2(),
        disp12MaxDiff=left_matcher.getDisp12MaxDiff(),
        preFilterCap=left_matcher.getPreFilterCap(),
        uniquenessRatio=left_matcher.getUniquenessRatio(),
        speckleWindowSize=left_matcher.getSpeckleWindowSize(),
        speckleRange=left_matcher.getSpeckleRange(),
        mode=left_matcher.getMode(),
    )


def evaluate_reconstruction(
    rectified_left,
    rectified_right,
    disparity_left,
    points_3d,
    matcher,
    output_dir,
    name="scene",
    min_depth_m=0.1,
    max_depth_m=10.0,
    exhaustive_left_right_check=False,
):
    """Evaluate matching coverage, depth distribution, and optional consistency."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    lines = []

    def note(message=""):
        print(message)
        lines.append(message)

    height, width = disparity_left.shape
    valid_left = np.isfinite(disparity_left) & (disparity_left > 1.0)
    valid_depth = valid_left & np.isfinite(points_3d[:, :, 2])
    valid_depth &= (points_3d[:, :, 2] > min_depth_m) & (points_3d[:, :, 2] < max_depth_m)

    if exhaustive_left_right_check:
        gray_left = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
        disparity_right = _right_matcher(matcher).compute(gray_right, gray_left).astype(np.float32) / 16.0
        yy, xx = np.indices((height, width))
        matched_x = np.rint(xx - disparity_left).astype(np.int32)
        inside = valid_left & (matched_x >= 0) & (matched_x < width)
        right_values = np.zeros_like(disparity_left)
        right_values[inside] = disparity_right[yy[inside], matched_x[inside]]
        left_right_error = np.full_like(disparity_left, np.nan)
        left_right_error[inside] = np.abs(disparity_left[inside] + right_values[inside])
        consistent = inside & (right_values < -1.0) & (left_right_error <= 1.0)
        consistency_percent = 100.0 * consistent.sum() / max(1, valid_left.sum())
    else:
        consistent = valid_left.copy()
        inside = valid_left
        left_right_error = None
        consistency_percent = None

    valid_percent = 100.0 * valid_left.mean()
    depth_percent = 100.0 * valid_depth.mean()
    disparity_values = disparity_left[valid_left]
    depth_values = points_3d[:, :, 2][valid_depth]

    note("\n========== RECONSTRUCTION EVALUATION ==========")
    note(f"Image: {name}; size: {width} x {height}")
    note("\nMATCHING COVERAGE")
    note(f"Positive-disparity coverage: {valid_percent:.1f}% [{_status(valid_percent, 50, 25)}]")
    note(f"Physically filtered depth coverage ({min_depth_m:g}-{max_depth_m:g} m): {depth_percent:.1f}%")
    if valid_left.any():
        note(f"Disparity px: median/p05/p95={np.median(disparity_values):.2f}/{np.percentile(disparity_values, 5):.2f}/{np.percentile(disparity_values, 95):.2f}")
    else:
        note("  FAIL: no positive disparity. Check image order, rectification, texture, and matcher range.")

    note("\nLEFT-RIGHT CONSISTENCY")
    if consistency_percent is None:
        note("  Skipped in fast mode. Set EXHAUSTIVE_LEFT_RIGHT_CHECK=True for the slower independent match check.")
    else:
        note(f"Consistent matches (<= 1 px disagreement): {consistency_percent:.1f}% [{_status(consistency_percent, 85, 60)}]")
        if np.any(inside):
            note(f"Left-right disagreement px: median/p95={np.nanmedian(left_right_error):.2f}/{np.nanpercentile(left_right_error, 95):.2f}")
        if consistency_percent < 60:
            note("  FAIL: matching is unstable. Recheck calibration/rectification, image synchronisation, texture, and SGBM settings.")

    note("\nDEPTH DISTRIBUTION")
    if depth_values.size:
        note(f"Depth m: median/p05/p95={np.median(depth_values):.3f}/{np.percentile(depth_values, 5):.3f}/{np.percentile(depth_values, 95):.3f}")
    else:
        note("  FAIL: no points remain after depth filtering. Calibration scale or disparity is invalid.")

    # White means consistent; red means a left-match without a reliable right match.
    consistency_map = np.zeros((height, width, 3), dtype=np.uint8)
    consistency_map[valid_left] = (0, 0, 180)
    consistency_map[consistent] = (220, 220, 220)
    cv2.imwrite(str(output / f"{name}_lr_consistency.png"), consistency_map)

    depth_preview = np.zeros((height, width), dtype=np.uint8)
    if depth_values.size:
        low, high = np.percentile(depth_values, [2, 98])
        clipped = np.clip(points_3d[:, :, 2], low, high)
        depth_preview[valid_depth] = ((clipped[valid_depth] - low) * 255.0 / max(high - low, 1e-6)).astype(np.uint8)
    cv2.imwrite(str(output / f"{name}_depth_quality.png"), cv2.applyColorMap(depth_preview, cv2.COLORMAP_TURBO))
    (output / f"{name}_reconstruction_report.txt").write_text("\n".join(lines) + "\n")
    return {"valid_percent": valid_percent, "consistency_percent": consistency_percent, "depth_percent": depth_percent}
