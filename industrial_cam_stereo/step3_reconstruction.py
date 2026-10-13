"""Step 3: live 3D reconstruction with the new calibration (denoised version)
=====================================================================
Improvements over the previous version:
  1. On SPACE, captures 9 frames and takes the per-pixel median disparity -
     temporal median filtering removes random matching noise, the most
     effective denoising trick for a static scene;
  2. Tighter SGBM parameters: uniquenessRatio 10->15,
     speckleWindowSize 100->200 - prefer no match over a wrong match,
     and larger speckle blobs are removed entirely;
  3. Voxel filtering of the point cloud: any 5 mm voxel containing fewer
     than 3 points is discarded, removing floating flyaway points while
     real surfaces (point spacing ~1-2 mm) are untouched;
  4. Imaging parameters (exposure / white balance / gain) are left fully
     to the camera's auto mode - this script does not touch them;
  5. Sharpness equalisation: if the left/right sharpness gap exceeds 1.3x
     (factory focus tolerance on fixed-focus lenses can leave one eye
     softer than the other), the sharper eye is lightly blurred before
     matching; colours are still taken from the original image.

Usage:
  1. Make sure the calibration file is in the project folder
     (stereo_calib_2.npz or stereo_calib_v2.npz)
  2. Set up the scene: texture-rich objects (books, pen holders, cups,
     small items) at staggered depths of 60 cm ~ 1.2 m
  3. Run:  python step3_reconstruction.py
  4. Two windows pop up: left = rectified stereo views,
     right = live disparity map (bright = near, dark = far)
     [Click a window first so it has keyboard focus]
     SPACE = capture 9 frames and save the scene
             (keep the camera and objects perfectly still while capturing)
     Q = quit
  5. Output goes to reconstruction_output/ - open shotNN.ply in MeshLab
"""

import cv2
import numpy as np
from pathlib import Path

# ===== Configuration =====
CAM_INDEX = 1                    # USB device index of the stereo module
FRAME_SIZE = (2560, 720)         # side-by-side stitched frame; halved per eye
WARMUP_FRAMES = 30               # warm-up frames (this module outputs black frames at first)
OUTPUT_DIR = "reconstruction_output"

# Denoising parameters
MEDIAN_FRAMES = 9                # frames captured per shot (odd number; scene must be static)
VOXEL_M = 0.005                  # voxel filter cell size: 5 mm
VOXEL_MIN_POINTS = 3             # voxels with fewer points are discarded as noise

# Calibration file auto-detection
NPZ_CANDIDATES = ["stereo_calib_v2.npz", "stereo_calib.npz"]

# Depth filter range (metres): close-range desktop scene
Z_MIN, Z_MAX = 0.15, 5.0


def load_calibration():
    for name in NPZ_CANDIDATES:
        if Path(name).exists():
            print(f"Using calibration file: {name}")
            calib = np.load(name)
            img_size = tuple(int(x) for x in calib["img_size"])
            return (calib["K1"], calib["D1"], calib["K2"], calib["D2"],
                    calib["R1"], calib["R2"], calib["P1"], calib["P2"],
                    calib["Q"], img_size)
    raise SystemExit("Calibration file not found! Make sure stereo_calib_v2.npz "
                     "is in this folder and StereoCalibration_v3.py has been run.")


def save_ply(path, points, colors):
    header = ("ply\nformat ascii 1.0\nelement vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n" % len(points))
    with open(path, "w") as f:
        f.write(header)
        for p, c in zip(points, colors):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")


def remove_outliers_voxel(pts, cls, voxel=VOXEL_M, min_count=VOXEL_MIN_POINTS):
    """Voxel filter: divide space into `voxel`-sized cells; cells with too few
    points are treated as noise. Real surfaces have point spacing of ~1-2 mm
    (a 5 mm cell holds a dozen points); flyaway points are isolated and removed."""
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    keep = counts[inv] >= min_count
    return pts[keep], cls[keep]


def main():
    K1, D1, K2, D2, R1, R2, P1, P2, Q, img_size = load_calibration()

    # Precompute rectification maps (only once)
    map1l, map2l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map1r, map2r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)

    # SGBM stereo matcher (tightened: prefer no match over a wrong match)
    block = 7
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,        # must be a multiple of 16
        blockSize=block,
        P1=8 * 3 * block ** 2,
        P2=32 * 3 * block ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,        # 10 -> 15: a match must clearly beat the runner-up
        speckleWindowSize=200,     # 100 -> 200: larger speckle blobs are also removed
        speckleRange=4,
    )

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera (device index {CAM_INDEX})")

    # Exposure / white balance / gain are left fully to the camera's auto mode;
    # this script does not touch any imaging parameters.
    # NOTE: FRAME_WIDTH/HEIGHT are NOT imaging parameters - they select the
    # resolution mode and must stay: calibration was done at 1280x720 per eye,
    # so a different resolution would break the left/right split alignment.
    # ALSO NOTE: if the calibration capture script (which locks manual
    # parameters) has been run since, run restore_camera_auto.py once or
    # replug the camera before running this script.

    # Warm-up: discard the first black frames
    print(f"Warming up the camera ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        cap.read()

    def grab_rectified():
        """Grab one frame and rectify it; returns (raw left, raw right, rectified left, rectified right)."""
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Frame grab failed")
        h, w = frame.shape[:2]
        left, right = frame[:, : w // 2], frame[:, w // 2:]
        rl = cv2.remap(left, map1l, map2l, cv2.INTER_LINEAR)
        rr = cv2.remap(right, map1r, map2r, cv2.INTER_LINEAR)
        return left, right, rl, rr

    def gray_sharpness(bgr):
        """Laplacian variance: larger = sharper."""
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(g, cv2.CV_64F).var()

    # ===== Sharpness equalisation: the two fixed-focus lenses may have
    # different factory focus distances, leaving one eye softer. The same
    # object then "looks different" in the two views, which badly hurts
    # matching. Measure the sharpness gap and lightly blur the SHARPER eye
    # down to the softer one's level - used for matching only; colours
    # still come from the original image. =====
    h_, w_ = img_size[1], img_size[0]
    ratios = []
    for _ in range(5):
        _, _, tl, tr = grab_rectified()
        # measure on the central region (avoid the distorted edges)
        sl = gray_sharpness(tl[h_ // 4: 3 * h_ // 4, w_ // 4: 3 * w_ // 4])
        sr = gray_sharpness(tr[h_ // 4: 3 * h_ // 4, w_ // 4: 3 * w_ // 4])
        if min(sl, sr) > 1:
            ratios.append(max(sl, sr) / min(sl, sr))
    sharp_ratio = float(np.median(ratios)) if ratios else 1.0
    blur_eye, blur_sigma = None, 0.0
    if sharp_ratio > 1.3:
        blur_sigma = min(3.0, float(np.sqrt(sharp_ratio)))
        blur_eye = "L" if sl > sr else "R"
        print(f"  Sharpness equalisation: eyes differ by {sharp_ratio:.1f}x -> "
              f"blurring the {'left' if blur_eye == 'L' else 'right'} eye "
              f"with sigma={blur_sigma:.1f} before matching")
    else:
        print(f"  Sharpness equalisation: eyes are close enough "
              f"({sharp_ratio:.2f}x), no processing needed")

    def compute_disparity(rl, rr):
        """SGBM matching; returns float disparity (divided by 16).
        Sharpness equalisation is applied to the matching input first."""
        ml, mr = rl, rr
        if blur_eye == "L":
            ml = cv2.GaussianBlur(rl, (0, 0), blur_sigma)
        elif blur_eye == "R":
            mr = cv2.GaussianBlur(rr, (0, 0), blur_sigma)
        return sgbm.compute(cv2.cvtColor(ml, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(mr, cv2.COLOR_BGR2GRAY)).astype(np.float32) / 16.0

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    shot = len(list(Path(OUTPUT_DIR).glob("*.ply")))  # continue numbering

    print("\nReady! [Click a window first] SPACE = capture 9 frames and save, Q = quit")
    print("Brighter disparity = closer; keep the camera and objects still while capturing\n")

    while True:
        left, right, rl, rr = grab_rectified()
        disp = compute_disparity(rl, rr)

        # Display: normalised disparity + colour map
        disp_show = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_show, cv2.COLORMAP_JET)
        cv2.putText(disp_color, "bright = near, dark = far", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("rectified L | R", np.hstack([rl, rr]))
        cv2.imshow("disparity", disp_color)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break
        if key != ord(" "):
            continue

        # ===== Capture MEDIAN_FRAMES frames, per-pixel median disparity =====
        print(f"  Capturing {MEDIAN_FRAMES} frames for median filtering - hold still...")
        disps = [disp]
        for _ in range(MEDIAN_FRAMES - 1):
            _, _, fl, fr = grab_rectified()
            disps.append(compute_disparity(fl, fr))
        disp_med = np.median(np.stack(disps), axis=0)

        # ===== Save the scene =====
        shot += 1
        tag = f"shot{shot:02d}"

        # Reproject to 3D
        points = cv2.reprojectImageTo3D(disp_med, Q)
        colors = cv2.cvtColor(rl, cv2.COLOR_BGR2RGB)
        valid = (disp_med > 1.0) & (points[:, :, 2] > Z_MIN) & (points[:, :, 2] < Z_MAX)
        pts, cls = points[valid], colors[valid]

        # Voxel filter to remove flyaway points
        pts, cls = remove_outliers_voxel(pts, cls)

        ply_path = Path(OUTPUT_DIR, f"{tag}.ply")
        save_ply(ply_path, pts, cls)
        cv2.imwrite(str(Path(OUTPUT_DIR, f"{tag}_raw_left.png")), left)
        cv2.imwrite(str(Path(OUTPUT_DIR, f"{tag}_raw_right.png")), right)
        cv2.imwrite(str(Path(OUTPUT_DIR, f"{tag}_rectified.png")), np.hstack([rl, rr]))

        # Save the median-filtered disparity map (matches the PLY)
        disp_med_show = cv2.normalize(disp_med, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cv2.imwrite(str(Path(OUTPUT_DIR, f"{tag}_disparity.png")),
                    cv2.applyColorMap(disp_med_show, cv2.COLORMAP_JET))
        print(f"  [+] {tag}: saved, filtered cloud has {len(pts):,} points -> {ply_path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone! Open the .ply files in {OUTPUT_DIR}/ with MeshLab to view the point clouds")


if __name__ == "__main__":
    main()
