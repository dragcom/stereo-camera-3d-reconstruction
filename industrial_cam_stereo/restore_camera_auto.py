"""restore_camera_auto.py
==========================================
The calibration script (StereoCalibration_v3.py --capture) locks the camera
for stable image quality: manual exposure, gain 0, white balance 4600K.
These settings PERSIST in the Windows driver after the script exits, so any
photos taken afterwards may show:
  - horizontal banding across the whole frame (short exposure + light flicker)
  - a yellowish colour cast (white balance locked)

Run this script once to restore full auto mode. A preview window opens so you
can visually confirm the image is back to normal - press Q to quit. If it
still looks wrong, unplug the camera's USB cable, plug it back in, and rerun.
"""

import cv2

CAM_INDEX = 1   # USB device index of the stereo module

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera (device index {CAM_INDEX})")

# Restore auto exposure + auto white balance
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)   # 0.75 = auto (DSHOW convention)
cap.set(cv2.CAP_PROP_AUTO_WB, 1)            # 1 = auto white balance

# Read back to confirm
print(f"Auto exposure: {cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):.2f}  (0.75 = auto)")
print(f"Exposure:      {cap.get(cv2.CAP_PROP_EXPOSURE):.1f}")
print(f"Gain:          {cap.get(cv2.CAP_PROP_GAIN):.1f}")

# Warm up, then open a preview so you can confirm the banding is gone
for _ in range(30):
    cap.read()
print("\nPreview open - confirm the image looks normal, then press Q to quit "
      "(click the window first)")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("camera check - press Q to quit", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Auto mode restored.")
