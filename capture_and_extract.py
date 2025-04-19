# capture_and_visualize.py

import os
import time
import sys
import cv2
import numpy as np
from PIL import ImageGrab  # Use in-memory grabs instead of writing to disk
import capture_util  # Only used for window geometry

# --- Configuration ---
DELAY_SECONDS = 1.0         # Capture interval in seconds
APP_NAME = capture_util.APP_NAME
WINDOW_TITLE = capture_util.WINDOW_TITLE
VIS_WINDOW_NAME = "Clash Royale Analysis"
SAVE_ANNOTATED = False      # Set True to save annotated frames
OUTPUT_FOLDER = "annotated"  # Only used if SAVE_ANNOTATED == True
# ------------------------

# Create folder for annotated frames if needed
if SAVE_ANNOTATED:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def grab_frame():
    geom = capture_util.get_window_geometry(APP_NAME, WINDOW_TITLE)
    if not geom:
        return None
    x, y, w, h = geom
    # PIL grab uses (left, top, right, bottom)
    bbox = (x, y, x + w, y + h)
    img_pil = ImageGrab.grab(bbox)
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img


def extract_game_stats(img):
    h, w = img.shape[:2]
    # Health bar ROIs (tweak fractions as needed)
    bar_h = int(0.03 * h)
    bar_y = int(0.06 * h)
    lx1, lx2 = int(0.20 * w), int(0.40 * w)
    rx1, rx2 = int(0.60 * w), int(0.80 * w)

    def hp_pct(roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([50,150,150]), np.array([90,255,255]))
        cols = np.any(mask, axis=0)
        if not np.any(cols):
            return 0.0
        left = np.argmax(cols)
        right = len(cols) - 1 - np.argmax(cols[::-1])
        return (right - left) / float(len(cols))

    roi_left  = img[bar_y:bar_y+bar_h, lx1:lx2]
    roi_right = img[bar_y:bar_y+bar_h, rx1:rx2]
    left_hp   = hp_pct(roi_left)
    right_hp  = hp_pct(roi_right)

    # Elixir ROI
    el_y1, el_y2 = int(0.88*h), int(0.97*h)
    el_x1, el_x2 = int(0.25*w), int(0.75*w)
    roi_el = img[el_y1:el_y2, el_x1:el_x2]
    hsv_el = cv2.cvtColor(roi_el, cv2.COLOR_BGR2HSV)
    mask_el = cv2.inRange(hsv_el, np.array([130,100,100]), np.array([160,255,255]))
    num_labels, _ = cv2.connectedComponents(mask_el)
    elixir = max(0, num_labels - 1)

    rois = ((lx1, bar_y, lx2, bar_y+bar_h),
            (rx1, bar_y, rx2, bar_y+bar_h),
            (el_x1, el_y1, el_x2, el_y2))
    stats = (left_hp, right_hp, elixir)
    return stats, rois


def annotate_image(img, stats, rois):
    left_hp, right_hp, elixir = stats
    (lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2), (ex1, ey1, ex2, ey2) = rois
    vis = img.copy()

    # Health bars
    cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0,255,0), 2)
    cv2.rectangle(vis, (lx1, ly1), (lx1 + int(left_hp*(lx2-lx1)), ly2), (0,255,0), -1)
    cv2.putText(vis, f"L: {int(left_hp*100)}%", (lx1, ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0,255,0), 2)
    cv2.rectangle(vis, (rx1, ry1), (rx1 + int(right_hp*(rx2-rx1)), ry2), (0,255,0), -1)
    cv2.putText(vis, f"R: {int(right_hp*100)}%", (rx1, ry1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Elixir
    cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (255,0,255), 2)
    cv2.putText(vis, f"Elixir: {elixir}", (ex1, ey1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    return vis


def main():
    cv2.namedWindow(VIS_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIS_WINDOW_NAME, 800, 600)

    frame_count = 0
    try:
        while True:
            img = grab_frame()
            if img is None:
                print("Failed to grab frame, retrying...", file=sys.stderr)
                time.sleep(DELAY_SECONDS)
                continue

            stats, rois = extract_game_stats(img)
            annotated = annotate_image(img, stats, rois)

            # Combine annotated (left) with raw (right)
            combined = np.hstack((annotated, img))
            cv2.imshow(VIS_WINDOW_NAME, combined)

            if SAVE_ANNOTATED:
                out_path = os.path.join(OUTPUT_FOLDER, f"ann_{frame_count}.png")
                cv2.imwrite(out_path, annotated)

            frame_count += 1
            key = cv2.waitKey(int(DELAY_SECONDS*1000)) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print(f"\nStopped after {frame_count} frames.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()