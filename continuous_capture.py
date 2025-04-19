# capture_and_visualize.py

import os
import time
import sys
import cv2
import numpy as np
import capture_util  # Make sure capture_util.py is alongside this script

# --- Configuration ---
OUTPUT_FOLDER = "img"      # Where to save raw frames (optional)
DELAY_SECONDS = 1.0         # Capture interval in seconds
APP_NAME = capture_util.APP_NAME
WINDOW_TITLE = capture_util.WINDOW_TITLE
VIS_WINDOW_NAME = "Clash Royale Analysis"
# ------------------------


def ensure_dir(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory '{folder_path}': {e}", file=sys.stderr)
        sys.exit(1)


def extract_game_stats(img):
    h, w = img.shape[:2]
    # Tower health bar ROIs (tweak fractions if needed)
    bar_h = int(0.03 * h)
    bar_y = int(0.06 * h)
    lx1, lx2 = int(0.20 * w), int(0.40 * w)
    rx1, rx2 = int(0.60 * w), int(0.80 * w)

    def hp_pct(roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower = np.array([50, 150, 150])
        upper = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        cols = np.any(mask, axis=0)
        if not np.any(cols):
            return 0.0
        left = np.argmax(cols)
        right = len(cols) - 1 - np.argmax(cols[::-1])
        return (right - left) / float(len(cols))

    roi_left = img[bar_y:bar_y+bar_h, lx1:lx2]
    roi_right = img[bar_y:bar_y+bar_h, rx1:rx2]
    left_hp = hp_pct(roi_left)
    right_hp = hp_pct(roi_right)

    # Elixir bar ROI
    el_y1 = int(0.88 * h)
    el_y2 = int(0.97 * h)
    el_x1 = int(0.25 * w)
    el_x2 = int(0.75 * w)
    roi_el = img[el_y1:el_y2, el_x1:el_x2]
    hsv_el = cv2.cvtColor(roi_el, cv2.COLOR_BGR2HSV)
    lower_p = np.array([130, 100, 100])
    upper_p = np.array([160, 255, 255])
    mask_el = cv2.inRange(hsv_el, lower_p, upper_p)
    num_labels, _ = cv2.connectedComponents(mask_el)
    elixir = max(0, num_labels - 1)

    return (left_hp, right_hp, elixir), ((lx1, bar_y, lx2, bar_y+bar_h),
                                          (rx1, bar_y, rx2, bar_y+bar_h),
                                          (el_x1, el_y1, el_x2, el_y2))


def annotate_image(img, stats, rois):
    (left_hp, right_hp, elixir) = stats
    ((lx1, ly1, lx2, ly2), (rx1, ry1, rx2, ry2), (ex1, ey1, ex2, ey2)) = rois
    vis = img.copy()

    # Draw health bar outlines
    cv2.rectangle(vis, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
    # Fill health bars
    cv2.rectangle(vis, (lx1, ly1), (lx1 + int(left_hp * (lx2-lx1)), ly2), (0, 255, 0), -1)
    cv2.rectangle(vis, (rx1, ry1), (rx1 + int(right_hp * (rx2-rx1)), ry2), (0, 255, 0), -1)
    # Annotate percentages
    cv2.putText(vis, f"L: {int(left_hp*100)}%", (lx1, ly1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(vis, f"R: {int(right_hp*100)}%", (rx1, ry1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # Draw elixir ROI and text
    cv2.rectangle(vis, (ex1, ey1), (ex2, ey2), (255, 0, 255), 2)
    cv2.putText(vis, f"Elixir: {elixir}", (ex1, ey1-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)

    return vis


def main():
    # Prepare output folder if needed
    out_dir = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    ensure_dir(out_dir)

    cv2.namedWindow(VIS_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VIS_WINDOW_NAME, 800, 600)
    # You can move the window to left monitor with moveWindow(x, y)
    # cv2.moveWindow(VIS_WINDOW_NAME, 0, 0)

    frame_count = 0
    try:
        while True:
            geom = capture_util.get_window_geometry(APP_NAME, WINDOW_TITLE)
            if not geom:
                print(f"Window not found, retrying...", file=sys.stderr)
                time.sleep(DELAY_SECONDS)
                continue

            x, y, w, h = geom
            ts = int(time.time())
            fname = f"frame_{ts}.png"
            fpath = os.path.join(out_dir, fname)
            if not capture_util.capture_window_region(x, y, w, h, fpath):
                time.sleep(DELAY_SECONDS)
                continue

            img = cv2.imread(fpath)
            if img is None:
                print(f"Failed to load captured image '{fpath}'", file=sys.stderr)
                time.sleep(DELAY_SECONDS)
                continue

            stats, rois = extract_game_stats(img)
            annotated = annotate_image(img, stats, rois)

            # Side-by-side: analysis left, original right
            combined = np.hstack((annotated, img))
            cv2.imshow(VIS_WINDOW_NAME, combined)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        print(f"\nStopped after {frame_count} frames.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()