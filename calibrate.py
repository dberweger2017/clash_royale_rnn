#!/usr/bin/env python3
"""
calibrate.py

Load a single image, interactively draw ROIs for Clash Royale analysis,
and save the coordinates to a JSON file.

Usage:
    python calibrate.py --image path/to/screenshot.png [--output rois.json]

You will be prompted to draw boxes for:
  1) Left tower HP bar
  2) Right tower HP bar
  3) Elixir meter

After drawing each, press ENTER or SPACE. The resulting ROIs are saved as {name: [x,y,w,h]}.
"""
import cv2
import json
import argparse
import os

ROI_NAMES = ["hp_left", "hp_right", "king", "enemy_hp_left", "enemy_hp_right", "enemy_king", "elixir"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate ROIs for Clash Royale capture/extraction"
    )
    parser.add_argument("--image", "-i", required=True,
                        help="Path to a sample screenshot image")
    parser.add_argument("--output", "-o", default="rois.json",
                        help="Output JSON file for ROI coordinates")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image file not found: {args.image}")
        return

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Failed to load image: {args.image}")
        return

    rois = {}
    print("\n=== ROI Calibration ===")
    print("Draw ROI with your mouse, then press ENTER or SPACE to confirm, ESC to skip.")

    for name in ROI_NAMES:
        win = f"Draw ROI: {name}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        x, y, w, h = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)

        # If user pressed ESC, selectROI returns (0,0,0,0)
        if w == 0 or h == 0:
            print(f"[WARNING] No ROI drawn for '{name}', skipping.")
            rois[name] = [0,0,0,0]
        else:
            print(f"{name}: x={x}, y={y}, w={w}, h={h}")
            rois[name] = [int(x), int(y), int(w), int(h)]

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(rois, f, indent=2)
    print(f"\nROIs saved to {args.output}")


if __name__ == '__main__':
    main()
