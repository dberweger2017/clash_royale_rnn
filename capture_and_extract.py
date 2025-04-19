# capture_and_extract.py

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
from PIL import ImageGrab
import capture_util  # your existing window‐grab helper

# Defaults
DEFAULT_DELAY     = 1.0
DEFAULT_ROIS_FILE = "rois.json"
OUTPUT_FOLDER     = "annotated"
WINDOW_NAME       = "Clash Royale Analysis"

def parse_args():
    p = argparse.ArgumentParser(
        description="Capture, extract and visualize Clash Royale stats (uses calibrated ROIs)."
    )
    p.add_argument(
        "-r", "--rois",
        default=DEFAULT_ROIS_FILE,
        help="Path to ROI JSON file (default: rois.json)"
    )
    p.add_argument(
        "-d", "--delay",
        type=float, default=DEFAULT_DELAY,
        help="Seconds between captures (default: 1.0)"
    )
    p.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save annotated frames to disk"
    )
    return p.parse_args()

def load_rois(path):
    if not os.path.exists(path):
        print(f"[ERROR] ROI file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)

def grab_frame():
    geom = capture_util.get_window_geometry(
        capture_util.APP_NAME, capture_util.WINDOW_TITLE
    )
    if not geom:
        return None
    x,y,w,h = geom
    img_pil = ImageGrab.grab((x, y, x+w, y+h))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def extract_game_stats(img, rois):
    stats = {}

    # health/king bars (green)
    def hp_pct(roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([50,150,150]), np.array([90,255,255]))
        cols = np.any(mask, axis=0)
        if not np.any(cols):
            return 0.0
        left  = np.argmax(cols)
        right = len(cols)-1-np.argmax(cols[::-1])
        return (right-left)/float(len(cols))

    for key in ("hp_left","hp_right","enemy_hp_left","enemy_hp_right","king","enemy_king"):
        if key in rois:
            x,y,w,h = rois[key]
            roi = img[y:y+h, x:x+w]
            stats[key] = hp_pct(roi)

    # elixir (magenta)
    if "elixir" in rois:
        x,y,w,h = rois["elixir"]
        roi = img[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([130,100,100]), np.array([160,255,255]))
        nlbl,_ = cv2.connectedComponents(mask)
        stats["elixir"] = max(0, nlbl-1)

    return stats

def annotate_image(img, stats, rois):
    vis = img.copy()
    # draw bars
    for key,val in stats.items():
        if key not in rois:
            continue
        x,y,w,h = rois[key]
        if key=="elixir":
            color=(255,0,255)
            cv2.rectangle(vis,(x,y),(x+w,y+h),color,2)
            cv2.putText(vis,f"Elixir:{val}",(x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
        else:
            pct=int(val*100)
            color=(0,255,0)
            cv2.rectangle(vis,(x,y),(x+w,y+h),color,2)
            cv2.rectangle(vis,(x,y),(x+int(val*w),y+h),color,-1)
            cv2.putText(vis,f"{key}:{pct}%",(x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    return vis

def main():
    args = parse_args()
    rois = load_rois(args.rois)

    if args.save:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 800,600)

    frame_count = 0
    try:
        while True:
            img = grab_frame()
            if img is None:
                time.sleep(args.delay)
                continue

            stats     = extract_game_stats(img, rois)
            annotated = annotate_image(img, stats, rois)
            combined  = np.hstack((annotated, img))
            cv2.imshow(WINDOW_NAME, combined)

            if args.save:
                out = os.path.join(OUTPUT_FOLDER, f"ann_{frame_count:04d}.png")
                cv2.imwrite(out, annotated)

            frame_count += 1
            if cv2.waitKey(int(args.delay*1000)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print(f"\nStopped after {frame_count} frames.")
    finally:
        cv2.destroyAllWindows()

if __name__=="__main__":
    main()
