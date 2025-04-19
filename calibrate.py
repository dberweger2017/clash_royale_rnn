"""
calibrate.py

Interactive ROI calibration and optional visualization for Clash Royale screen analysis.

Usage:
    # Calibrate and save ROIs:
    python calibrate.py --image path/to/screenshot.png --output rois.json

    # Visualize existing ROIs on the image:
    python calibrate.py --image path/to/screenshot.png --output rois.json --visualize --labeled-output labeled.png
"""
import cv2
import json
import argparse
import os

ROI_NAMES = ["hp_left", "hp_right", "king", "enemy_hp_left", "enemy_hp_right", "enemy_king", "elixir"]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate or visualize ROIs for Clash Royale extractor"
    )
    parser.add_argument(
        "--image", "-i", required=True,
        help="Path to a sample screenshot image"
    )
    parser.add_argument(
        "--output", "-o", default="rois.json",
        help="JSON file to load/save ROI coordinates"
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Load ROIs from JSON and draw boundaries on the image"
    )
    parser.add_argument(
        "--labeled-output", "-l", default="labeled.png",
        help="Output path for the labeled image when using --visualize"
    )
    return parser.parse_args()


def draw_rois_on_image(img, rois):
    """Draw rectangles and labels for each ROI on the image."""
    for name, coords in rois.items():
        x, y, w, h = coords
        # choose a distinct color per ROI type
        if name.startswith("hp_"):
            color = (0, 255, 0)      # green for health bars
        else:
            color = (255, 0, 255)    # magenta for elixir
        # draw rectangle and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            img, name, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
        )
    return img


def main():
    args = parse_args()
    if not os.path.isfile(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        return

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Failed to load image: {args.image}")
        return

    # Visualization mode: draw existing ROIs
    if args.visualize:
        if not os.path.exists(args.output):
            print(f"[ERROR] ROI JSON not found: {args.output}")
            return
        with open(args.output) as f:
            rois = json.load(f)
        labeled = draw_rois_on_image(img.copy(), rois)
        cv2.imwrite(args.labeled_output, labeled)
        print(f"Labeled image saved to: {args.labeled_output}")
        cv2.imshow("Labeled ROIs", labeled)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Calibration mode: interactively draw ROIs
    rois = {}
    print("\n=== ROI Calibration ===")
    print("Draw each ROI with the mouse, then press ENTER or SPACE. ESC skips.")
    for name in ROI_NAMES:
        window_name = f"Draw ROI: {name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        x, y, w, h = cv2.selectROI(
            window_name, img,
            showCrosshair=True, fromCenter=False
        )
        cv2.destroyWindow(window_name)

        if w == 0 or h == 0:
            print(f"[WARNING] No ROI drawn for '{name}', defaulting to (0,0,0,0)")
            rois[name] = [0, 0, 0, 0]
        else:
            rois[name] = [int(x), int(y), int(w), int(h)]
            print(f"{name} ROI -> x={x}, y={y}, w={w}, h={h}")

    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(rois, f, indent=2)
    print(f"ROIs saved to: {args.output}")


if __name__ == '__main__':
    main()