#!/usr/bin/env python3

import cv2 # OpenCV for image processing
import os
import sys
import time
# import capture_util # No longer needed for this development version

# --- Configuration ---
# Define ROIs as percentages (x_start, y_start, width, height) of the window
# These are ESTIMATES based on the provided screenshot and may need tuning!
# Format: (left_edge_pct, top_edge_pct, width_pct, height_pct)
ROI_DEFINITIONS_PCT = {
    "timer":            (0.83, 0.065, 0.12, 0.04),  # Moved down/right slightly, narrowed
    "elixir_bar":       (0.28, 0.905, 0.44, 0.03),  # Defined more accurately, smaller height
    "cards_area":       (0.16, 0.81, 0.68, 0.09),   # Reduced height slightly
    "card_1":           (0.175, 0.815, 0.15, 0.075), # Shifted right, narrowed, reduced height
    "card_2":           (0.345, 0.815, 0.15, 0.075), # Shifted right, narrowed, reduced height
    "card_3":           (0.515, 0.815, 0.15, 0.075), # Shifted right, narrowed, reduced height
    "card_4":           (0.685, 0.815, 0.15, 0.075), # Shifted right, narrowed, reduced height
    "next_card":        (0.035, 0.835, 0.09, 0.06),  # Shifted right/down, narrowed, reduced height
    # --- Health Bars (Approximate - focus on the bar itself) ---
    # Player Towers
    "player_hp_left":   (0.18, 0.67, 0.12, 0.015),  # Moved up
    "player_hp_king":   (0.43, 0.75, 0.14, 0.015),  # Moved up
    "player_hp_right":  (0.70, 0.67, 0.12, 0.015),  # Moved up
    # Opponent Towers
    "opp_hp_left":      (0.18, 0.18, 0.12, 0.015),  # Moved up
    "opp_hp_king":      (0.43, 0.10, 0.14, 0.015),  # Moved up
    "opp_hp_right":     (0.70, 0.18, 0.12, 0.015),  # Moved up
    # --- Full Arena (for CNN input, excluding hand/top bar maybe?) ---
    "arena_full":       (0.02, 0.05, 0.96, 0.75), # Kept same for now
    "arena_playable":   (0.03, 0.25, 0.94, 0.55), # Kept same for now
}

# Drawing Configuration
RECT_COLOR = (0, 255, 0) # Green in BGR format
RECT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255) # White
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.4
TEXT_THICKNESS = 1

# --- Development Configuration ---
# Specify the exact image file to load for development
DEV_IMAGE_FILENAME = "bluestacks_capture_1743696849.png"
# --- End Development Configuration ---


def calculate_absolute_rois(definitions_pct, image_width, image_height):
    """
    Converts percentage-based ROI definitions to absolute pixel coordinates.

    Args:
        definitions_pct (dict): Dictionary mapping ROI names to percentage tuples.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        dict: Dictionary mapping ROI names to absolute coordinate tuples (x, y, w, h).
    """
    absolute_rois = {}
    for name, (x_pct, y_pct, w_pct, h_pct) in definitions_pct.items():
        x = int(image_width * x_pct)
        y = int(image_height * y_pct)
        w = int(image_width * w_pct)
        h = int(image_height * h_pct)
        # Basic validation to ensure width/height are positive
        if w <= 0: w = 1
        if h <= 0: h = 1
        absolute_rois[name] = (x, y, w, h)
    return absolute_rois

def draw_rois_on_image(image, rois):
    """
    Draws rectangles and labels for defined ROIs on a copy of the image.

    Args:
        image (numpy.ndarray): The input image (loaded by OpenCV).
        rois (dict): Dictionary mapping ROI names to absolute coordinate tuples (x, y, w, h).

    Returns:
        numpy.ndarray: A new image with ROIs drawn on it.
    """
    output_image = image.copy() # Work on a copy

    for name, (x, y, w, h) in rois.items():
        # Define top-left and bottom-right points for the rectangle
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        # Draw the rectangle
        cv2.rectangle(output_image, pt1, pt2, RECT_COLOR, RECT_THICKNESS)
        # Add a label above the rectangle
        label_pos = (x, y - 5 if y > 10 else y + 10) # Adjust label position if box is near top
        cv2.putText(output_image, name, label_pos, TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

    return output_image

# --- Main Execution (Development Mode) ---
if __name__ == "__main__":
    print("Starting Game Analyzer in Development Mode...")

    # 1. Define the input image path
    input_filepath = os.path.join(os.getcwd(), DEV_IMAGE_FILENAME)

    # Check if the development image exists
    if not os.path.exists(input_filepath):
        print(f"Error: Development image not found at '{input_filepath}'", file=sys.stderr)
        print("Please ensure the file exists in the same directory as the script.", file=sys.stderr)
        sys.exit(1)

    # 2. Load the specified image with OpenCV
    print(f"Loading development image: {input_filepath}")
    img = cv2.imread(input_filepath)

    if img is None:
        print(f"Error: Failed to load image '{input_filepath}'. Check file path and integrity.", file=sys.stderr)
        sys.exit(1)

    img_height, img_width, _ = img.shape
    print(f"Image dimensions: Width={img_width}, Height={img_height}")

    # 3. Calculate Absolute ROI coordinates based on percentages
    print("Calculating absolute ROI coordinates...")
    absolute_rois = calculate_absolute_rois(ROI_DEFINITIONS_PCT, img_width, img_height)

    # 4. Draw the ROIs on the image
    print("Drawing ROIs on the image...")
    img_with_rois = draw_rois_on_image(img, absolute_rois)

    # 5. Save the image with ROIs drawn
    # Create a distinct output name for the analyzed development image
    output_filename = f"analyzed_{DEV_IMAGE_FILENAME}"
    output_filepath = os.path.join(os.getcwd(), output_filename)
    print(f"Saving analyzed image to: {output_filepath}")
    save_success = cv2.imwrite(output_filepath, img_with_rois)

    if not save_success:
         print(f"Error: Failed to save analyzed image to '{output_filepath}'.", file=sys.stderr)
    else:
         print("Analysis complete. Check the saved image.")

    # Optional: Display the image in a window
    # cv2.imshow("Analyzed Game Screen", img_with_rois)
    # print("Press any key to close the image window...")
    # cv2.waitKey(0) # Wait indefinitely for a key press
    # cv2.destroyAllWindows()
