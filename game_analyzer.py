#!/usr/bin/env python3

import cv2 # OpenCV for image processing
import os
import sys
import time
import numpy as np
import pytesseract # <-- Import pytesseract
import re # <-- Import regex for timer parsing
# import capture_util # Keep commented for development mode

# --- Configuration ---
# Define ROIs as percentages (Using your final tuned values)
ROI_DEFINITIONS_PCT = {
    "timer":            (0.785, 0.059, 0.13, 0.035),
    "elixir_bar":       (0.248, 0.953, 0.08, 0.03), # Targets the number
    "card_1":           (0.21, 0.837, 0.163, 0.12),
    "card_2":           (0.383, 0.837, 0.163, 0.12),
    "card_3":           (0.558, 0.837, 0.163, 0.12),
    "card_4":           (0.734, 0.837, 0.163, 0.12),
    "next_card":        (0.048, 0.935, 0.076, 0.056),
    "player_hp_left":   (0.182, 0.635, 0.078, 0.017),
    "player_hp_king":   (0.43, 0.75, 0.14, 0.015),
    "player_hp_right":  (0.67, 0.635, 0.078, 0.017),
    "opp_hp_left":      (0.185, 0.167, 0.078, 0.017),
    "opp_hp_king":      (0.43, 0.10, 0.14, 0.015),
    "opp_hp_right":     (0.673, 0.167, 0.078, 0.017),
    "arena_playable":   (0.064, 0.112, 0.8, 0.668)
}

# --- Card Info (Placeholder - Replace with actual data) ---
CARD_INFO = {
    "Arrows": {"cost": 3},
    "Archers": {"cost": 3},
    "Giant": {"cost": 5},
    "Musketeer": {"cost": 4},
    "Knight": {"cost": 3},
    "Unknown": {"cost": -1},
    "Placeholder": {"cost": 0}
}
# --- Card Templates (Placeholder - Load actual images later) ---
CARD_TEMPLATES = {}

# Drawing Configuration
RECT_COLOR = (0, 255, 0)
RECT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.4
TEXT_THICKNESS = 1

# --- Development Configuration ---
DEV_IMAGE_FILENAME = "bluestacks_capture_1743696849.png"
# --- End Development Configuration ---

# --- Helper Functions ---
def calculate_absolute_rois(definitions_pct, image_width, image_height):
    """Converts percentage-based ROI definitions to absolute pixel coordinates."""
    absolute_rois = {}
    for name, (x_pct, y_pct, w_pct, h_pct) in definitions_pct.items():
        x = int(image_width * x_pct)
        y = int(image_height * y_pct)
        w = int(image_width * w_pct)
        h = int(image_height * h_pct)
        if w <= 0: w = 1
        if h <= 0: h = 1
        absolute_rois[name] = (x, y, w, h)
    return absolute_rois

def draw_rois_on_image(image, rois):
    """Draws rectangles and labels for defined ROIs on a copy of the image."""
    output_image = image.copy()
    for name, (x, y, w, h) in rois.items():
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        cv2.rectangle(output_image, pt1, pt2, RECT_COLOR, RECT_THICKNESS)
        label_pos = (x, y - 5 if y > 10 else y + 10)
        cv2.putText(output_image, name, label_pos, TEXT_FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    return output_image

def crop_roi(image, roi_coords):
    """Crops the image based on ROI coordinates (x, y, w, h)."""
    x, y, w, h = roi_coords
    # Add basic bounds checking
    h_img, w_img = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    if y1 >= y2 or x1 >= x2:
        print(f"Warning: Invalid crop coordinates for ROI {roi_coords} on image size {image.shape}", file=sys.stderr)
        # Return a small black image or handle error appropriately
        return np.zeros((1, 1, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2]

# --- Extraction Functions ---

def extract_health_percentage(image_crop, is_player):
    """Estimates health percentage from a cropped health bar image."""
    # --- Placeholder Logic ---
    # TODO: Implement actual color detection and pixel counting
    return 0.5 # Return dummy value for now

def extract_elixir_ocr(image_crop):
    """
    Extracts the elixir number using OCR from the cropped image.
    Args:
        image_crop: The cropped numpy array of the elixir number ROI.
    Returns:
        int: The extracted elixir value (0-10), or -1 on error.
    """
    if image_crop is None or image_crop.shape[0] < 5 or image_crop.shape[1] < 5:
        print("Error: Invalid image crop for Elixir OCR.", file=sys.stderr)
        return -1

    try:
        # Preprocessing
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        # Enlarge image slightly - can sometimes help OCR with small text
        scale_factor = 2
        width = int(gray.shape[1] * scale_factor)
        height = int(gray.shape[0] * scale_factor)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # Thresholding (Otsu often works well, might need tuning)
        # The elixir number seems dark on a light background in the purple drop
        _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Optional: Save preprocessed image for debugging
        # cv2.imwrite("debug_elixir_thresh.png", thresh)

        # Tesseract Configuration
        # --psm 7: Treat the image as a single text line.
        # --psm 10: Treat the image as a single character. (Good for single digits)
        # --psm 6: Assume a single uniform block of text.
        # Whitelist digits 0-9
        config = "--psm 10 -c tessedit_char_whitelist=0123456789"

        text = pytesseract.image_to_string(thresh, config=config).strip()

        # Basic validation
        if text.isdigit():
            value = int(text)
            if 0 <= value <= 10:
                return value
            else:
                print(f"Warning: Elixir OCR returned out-of-range value: {text}", file=sys.stderr)
                return -1 # Value out of expected range
        else:
            # Handle cases like empty string or non-digit characters
            if text: # Log if it found something non-numeric
                 print(f"Warning: Elixir OCR returned non-digit text: '{text}'", file=sys.stderr)
            return -1

    except Exception as e:
        print(f"Error during Elixir OCR: {e}", file=sys.stderr)
        return -1

def extract_timer_ocr(image_crop):
    """
    Extracts the timer value (M:SS) using OCR and converts to seconds.
    Args:
        image_crop: The cropped numpy array of the timer ROI.
    Returns:
        int: Total seconds remaining, or -1 on error.
    """
    if image_crop is None or image_crop.shape[0] < 5 or image_crop.shape[1] < 5:
        print("Error: Invalid image crop for Timer OCR.", file=sys.stderr)
        return -1

    try:
        # Preprocessing
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        # Enlarge image slightly
        scale_factor = 2
        width = int(gray.shape[1] * scale_factor)
        height = int(gray.shape[0] * scale_factor)
        resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)

        # Thresholding (Timer text is white/light on dark background)
        # Simple binary threshold might work, adjust threshold value (e.g., 150-200)
        _, thresh = cv2.threshold(resized, 180, 255, cv2.THRESH_BINARY)

        # Optional: Save preprocessed image for debugging
        # cv2.imwrite("debug_timer_thresh.png", thresh)

        # Tesseract Configuration
        # --psm 7: Treat the image as a single text line.
        # Whitelist digits 0-9 and colon
        config = "--psm 7 -c tessedit_char_whitelist=0123456789:"

        text = pytesseract.image_to_string(thresh, config=config).strip()

        # Parsing M:SS format
        match = re.match(r"(\d):(\d{2})", text)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            # Basic validation for plausible game time
            if 0 <= minutes <= 5 and 0 <= seconds <= 59:
                 return minutes * 60 + seconds
            else:
                 print(f"Warning: Timer OCR parsed implausible time: {text}", file=sys.stderr)
                 return -1
        else:
            if text: # Log if it found something non-matching
                print(f"Warning: Timer OCR returned non-matching text: '{text}'", file=sys.stderr)
            return -1 # Parsing failed

    except Exception as e:
        print(f"Error during Timer OCR: {e}", file=sys.stderr)
        return -1

def identify_card_template(image_crop, templates):
    """Identifies a card from a crop using template matching."""
    # --- Placeholder Logic ---
    # TODO: Implement template matching
    return "Placeholder" # Return dummy value

def get_arena_image(image, roi_coords, target_size=None):
    """Extracts and optionally resizes the main arena image."""
    arena_crop = crop_roi(image, roi_coords)
    if target_size and arena_crop.size > 0: # Check if crop is valid
        arena_crop = cv2.resize(arena_crop, target_size, interpolation=cv2.INTER_AREA)
    elif target_size and arena_crop.size == 0:
        print("Warning: Cannot resize empty arena crop.", file=sys.stderr)
        return None # Or return a default blank image of target_size
    return arena_crop

# --- Main Execution (Refactored) ---
if __name__ == "__main__":
    print("Starting Game Analyzer (Refactored) in Development Mode...")

    # 1. Define the input image path
    input_filepath = os.path.join(os.getcwd(), DEV_IMAGE_FILENAME)
    if not os.path.exists(input_filepath):
        print(f"Error: Development image not found at '{input_filepath}'", file=sys.stderr)
        sys.exit(1)

    # 2. Load the specified image
    print(f"Loading development image: {input_filepath}")
    img = cv2.imread(input_filepath)
    if img is None:
        print(f"Error: Failed to load image '{input_filepath}'.", file=sys.stderr)
        sys.exit(1)
    img_height, img_width, _ = img.shape
    print(f"Image dimensions: Width={img_width}, Height={img_height}")

    # 3. Calculate Absolute ROI coordinates
    print("Calculating absolute ROI coordinates...")
    absolute_rois = calculate_absolute_rois(ROI_DEFINITIONS_PCT, img_width, img_height)

    # 4. Initialize Game State Dictionary
    game_state = {'timestamp': time.time()}

    # 5. Extract Information using defined functions
    print("Extracting game state information...")

    # --- Health ---
    hp_keys = [k for k in absolute_rois if '_hp_' in k]
    for key in hp_keys:
        crop = crop_roi(img, absolute_rois[key])
        is_player = key.startswith('player_')
        game_state[key] = extract_health_percentage(crop, is_player) # Still placeholder

    # --- Elixir ---
    if "elixir_bar" in absolute_rois:
        elixir_crop = crop_roi(img, absolute_rois["elixir_bar"])
        game_state['elixir'] = extract_elixir_ocr(elixir_crop) # <-- Now implemented
    else:
        game_state['elixir'] = -1

    # --- Timer ---
    if "timer" in absolute_rois:
        timer_crop = crop_roi(img, absolute_rois["timer"])
        game_state['time_seconds'] = extract_timer_ocr(timer_crop) # <-- Now implemented
    else:
        game_state['time_seconds'] = -1

    # --- Cards ---
    card_keys = [k for k in absolute_rois if k.startswith('card_')]
    card_keys.append("next_card")
    game_state['cards'] = {}
    for key in card_keys:
        if key in absolute_rois:
            card_crop = crop_roi(img, absolute_rois[key])
            card_name = identify_card_template(card_crop, CARD_TEMPLATES) # Still placeholder
            card_cost = CARD_INFO.get(card_name, {"cost": -1})["cost"]
            game_state['cards'][key] = {'name': card_name, 'cost': card_cost}
        else:
             game_state['cards'][key] = {'name': 'Unknown', 'cost': -1}

    # --- Arena Image ---
    if "arena_full" in absolute_rois:
        target_cnn_size = (224, 224)
        arena_image_data = get_arena_image(img, absolute_rois["arena_full"], target_size=target_cnn_size)
        game_state['arena_image'] = arena_image_data
        print(f"Arena image extracted and resized to: {arena_image_data.shape if arena_image_data is not None else 'None'}")
    else:
        game_state['arena_image'] = None

    # 6. Print the extracted game state
    print("\n--- Extracted Game State ---")
    for key, value in game_state.items():
        if key == 'arena_image':
            print(f"  {key}: Numpy array of shape {value.shape if value is not None else 'None'}")
        else:
            print(f"  {key}: {value}")
    print("---------------------------\n")

    # 7. Draw ROIs for visualization (Optional)
    print("Drawing ROIs on the image for visualization...")
    img_with_rois = draw_rois_on_image(img, absolute_rois)

    # 8. Save the visualized image
    output_filename = f"analyzed_{DEV_IMAGE_FILENAME}"
    output_filepath = os.path.join(os.getcwd(), output_filename)
    print(f"Saving analyzed image to: {output_filepath}")
    save_success = cv2.imwrite(output_filepath, img_with_rois)

    if not save_success:
         print(f"Error: Failed to save analyzed image to '{output_filepath}'.", file=sys.stderr)
    else:
         print("Analysis complete. Check the saved image and printed game state.")

    # Optional: Display the image
    # cv2.imshow("Analyzed Game Screen", img_with_rois)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()