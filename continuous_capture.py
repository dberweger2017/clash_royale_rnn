import os
import time
import sys
import capture_util # Assuming capture_util.py is in the same directory

# --- Configuration ---
OUTPUT_FOLDER = "img"
DELAY_SECONDS = 1.0 # Interval between screenshots
# Use the same constants as capture_util for consistency
APP_NAME = capture_util.APP_NAME
WINDOW_TITLE = capture_util.WINDOW_TITLE
# --- End Configuration ---

def ensure_dir(folder_path):
    """Creates the directory if it doesn't exist."""
    try:
        os.makedirs(folder_path, exist_ok=True)
        # print(f"Output directory '{folder_path}' ensured.") # Optional
    except OSError as e:
        print(f"Error creating directory '{folder_path}': {e}", file=sys.stderr)
        sys.exit(1) # Exit if we can't create the folder

def main():
    print("--- Continuous Screenshot Utility ---")
    print(f"Target App: {APP_NAME}")
    print(f"Target Window: {WINDOW_TITLE}")
    print(f"Saving images to: {OUTPUT_FOLDER}/")
    print(f"Interval: {DELAY_SECONDS} seconds")
    print("Press Ctrl+C to stop.")
    print("\nNOTE: Ensure Accessibility and Screen Recording permissions are granted.")

    # Create full path for the output folder relative to the script location
    output_path_full = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    ensure_dir(output_path_full)

    screenshot_count = 0
    try:
        while True:
            # print(f"\n[{time.strftime('%H:%M:%S')}] Searching for window...") # Optional verbose
            geometry = capture_util.get_window_geometry(APP_NAME, WINDOW_TITLE)

            if geometry:
                x, y, w, h = geometry
                timestamp = int(time.time())
                filename = f"screenshot_{timestamp}.png"
                filepath = os.path.join(output_path_full, filename)

                # print(f"[{time.strftime('%H:%M:%S')}] Capturing...") # Optional verbose
                # Use the capture function from capture_util
                # Note: capture_window_region already prints messages
                success = capture_util.capture_window_region(x, y, w, h, filepath)

                if success:
                    screenshot_count += 1
                    # print(f"Screenshot {screenshot_count} saved.") # Optional verbose
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Capture failed.", file=sys.stderr)
                    # Optional: Add a small extra delay on failure?
                    # time.sleep(1)

            else:
                print(f"[{time.strftime('%H:%M:%S')}] Window '{WINDOW_TITLE}' not found. Retrying...")
                # No screenshot taken, just wait for the next cycle

            # Wait for the specified delay before the next attempt
            time.sleep(DELAY_SECONDS)

    except KeyboardInterrupt:
        print("\n--- Stopping Screenshot Utility ---")
        print(f"Captured {screenshot_count} screenshots in '{OUTPUT_FOLDER}'.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
    finally:
        print("Exiting.")


if __name__ == "__main__":
    main()