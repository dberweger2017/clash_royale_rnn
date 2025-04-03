# test.py
import subprocess
import re
import os
import time
import sys

# --- Configuration ---
APP_NAME = "BlueStacks"
WINDOW_TITLE = "BlueStacks Air"
# --- End Configuration ---

def get_window_geometry(app_name, window_title):
    """
    Uses AppleScript to get the geometry {position, size} of a specific window.

    Returns:
        tuple: (x, y, width, height) or None if not found/error.
    """
    # Use .format() and double the literal braces needed for AppleScript.
    applescript_template = """
    tell application "System Events"
        try
            tell process "{0}" -- Placeholder for app_name
                set target_window to first window whose name is "{1}" -- Placeholder for window_title
                if not (exists target_window) then
                    error "Window '{1}' not found for process '{0}'."
                end if
                -- Escape AppleScript's braces by doubling them for Python's .format()
                set geom to {{ {{position of target_window}}, {{size of target_window}} }}
                return geom
            end tell
        on error errMsg number errNum
            return "Error: " & errMsg
        end try
    end tell
    """
    applescript = applescript_template.format(app_name, window_title) # Insert variables here

    try:
        # Debug: Print the exact AppleScript being executed
        # print("--- Executing AppleScript ---")
        # print(applescript)
        # print("---------------------------")

        result = subprocess.run(
            ["osascript", "-e", applescript],
            capture_output=True,
            text=True,
            check=True,  # Raise exception on non-zero exit code
        )
        output = result.stdout.strip()

        if output.startswith("Error:"):
             print(f"AppleScript Error: {output}", file=sys.stderr)
             return None

        match = re.match(r"(\d+),\s*(\d+),\s*(\d+),\s*(\d+)", output)

        if match:
            x, y, w, h = map(int, match.groups())
            print(f"Found window '{window_title}' geometry: x={x}, y={y}, w={w}, h={h}")
            return x, y, w, h
        else:
            print(f"Error: Could not parse geometry output: '{output}'", file=sys.stderr)
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error running AppleScript to get geometry:", file=sys.stderr)
        print(f"Command: osascript -e '...' (Check script content/permissions)", file=sys.stderr)
        print(f"Return Code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        if "Application isn't running" in e.stderr or "Application isn't running" in e.stdout:
             print(f"'{app_name}' process might not be running.", file=sys.stderr)
        elif "access" in e.stderr or "access" in e.stdout or "assistive devices" in e.stderr:
             print("Hint: Check Accessibility permissions for Terminal/Python/IDE in System Settings > Privacy & Security.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred getting geometry: {e}", file=sys.stderr)
        return None

def capture_window_region(x, y, width, height, output_path):
    """
    Captures a specific screen region using screencapture.

    Returns:
        bool: True if successful, False otherwise.
    """
    capture_command = [
        "screencapture",
        "-R", f"{x},{y},{width},{height}", # Region flag
        "-o",  # Omit window shadow
        "-x",  # No sound
        output_path,
    ]

    print(f"Executing: {' '.join(capture_command)}")

    try:
        subprocess.run(capture_command, check=True)
        print(f"Screenshot saved successfully to: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running screencapture:", file=sys.stderr)
        print(f"Command: {' '.join(capture_command)}", file=sys.stderr)
        print(f"Return Code: {e.returncode}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        if "permissions" in e.stderr or "permissions" in e.stdout:
             print("Hint: Check Screen Recording permissions for Terminal/Python/IDE in System Settings > Privacy & Security.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during capture: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    print(f"Attempting to find window '{WINDOW_TITLE}' for application '{APP_NAME}'...")

    # --- Important Permission Note ---
    print("\n---")
    print("NOTE: This script requires permissions!")
    print("1. Accessibility: To get window position/size.")
    print("2. Screen Recording: To take the screenshot.")
    print("Go to System Settings > Privacy & Security to grant access to Terminal (or your IDE).")
    print("---\n")
    # ---

    geometry = get_window_geometry(APP_NAME, WINDOW_TITLE)

    if geometry:
        x, y, w, h = geometry
        timestamp = int(time.time())
        # Save in the current directory
        output_filename = f"bluestacks_capture_{timestamp}.png"
        output_filepath = os.path.join(os.getcwd(), output_filename)

        capture_success = capture_window_region(x, y, w, h, output_filepath)
        if not capture_success:
             sys.exit(1) # Exit with error if capture failed

    else:
        print(f"Could not capture screenshot because window geometry for '{WINDOW_TITLE}' was not found.")
        sys.exit(1) # Exit with error code