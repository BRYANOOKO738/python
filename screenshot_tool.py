import time
import os
from datetime import datetime
import pyautogui

# Screenshot directory
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def take_screenshot():
    """Take a screenshot and save it with a timestamp filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SCREENSHOT_DIR}/screenshot_{timestamp}.png"
    
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    
    print(f"Screenshot saved as {filename}")

def main():
    print("Screenshot tool starting...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            take_screenshot()
            time.sleep(30)  # Wait 30 seconds
    except KeyboardInterrupt:
        print("Program stopped by user")

if __name__ == "__main__":
    main()