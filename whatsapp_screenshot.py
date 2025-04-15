import time
import os
from datetime import datetime
import pyautogui
import pywhatkit

# WhatsApp configuration
PHONE_NUMBER = "+254748951330"  # Include country code (e.g., +1 for US)

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
    return filename

def send_whatsapp(screenshot_path):
    """Send the screenshot to WhatsApp."""
    try:
        # Get current time + 1 minute (pywhatkit needs future time)
        now = datetime.now()
        hour = now.hour
        minute = now.minute + 1
        
        # Adjust hour if minute rolls over
        if minute >= 60:
            minute -= 60
            hour += 1
        if hour >= 24:
            hour -= 24
        
        # Send image via WhatsApp
        pywhatkit.sendwhats_image(
            PHONE_NUMBER,
            screenshot_path,
            "Automated screenshot",
            wait_time=15  # seconds to wait before sending
        )
        
        print(f"WhatsApp message sent with attachment: {screenshot_path}")
        
    except Exception as e:
        print(f"Failed to send WhatsApp message: {str(e)}")

def screenshot_and_send():
    """Take a screenshot and send it via WhatsApp."""
    screenshot_path = take_screenshot()
    send_whatsapp(screenshot_path)

def main():
    print("WhatsApp Screenshot Sender starting...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            screenshot_and_send()
            time.sleep(30)  # Wait 30 seconds before next screenshot
    except KeyboardInterrupt:
        print("Program stopped by user")

if __name__ == "__main__":
    main()