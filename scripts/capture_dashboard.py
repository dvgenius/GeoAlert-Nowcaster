"""
Captures high-resolution screenshot of the GeoAlert Nowcaster dashboard.
"""

import os
import time
from playwright.sync_api import sync_playwright

ARTIFACT_DIR = r"C:\Users\Lenovo\.gemini\antigravity-ide\brain\e228a724-07af-4917-baac-db9ff26a6f55"
SCREENSHOT_PATH = os.path.join(ARTIFACT_DIR, "dashboard_preview.png")

def capture():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        
        print("Navigating to http://localhost:5173/...")
        page.goto("http://localhost:5173/", wait_until="networkidle")
        
        # Wait 3 seconds for map tiles and GeoJSON layers to settle
        time.sleep(3)
        
        # Save screenshot
        page.screenshot(path=SCREENSHOT_PATH, full_page=True)
        print(f"Screenshot successfully saved to: {SCREENSHOT_PATH}")
        browser.close()

if __name__ == "__main__":
    capture()
