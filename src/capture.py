"""Screen capture and image detection functions."""
import pyautogui
import cv2
import numpy as np


def capture_screen_region(x, y, width, height):
    """
    Capture a specific region of the screen.
    
    Args:
        x (int): Top-left x coordinate
        y (int): Top-left y coordinate
        width (int): Width of the region
        height (int): Height of the region
    
    Returns:
        numpy.ndarray: OpenCV BGR image of the captured region
    """
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    # Convert to OpenCV format
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)


def find_image_on_screen(template_path, screenshot, threshold=0.8):
    """
    Find a template image within a screenshot.
    
    Args:
        template_path (str): Path to the template image file
        screenshot (numpy.ndarray): Screenshot image to search in
        threshold (float): Confidence threshold (0.0 to 1.0)
    
    Returns:
        tuple: (center_x, center_y) if found, None otherwise
    """
    template = cv2.imread(template_path)
    if template is None:
        print(f"Warning: Could not load template image from {template_path}")
        return None
    
    result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    if max_val >= threshold:
        # Calculate center point
        h, w = template.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2
        return (center_x, center_y)
    return None
