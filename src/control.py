"""Mouse and keyboard control functions."""
import pyautogui
import time


def click_at(x, y, delay=0.5):
    """
    Click at the specified coordinates.
    
    Args:
        x (int): X coordinate
        y (int): Y coordinate
        delay (float): Delay in seconds after clicking
    """
    pyautogui.click(x, y)
    time.sleep(delay)
