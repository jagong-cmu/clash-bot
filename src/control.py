"""Mouse and keyboard control functions."""
import pyautogui
import time


def click_at(x, y, delay=0.5):
    """
    Click at the specified absolute screen coordinates.

    Args:
        x (int): X coordinate (screen)
        y (int): Y coordinate (screen)
        delay (float): Delay in seconds after clicking
    """
    pyautogui.click(x, y)
    time.sleep(delay)


def click_in_window(rel_x, rel_y, window_coords, delay=0.5):
    """
    Click at coordinates relative to a window's top-left corner.
    Use this with the (x, y, width, height) returned by get_window_coordinates().

    Args:
        rel_x (int): X offset from the window's left edge (pixels)
        rel_y (int): Y offset from the window's top edge (pixels)
        window_coords (tuple): (x, y, width, height) from get_window_coordinates()
        delay (float): Delay in seconds after clicking
    """
    wx, wy, _, _ = window_coords
    abs_x = wx + rel_x
    abs_y = wy + rel_y
    pyautogui.click(abs_x, abs_y)
    time.sleep(delay)


def click_at_relative(rel_x, rel_y, window_x, window_y, delay=0.5):
    """
    Click at coordinates relative to a known window origin (top-left).
    Use when you only have the window position, not full coords.

    Args:
        rel_x (int): X offset from the window's left edge (pixels)
        rel_y (int): Y offset from the window's top edge (pixels)
        window_x (int): Window top-left X (screen)
        window_y (int): Window top-left Y (screen)
        delay (float): Delay in seconds after clicking
    """
    pyautogui.click(window_x + rel_x, window_y + rel_y)
    time.sleep(delay)
