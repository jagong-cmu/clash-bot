"""Main bot loop and orchestration."""
import sys
import os
import pyautogui
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.coords import get_window_coordinates
from src.capture import capture_screen_region, find_image_on_screen
from src.control import click_at

# Disable pyautogui's failsafe (move mouse to corner to stop)
# Keep this enabled while testing!
pyautogui.FAILSAFE = True

# Main bot loop
def main():
    print("Starting bot in 3 seconds... Move mouse to corner to stop!")
    time.sleep(3)
    
    # Automatically detect game window (adjust the window name as needed)
    # Common names: "Clash Royale", "BlueStacks", "NoxPlayer", etc.
    window_coords = get_window_coordinates("Clash Royale")  # Change this to your game window name
    
    if window_coords:
        game_x, game_y, game_width, game_height = window_coords
    else:
        # Fallback to manual coordinates if window not found
        print("Using manual coordinates as fallback...")
        game_x, game_y = 100, 100  # Top-left corner of game window
        game_width, game_height = 800, 600  # Size of game window
    
    while True:
        # Capture the game area
        screen = capture_screen_region(game_x, game_y, game_width, game_height)
        
        # Example: Look for a specific card image
        card_position = find_image_on_screen('card_template.png', screen, threshold=0.8)
        
        if card_position:
            # Adjust coordinates to absolute screen position
            abs_x = game_x + card_position[0]
            abs_y = game_y + card_position[1]
            
            print(f"Found card at {abs_x}, {abs_y}")
            click_at(abs_x, abs_y)
        
        time.sleep(0.1)  # Don't run too fast

if __name__ == "__main__":
    main()