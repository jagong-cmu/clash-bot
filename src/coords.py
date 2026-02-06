"""Window coordinate detection and management functions."""
import subprocess


def get_window_coordinates(window_name):
    """
    Automatically detect the coordinates and size of a window by its name.
    
    Args:
        window_name (str): Name or partial name of the window to find (case-insensitive)
    
    Returns:
        tuple: (x, y, width, height) if window found, None otherwise
        x, y: Top-left corner coordinates
        width, height: Window dimensions
    """
    # AppleScript to find window by name and get its bounds
    # Note: {{}} escapes curly braces for literal use in AppleScript
    applescript = '''
    tell application "System Events"
        set windowList to {{}}
        repeat with proc in processes
            try
                set windowList to windowList & (windows of proc whose name contains "{}")
            end try
        end repeat
        
        if (count of windowList) > 0 then
            set targetWindow to item 1 of windowList
            set windowPosition to position of targetWindow
            set windowSize to size of targetWindow
            return (item 1 of windowPosition) & "," & (item 2 of windowPosition) & "," & (item 1 of windowSize) & "," & (item 2 of windowSize)
        else
            return "NOT_FOUND"
        end if
    end tell
    '''.format(window_name)
    
    try:
        # Run AppleScript
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            check=True
        )
        
        output = result.stdout.strip()
        
        if output == "NOT_FOUND":
            print(f"Window '{window_name}' not found. Make sure the window is open and the name is correct.")
            return None
        
        # Parse the result: "x,y,width,height"
        coords = [int(x) for x in output.split(',')]
        x, y, width, height = coords
        
        print(f"Found window '{window_name}' at ({x}, {y}) with size {width}x{height}")
        return (x, y, width, height)
        
    except subprocess.CalledProcessError as e:
        print(f"Error finding window: {e}")
        return None
    except ValueError as e:
        print(f"Error parsing window coordinates: {e}")
        return None
