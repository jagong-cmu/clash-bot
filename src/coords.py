"""Window coordinate detection and management functions."""
import subprocess


def list_open_windows():
    """
    List all open windows (process name and window title) for debugging.
    Useful to see exact names when get_window_coordinates() doesn't find a window.
    
    Returns:
        list of (process_name, window_name) or None on error
    """
    applescript = '''
    tell application "System Events"
        set out to ""
        repeat with proc in processes
            try
                set procName to name of proc
                repeat with w in (windows of proc)
                    try
                        set winName to name of w
                        set out to out & procName & "|" & winName & "\\n"
                    end try
                end repeat
            end try
        end repeat
        return out
    end tell
    '''
    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        lines = [ln.strip() for ln in result.stdout.strip().split('\n') if '|' in ln]
        pairs = []
        for ln in lines:
            parts = ln.split('|', 1)
            if len(parts) == 2:
                pairs.append((parts[0].strip(), parts[1].strip()))
        return pairs
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Could not list windows: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(e.stderr)
        return None


def get_window_coordinates(window_name):
    """
    Automatically detect the coordinates and size of a window by its name.
    Matches against both the application (process) name and the window title,
    case-insensitive.
    
    Args:
        window_name (str): Name or partial name of the app or window (case-insensitive)
    
    Returns:
        tuple: (x, y, width, height) if window found, None otherwise
        x, y: Top-left corner coordinates
        width, height: Window dimensions
    """
    search_lower = window_name.lower()
    # Get all windows in one pass (no slow "do shell script" per window)
    applescript = '''
    tell application "System Events"
        set out to ""
        repeat with proc in processes
            try
                set procName to name of proc
                repeat with w in (windows of proc)
                    try
                        set winName to name of w
                        set pos to position of w
                        set sz to size of w
                        set out to out & (procName as text) & "|" & (winName as text) & "|" & (item 1 of pos) & "," & (item 2 of pos) & "," & (item 1 of sz) & "," & (item 2 of sz) & (ASCII character 10)
                    end try
                end repeat
            end try
        end repeat
        return out
    end tell
    '''

    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        print("Window listing timed out. Try closing some apps or increase timeout in coords.py.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error finding window: {e}")
        if e.stderr:
            print(e.stderr)
        return None

    # Case-insensitive search in Python
    for line in output.split('\n'):
        if '|' not in line:
            continue
        parts = line.split('|', 2)  # procName | winName | x,y,w,h
        if len(parts) < 3:
            continue
        proc_name, win_name, coords_str = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if search_lower in proc_name.lower() or search_lower in win_name.lower():
            coords = [p.strip() for p in coords_str.split(',') if p.strip()]
            if len(coords) != 4:
                continue
            try:
                x, y, width, height = [int(c) for c in coords]
            except ValueError:
                continue
            print(f"Found window '{window_name}' at ({x}, {y}) with size {width}x{height}")
            return (x, y, width, height)

    print(f"Window '{window_name}' not found. Run list_open_windows() to see exact app/window names.")
    return None
