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
    # Escape double quotes for AppleScript
    search_escaped = search_lower.replace('\\', '\\\\').replace('"', '\\"')

    # Match by process name OR window name, case-insensitive
    # AppleScript has no "to lower"; use shell to get lowercase
    applescript = '''
    tell application "System Events"
        set searchStr to "{}"
        repeat with proc in processes
            try
                set procName to name of proc
                if (procName is not "") then
                    try
                        set procNameLower to do shell script "echo " & quoted form of (procName as text) & " | tr '[:upper:]' '[:lower:]'"
                        if procNameLower contains searchStr then
                            set wins to windows of proc
                            if (count of wins) > 0 then
                                set w to item 1 of wins
                                set pos to position of w
                                set sz to size of w
                                return (item 1 of pos) & "," & (item 2 of pos) & "," & (item 1 of sz) & "," & (item 2 of sz)
                            end if
                        end if
                    end try
                end if
                repeat with w in (windows of proc)
                    try
                        set winName to name of w
                        if (winName is not "") then
                            try
                                set winNameLower to do shell script "echo " & quoted form of (winName as text) & " | tr '[:upper:]' '[:lower:]'"
                                if winNameLower contains searchStr then
                                    set pos to position of w
                                    set sz to size of w
                                    return (item 1 of pos) & "," & (item 2 of pos) & "," & (item 1 of sz) & "," & (item 2 of sz)
                                end if
                            end try
                        end if
                    end try
                end repeat
            end try
        end repeat
        return "NOT_FOUND"
    end tell
    '''.format(search_escaped)

    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        output = result.stdout.strip()

        if output == "NOT_FOUND":
            print(f"Window '{window_name}' not found. Run list_open_windows() to see exact app/window names.")
            return None

        # Parse "x,y,width,height" — AppleScript can sometimes return empty or extra fields
        parts = [p.strip() for p in output.split(',') if p.strip()]
        if len(parts) != 4:
            print(f"Error parsing window coordinates: got {len(parts)} values (expected 4). Raw output: {output!r}")
            return None
        try:
            x, y, width, height = [int(p) for p in parts]
        except ValueError:
            print(f"Error parsing window coordinates: expected four integers, got {output!r}")
            return None

        print(f"Found window '{window_name}' at ({x}, {y}) with size {width}x{height}")
        return (x, y, width, height)

    except subprocess.CalledProcessError as e:
        print(f"Error finding window: {e}")
        if e.stderr:
            print(e.stderr)
        return None
