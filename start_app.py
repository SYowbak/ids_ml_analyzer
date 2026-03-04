import subprocess
import os
import sys
import time
import webbrowser
import socket
from pathlib import Path

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start():
    print("===========================================")
    print("   IDS ML Analyzer - STARTING")
    print("===========================================")
    
    root_dir = Path(__file__).parent.absolute()
    os.chdir(root_dir)
    
    venv_python = root_dir / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print(f"[ERROR] .venv not found at {venv_python}")
        return

    # Check if port 8501 is already taken
    if is_port_in_use(8501):
        print("[INFO] Port 8501 is busy. Attempting to kill old process...")
        try:
            # Simple windows command to kill whatever is on the port
            subprocess.run('for /f "tokens=5" %a in (\'netstat -aon ^| findstr :8501 ^| findstr LISTENING\') do taskkill /F /PID %a', shell=True, capture_output=True)
            time.sleep(1)
        except:
            pass

    print("[1/2] Starting Streamlit server...")
    # Hide window for the subprocess if possible, or just run it background
    cmd = [
        str(venv_python), "-m", "streamlit", "run", 
        "src/ui/app.py", 
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    # We use CREATE_NO_WINDOW to keep it clean on Windows
    process = subprocess.Popen(cmd, creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    
    print("[2/2] Opening browser...")
    url = "http://localhost:8501"
    
    # Try to wait until port is open
    for i in range(15):
        if is_port_in_use(8501):
            break
        print(".", end="", flush=True)
        time.sleep(1)
    
    print(f"\n[INFO] Launching {url}")
    webbrowser.open(url)
    
    print("\n[READY] Application launched successfully!")
    print("You can close this black window now if you want,")
    print("but the server will stay running in the background.")
    print("===========================================")
    time.sleep(5)

if __name__ == "__main__":
    start()
