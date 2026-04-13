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
    print("   IDS ML Analyzer - ЗАПУСК")
    print("===========================================")
    
    root_dir = Path(__file__).parent.absolute()
    os.chdir(root_dir)
    
    venv_python = root_dir / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        print(f"[ПОМИЛКА] .venv не знайдено за шляхом: {venv_python}")
        return

    # Перевіряємо, чи зайнятий порт Streamlit.
    if is_port_in_use(8501):
        print("[ІНФО] Порт 8501 зайнятий. Пробуємо завершити попередній процес...")
        try:
            # Акуратно зупиняємо процес, що слухає 8501, щоб уникнути конфлікту запуску.
            subprocess.run('for /f "tokens=5" %a in (\'netstat -aon ^| findstr :8501 ^| findstr LISTENING\') do taskkill /F /PID %a', shell=True, capture_output=True)
            time.sleep(1)
        except Exception as exc:
            print(f"[ПОПЕРЕДЖЕННЯ] Не вдалося звільнити порт 8501: {exc}")

    print("[1/2] Запускаємо Streamlit сервер...")
    # Запускаємо Streamlit окремим процесом.
    cmd = [
        str(venv_python), "-m", "streamlit", "run", 
        "src/ui/app.py", 
        "--server.port", "8501",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    # Логи/traceback залишаються в консолі, щоб спростити діагностику.
    process = subprocess.Popen(cmd)
    
    print("[2/2] Відкриваємо браузер...")
    url = "http://localhost:8501"
    
    # Чекаємо підняття порту перед відкриттям браузера.
    for i in range(15):
        if is_port_in_use(8501):
            break
        print(".", end="", flush=True)
        time.sleep(1)
    
    print(f"\n[ІНФО] Відкриваємо {url}")
    webbrowser.open(url)
    print("\n[ГОТОВО] Застосунок успішно запущено!")
    print("За потреби це вікно можна закрити,")
    print("сервер продовжить працювати у фоні.")
    print("===========================================")

if __name__ == "__main__":
    start()
