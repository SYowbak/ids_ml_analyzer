import os
import re
from pathlib import Path

def fix_match(m):
    original = m.group(0)
    try:
        # Спроба розкодувати кракозябру (UTF-8 текст, який був прочитаний як CP1251)
        fixed = original.encode('cp1251').decode('utf-8')
        return fixed
    except Exception:
        # Якщо розкодування не вдалося, це нормальний текст або інша кодування
        return original

def process_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return False

    # Всі байти UTF-8 для кирилиці >= 0x80, тобто жоден з них не є ASCII (<= 0x7F).
    # У кодуванні CP1251 всі байти >= 0x80 - це не-ASCII символи.
    # Тому такий регулярний вираз ідеально і безпечно відділяє кракозябри.
    new_content = re.sub(r'[^\x00-\x7F]+', fix_match, content)
    
    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"✅ Виправлено: {filepath}")
        return True
    return False

if __name__ == '__main__':
    project_root = Path(__file__).parent
    count = 0
    
    extensions = ('.py', '.md', '.json', '.txt', '.bat')
    
    for root, dirs, files in os.walk(project_root):
        # Пропускаємо системні директорії
        if any(skip in root for skip in ['.git', '__pycache__', 'venv', '.venv', '.gemini']):
            continue
            
        for file in files:
            if file.endswith(extensions):
                filepath = Path(root) / file
                if filepath.name == 'fix_mojibake.py':
                    continue
                    
                if process_file(filepath):
                    count += 1
                    
    print(f"\n✨ Готово! Виправлено кракозябри у {count} файлах.")
