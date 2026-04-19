
"""
IDS ML Analyzer — Каталог загроз

Каталог загроз: опис, рівень небезпеки (severity), рекомендації для всіх типів атак
з датасетів CIC-IDS2017, CIC-IDS2018, NSL-KDD, UNSW-NB15.

Використання:
- Декодування prediction у читабельний опис
- Визначення рівня небезпеки
- Генерація рекомендацій
- Вибір кольорів та іконок для UI
"""

from __future__ import annotations
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Рівні небезпеки (Severity levels)
# ═══════════════════════════════════════════════════════════════

SEVERITY_LEVELS = {
    'critical': {'label': 'Критичний', 'color': '#DC2626', 'icon': '🔴', 'weight': 4},
    'high':     {'label': 'Високий',   'color': '#EF4444', 'icon': '🟠', 'weight': 3},
    'medium':   {'label': 'Помірний',  'color': '#F59E0B', 'icon': '🟡', 'weight': 2},
    'low':      {'label': 'Низький',   'color': '#10B981', 'icon': '🟢', 'weight': 1},
    'info':     {'label': 'Інфо',      'color': '#6366F1', 'icon': '🔵', 'weight': 0},
}



# ═══════════════════════════════════════════════════════════════
# Каталог загроз — всі відомі типи атак
# ═══════════════════════════════════════════════════════════════

THREAT_CATALOG: dict[str, dict] = {
    # ── CIC-IDS 2017/2018: DDoS ──
    'ddos': {
        'name_uk': 'DDoS-атака',
        'severity': 'critical',
        'description': 'Розподілена атака відмови в обслуговуванні — масовий потік шкідливого трафіку з багатьох джерел для перевантаження сервера.',
        'impact': 'Повна недоступність сервісу для легітимних користувачів',
        'actions': [
            'Увімкнути DDoS-захист (Cloudflare, AWS Shield)',
            'Заблокувати підозрілі IP на файрволі',
            'Збільшити пропускну здатність тимчасово',
            'Повідомити провайдера',
        ],
    },
    'ddos attack': {
        'alias_of': 'ddos',
    },

    # ── DoS variants ──
    'dos hulk': {
        'name_uk': 'DoS Hulk',
        'severity': 'high',
        'description': 'HTTP-flood атака, що генерує унікальні запити для обходу кешування та перевантаження веб-серверу.',
        'impact': 'Недоступність веб-додатку, вичерпання ресурсів серверу',
        'actions': [
            'Налаштувати rate-limiting для HTTP-запитів',
            'Увімкнути WAF з правилами проти flood',
            'Заблокувати IP-адреси атакуючих',
        ],
    },
    'dos goldeneye': {
        'name_uk': 'DoS GoldenEye',
        'severity': 'high',
        'description': 'HTTP-flood атака з використанням keep-alive з\'єднань для вичерпання пулу потоків сервера.',
        'impact': 'Вичерпання з\'єднань сервера, відмова в обслуговуванні',
        'actions': [
            'Обмежити max-connections per IP',
            'Налаштувати timeout для keep-alive',
            'Моніторинг аномальної кількості з\'єднань',
        ],
    },
    'dos slowloris': {
        'name_uk': 'DoS Slowloris',
        'severity': 'high',
        'description': 'Повільна HTTP-атака, що тримає безліч з\'єднань відкритими, повільно надсилаючи заголовки.',
        'impact': 'Вичерпання пулу з\'єднань, новий трафік блокується',
        'actions': [
            'Встановити мінімальну швидкість передачі даних',
            'Зменшити timeout з\'єднань',
            'Використовувати reverse proxy (nginx)',
        ],
    },
    'dos slowhttptest': {
        'name_uk': 'DoS SlowHTTP',
        'severity': 'high',
        'description': 'Повільна HTTP POST-атака, що імітує повільне завантаження даних на сервер.',
        'impact': 'Блокування ресурсів сервера на обробку "повільних" запитів',
        'actions': [
            'Налаштувати мінімальний body rate',
            'Обмежити розмір POST-запитів',
            'Використовувати timeout для request body',
        ],
    },
    'dos': {
        'name_uk': 'DoS-атака',
        'severity': 'high',
        'description': 'Атака відмови в обслуговуванні — перевантаження цільової системи для зупинки сервісу.',
        'impact': 'Недоступність сервісу, можливе пошкодження даних',
        'actions': [
            'Увімкнути захист від DoS',
            'Налаштувати rate-limiting',
            'Перевірити логи для ідентифікації джерела',
        ],
    },

    # ── Brute Force / Patator ──
    'ftp-patator': {
        'name_uk': 'FTP Brute Force',
        'severity': 'high',
        'description': 'Перебір паролів до FTP-сервера з використанням словників та автоматичних інструментів.',
        'impact': 'Несанкціонований доступ до файлового сервера, витік даних',
        'actions': [
            'Обмежити кількість спроб входу (fail2ban)',
            'Увімкнути двофакторну автентифікацію',
            'Замінити FTP на SFTP/SCP',
            'Перевірити та змінити паролі',
        ],
    },
    'ssh-patator': {
        'name_uk': 'SSH Brute Force',
        'severity': 'high',
        'description': 'Перебір паролів до SSH-сервера. Одна з найпоширеніших атак на Linux-сервери.',
        'impact': 'Повний контроль над сервером при успішному зламі',
        'actions': [
            'Вимкнути автентифікацію за паролем (тільки ключі)',
            'Налаштувати fail2ban для SSH',
            'Змінити стандартний порт SSH',
            'Перевірити authorized_keys на підозрілі ключі',
        ],
    },
    'bruteforce': {
        'name_uk': 'Brute Force',
        'severity': 'high',
        'description': 'Атака перебору паролів або облікових даних для отримання несанкціонованого доступу.',
        'impact': 'Компрометація облікових записів, несанкціонований доступ',
        'actions': [
            'Увімкнути блокування після N невдалих спроб',
            'Налаштувати двофакторну автентифікацію',
            'Використовувати CAPTCHA',
        ],
    },
    'brute force -web': {
        'alias_of': 'bruteforce',
    },

    # ── Web Attacks ──
    'web attack \u2013 brute force': {
        'name_uk': 'Веб-атака (Brute Force)',
        'severity': 'high',
        'description': 'Перебір облікових даних через веб-форми входу.',
        'impact': 'Компрометація веб-акаунтів, несанкціонований доступ до додатку',
        'actions': [
            'Додати CAPTCHA до форм входу',
            'Обмежити швидкість запитів (rate limiting)',
            'Увімкнути блокування IP після N спроб',
        ],
    },
    'web attack \u2013 xss': {
        'name_uk': 'Веб-атака (XSS)',
        'severity': 'high',
        'description': 'Cross-Site Scripting — впровадження шкідливого JavaScript-коду у веб-сторінки.',
        'impact': 'Крадіжка cookies, перенаправлення на шкідливі сайти, defacement',
        'actions': [
            'Санітизувати вхідні дані (input validation)',
            'Використовувати Content Security Policy (CSP)',
            'Увімкнути HTTPOnly для cookies',
        ],
    },
    'web attack \u2013 sql injection': {
        'name_uk': 'SQL-ін\'єкція',
        'severity': 'critical',
        'description': 'Впровадження SQL-коду для маніпуляції базою даних через вразливі веб-форми.',
        'impact': 'Витік всієї бази даних, видалення даних, повний контроль над сервером',
        'actions': [
            'Використовувати параметризовані запити (prepared statements)',
            'Увімкнути WAF з правилами SQL injection',
            'Аудит коду на вразливості',
            'Обмежити права користувача БД',
        ],
    },
    'sql injection': {
        'alias_of': 'web attack \u2013 sql injection',
    },
    'xss': {
        'alias_of': 'web attack \u2013 xss',
    },

    # ── Port Scanning / Reconnaissance ──
    'portscan': {
        'name_uk': 'Сканування портів',
        'severity': 'medium',
        'description': 'Автоматичне сканування відкритих портів для виявлення вразливих сервісів.',
        'impact': 'Збір інформації для подальших атак, виявлення вразливостей',
        'actions': [
            'Закрити непотрібні порти',
            'Налаштувати IPS для блокування сканерів',
            'Моніторинг з\'єднань до нетипових портів',
        ],
    },
    'probe': {
        'name_uk': 'Пробінг мережі',
        'severity': 'medium',
        'description': 'Розвідувальне зондування мережі для виявлення активних хостів та сервісів (NSL-KDD).',
        'impact': 'Картування мережі атакуючим, підготовка до цільової атаки',
        'actions': [
            'Налаштувати firewall для блокування ICMP/сканування',
            'Увімкнути logging підозрілих з\'єднань',
            'Перевірити налаштування мережевої сегментації',
        ],
    },
    'reconnaissance': {
        'name_uk': 'Розвідка',
        'severity': 'medium',
        'description': 'Збір інформації про цільову мережу або систему перед основною атакою (UNSW-NB15).',
        'impact': 'Виявлення слабких місць інфраструктури',
        'actions': [
            'Мінімізувати публічно доступну інформацію',
            'Налаштувати honeynet для виявлення розвідки',
            'Перевірити DNS та WHOIS на витік даних',
        ],
    },

    # ── Bot / Botnet ──
    'bot': {
        'name_uk': 'Ботнет',
        'severity': 'critical',
        'description': 'Заражений хост, що входить у мережу скомпрометованих комп\'ютерів під контролем атакуючого.',
        'impact': 'Використання ресурсів для DDoS, спам, майнінг, та інші зловмисні дії',
        'actions': [
            'Ізолювати заражені хости',
            'Просканувати антивірусом всю мережу',
            'Заблокувати C&C домени/IP',
            'Перевірити DNS-запити на підозрілі домени',
        ],
    },

    # ── Infiltration ──
    'infiltration': {
        'name_uk': 'Вторгнення',
        'severity': 'critical',
        'description': 'Проникнення в мережу з метою встановлення бекдору або крадіжки даних.',
        'impact': 'Повна компрометація мережі, тривала прихована присутність атакуючого',
        'actions': [
            'Негайно ізолювати скомпрометований сегмент',
            'Провести forensic-аналіз',
            'Перевірити всі облікові записи',
            'Переглянути мережеві правила та ACL',
        ],
    },

    # ── Heartbleed ──
    'heartbleed': {
        'name_uk': 'Heartbleed',
        'severity': 'critical',
        'description': 'Експлуатація вразливості OpenSSL CVE-2014-0160 для витоку пам\'яті серверу.',
        'impact': 'Витік приватних ключів, паролів, сесійних cookies з пам\'яті серверу',
        'actions': [
            'Негайно оновити OpenSSL',
            'Перегенерувати всі SSL-сертифікати',
            'Змінити всі паролі',
            'Перевірити логи на ознаки експлуатації',
        ],
    },

    # ── NSL-KDD specific ──
    'r2l': {
        'name_uk': 'Remote-to-Local',
        'severity': 'high',
        'description': 'Атака з віддаленого хоста для отримання локального доступу до машини (NSL-KDD).',
        'impact': 'Несанкціонований локальний доступ, можливість ескалації привілеїв',
        'actions': [
            'Перевірити та оновити сервіси з мережевим доступом',
            'Обмежити мережевий доступ через firewall',
            'Увімкнути моніторинг нових локальних сесій',
        ],
    },
    'u2r': {
        'name_uk': 'User-to-Root',
        'severity': 'critical',
        'description': 'Ескалація привілеїв — звичайний користувач отримує root-доступ (NSL-KDD).',
        'impact': 'Повний контроль над системою, можливість встановити rootkit',
        'actions': [
            'Перевірити SUID/SGID файли',
            'Оновити ядро та системні пакети',
            'Перевірити sudo-правила',
            'Провести аудит привілеїв',
        ],
    },

    # ── UNSW-NB15 specific ──
    'exploits': {
        'name_uk': 'Експлойт',
        'severity': 'critical',
        'description': 'Використання відомих вразливостей програмного забезпечення для несанкціонованого доступу.',
        'impact': 'Виконання довільного коду, повний контроль над системою',
        'actions': [
            'Негайно оновити вразливе ПЗ',
            'Увімкнути IPS з сигнатурами експлойтів',
            'Перевірити систему на ознаки компрометації',
        ],
    },
    'fuzzers': {
        'name_uk': 'Фаззінг',
        'severity': 'medium',
        'description': 'Надсилання випадкових або спеціально сформованих даних для виявлення вразливостей.',
        'impact': 'Виявлення buffer overflow, crash, інших вразливостей',
        'actions': [
            'Перевірити input validation',
            'Оновити ПЗ до останньої версії',
            'Налаштувати WAF для фільтрації малформованих запитів',
        ],
    },
    'generic': {
        'name_uk': 'Загальна загроза',
        'severity': 'medium',
        'description': 'Загальний клас атак, що не підпадають під конкретну категорію (UNSW-NB15).',
        'impact': 'Потенційна компрометація системи',
        'actions': [
            'Детально проаналізувати мережевий трафік',
            'Перевірити логи на підозрілу активність',
            'Посилити моніторинг',
        ],
    },
    'shellcode': {
        'name_uk': 'Шелл-код',
        'severity': 'critical',
        'description': 'Впровадження виконуваного коду через вразливість для отримання командного shell.',
        'impact': 'Повний контроль над процесом або системою',
        'actions': [
            'Увімкнути DEP/ASLR',
            'Оновити вразливе ПЗ',
            'Перевірити систему на rootkits',
        ],
    },
    'worms': {
        'name_uk': 'Хробак',
        'severity': 'critical',
        'description': 'Самопоширюваний шкідливий код, що автоматично заражає інші системи в мережі.',
        'impact': 'Масове зараження мережі, вичерпання ресурсів, можливе знищення даних',
        'actions': [
            'Негайно ізолювати заражені хости',
            'Заблокувати мережеву комунікацію між сегментами',
            'Розгорнути антивірус на всіх хостах',
            'Оновити всі системи',
        ],
    },
    'analysis': {
        'name_uk': 'Аналіз (розвідка)',
        'severity': 'medium',
        'description': 'Аналітичні атаки для збору інформації про конфігурацію мережі та сервісів (UNSW-NB15).',
        'impact': 'Збір інформації для планування подальших атак',
        'actions': [
            'Закрити непотрібні сервіси',
            'Налаштувати firewall для обмеження інформації',
            'Моніторинг нетипових запитів',
        ],
    },
    'backdoor': {
        'name_uk': 'Бекдор',
        'severity': 'critical',
        'description': 'Прихований канал доступу до системи, що обходить стандартну автентифікацію.',
        'impact': 'Постійний несанкціонований доступ, можливість повторного зламу',
        'actions': [
            'Повний аудит системи (файли, процеси, мережеві з\'єднання)',
            'Перевстановити скомпрометовану систему',
            'Змінити всі облікові дані',
            'Переглянути мережеві правила',
        ],
    },
    'backdoors': {
        'alias_of': 'backdoor',
    },

    # ── Flood variants ──
    'synflood': {
        'name_uk': 'SYN Flood',
        'severity': 'high',
        'description': 'Надсилання масової кількості SYN-пакетів для вичерпання таблиці з\'єднань серверу.',
        'impact': 'Неможливість встановлення нових TCP-з\'єднань',
        'actions': [
            'Увімкнути SYN cookies',
            'Налаштувати rate-limiting для SYN',
            'Збільшити backlog queue',
        ],
    },
    'udpflood': {
        'name_uk': 'UDP Flood',
        'severity': 'high',
        'description': 'Надсилання великої кількості UDP-пакетів для перевантаження мережевого каналу.',
        'impact': 'Перевантаження каналу, недоступність UDP-сервісів',
        'actions': [
            'Rate-limiting для UDP',
            'Заблокувати невикористовувані UDP-порти',
            'Увімкнути anti-DDoS захист',
        ],
    },

    # ── CIC-IDS 2018 specific ──
    'ddos attack-loic-udp': {
        'alias_of': 'ddos',
    },
    'ddos attack-hoic': {
        'alias_of': 'ddos',
    },
    'ddos attacks-loic-http': {
        'alias_of': 'ddos',
    },
    'ddos attack-loic-http': {
        'alias_of': 'ddos',
    },
    'dos attacks-hulk': {
        'alias_of': 'dos hulk',
    },
    'dos attacks-slowhttptest': {
        'alias_of': 'dos slowhttptest',
    },
    'dos attacks-goldeneye': {
        'alias_of': 'dos goldeneye',
    },
    'dos attacks-slowloris': {
        'alias_of': 'dos slowloris',
    },
    'brute force -xss': {
        'alias_of': 'web attack \u2013 xss',
    },

    # ── Aliases for localized (Ukrainian) names ──
    'ftp brute force': {
        'alias_of': 'ftp-patator',
    },
    'ssh brute force': {
        'alias_of': 'ssh-patator',
    },
    'ddos-атака': {
        'alias_of': 'ddos',
    },
    'dos-атака': {
        'alias_of': 'dos',
    },
    'dos (goldeneye)': {
        'alias_of': 'dos goldeneye',
    },
    'dos (hulk)': {
        'alias_of': 'dos hulk',
    },
    'dos (slowloris)': {
        'alias_of': 'dos slowloris',
    },
    'dos (slowhttp)': {
        'alias_of': 'dos slowhttptest',
    },
    'сканування портів': {
        'alias_of': 'portscan',
    },
    'ботнет': {
        'alias_of': 'bot',
    },
    'вторгнення': {
        'alias_of': 'infiltration',
    },
    'веб-атака (brute force)': {
        'alias_of': 'web attack \u2013 brute force',
    },
    'веб-атака (xss)': {
        'alias_of': 'web attack \u2013 xss',
    },
    'sql-ін\'єкція': {
        'alias_of': 'web attack \u2013 sql injection',
    },
    'експлойт': {
        'alias_of': 'exploits',
    },
    'бекдор': {
        'alias_of': 'backdoor',
    },
    'хробак': {
        'alias_of': 'worms',
    },
    'шелл-код': {
        'alias_of': 'shellcode',
    },
    'розвідка': {
        'alias_of': 'reconnaissance',
    },
    'фаззінг': {
        'alias_of': 'fuzzers',
    },
    'brute force': {
        'alias_of': 'bruteforce',
    },
    'вразливість heartbleed': {
        'alias_of': 'heartbleed',
    },
}


# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════

def _resolve(key: str) -> dict:
    """
    Розв'язання ланцюжків аліасів (максимум 3 рівня для запобігання циклам).
    """
    entry = THREAT_CATALOG.get(key)
    for _ in range(3):
        if entry is None or 'alias_of' not in entry:
            break
        entry = THREAT_CATALOG.get(entry['alias_of'])
    return entry or {}


def get_threat_info(prediction_label: str) -> dict:
    """
    Отримати інформацію про загрозу за prediction label.

    Повертає dict з ключами:
        name_uk, severity, description, impact, actions
    Або fallback-словник для невідомих загроз.
    """
    label = str(prediction_label).strip()
    label_lower = label.lower()

    # Exact match (case-insensitive)
    entry = _resolve(label_lower)
    if entry and 'name_uk' in entry:
        return entry

    # Fuzzy match: check if any catalog key is contained in the label
    for key in THREAT_CATALOG:
        if len(key) > 3 and key in label_lower:
            resolved = _resolve(key)
            if resolved and 'name_uk' in resolved:
                return resolved

    # Fallback for completely unknown threats
    return {
        'name_uk': label if label else 'Невідома загроза',
        'severity': 'medium',
        'description': f'Виявлено підозрілу мережеву активність типу "{label}".',
        'impact': 'Потенційний ризик безпеці мережі',
        'actions': [
            'Проаналізувати деталі мережевого трафіку',
            'Перевірити логи на підозрілу активність',
            'За потреби — заблокувати підозрілі IP',
        ],
    }


def get_severity(prediction_label: str) -> str:
    """
    Повертає рівень небезпеки (severity) для prediction label.
    """
    info = get_threat_info(prediction_label)
    return info.get('severity', 'medium')


def get_severity_color(severity: str) -> str:
    """
    Повертає HEX-колір для рівня небезпеки.
    """
    return SEVERITY_LEVELS.get(severity, SEVERITY_LEVELS['medium'])['color']


def get_severity_label(severity: str) -> str:
    """
    Повертає українську назву рівня небезпеки.
    """
    return SEVERITY_LEVELS.get(severity, SEVERITY_LEVELS['medium'])['label']


def get_severity_icon(severity: str) -> str:
    """
    Повертає emoji-іконку для рівня небезпеки.
    """
    return SEVERITY_LEVELS.get(severity, SEVERITY_LEVELS['medium'])['icon']


def classify_if_anomaly_score(score: float) -> dict:
    """
    Класифікує anomaly score від Isolation Forest у рівень небезпеки.

    Логіка (decision_function: score < 0 = аномалія):
      score < -0.4 → critical (дуже сильна аномалія)
      score < -0.2 → high
      score < -0.1 → medium
      score <  0.0 → low
      score >= 0.0 → normal (не аномалія)
    """
    if score < -0.4:
        return {
            'severity': 'critical',
            'label': 'Критична аномалія',
            'description': 'Різко аномальна поведінка мережевого трафіку. Ймовірна активна атака.',
        }
    elif score < -0.2:
        return {
            'severity': 'high',
            'label': 'Висока аномалія',
            'description': 'Значне відхилення від нормальної поведінки. Потребує негайної уваги.',
        }
    elif score < -0.1:
        return {
            'severity': 'medium',
            'label': 'Помірна аномалія',
            'description': 'Помітне відхилення від базової лінії. Рекомендується перевірка.',
        }
    else:
        return {
            'severity': 'low',
            'label': 'Слабка аномалія',
            'description': 'Незначне відхилення. Може бути нормальною варіацією трафіку.',
        }

def enrich_predictions(predictions: list, prediction_labels: list,
                       anomaly_scores: Optional[list] = None,
                       is_isolation_forest: bool = False) -> list[dict]:
    """
    Збагачує predictions деталями з каталогу загроз.

    Повертає список dict, по одному на кожен prediction:
        {
            'label': str,          # Оригінальний label
            'name_uk': str,        # Українська назва
            'severity': str,       # critical/high/medium/low/info
            'severity_label': str, # Критичний/Високий/...
            'severity_color': str, # #DC2626/...
            'description': str,    # Опис загрози
            'is_threat': bool,     # True якщо це загроза
        }
    """
    results = []
    benign_set = {'норма', 'benign', 'normal', '0', '0.0'}

    for i, label in enumerate(prediction_labels):
        label_str = str(label).strip()
        label_lower = label_str.lower()

        if label_lower in benign_set:
            results.append({
                'label': label_str,
                'name_uk': 'Норма',
                'severity': 'info',
                'severity_label': 'Безпечно',
                'severity_color': '#10B981',
                'description': 'Нормальний мережевий трафік',
                'is_threat': False,
            })
            continue

        # For Isolation Forest: use anomaly scores for severity
        if is_isolation_forest and anomaly_scores and i < len(anomaly_scores):
            score = anomaly_scores[i]
            score_info = classify_if_anomaly_score(score)
            results.append({
                'label': label_str,
                'name_uk': score_info['label'],
                'severity': score_info['severity'],
                'severity_label': get_severity_label(score_info['severity']),
                'severity_color': get_severity_color(score_info['severity']),
                'description': score_info['description'],
                'is_threat': True,
            })
        else:
            # For classification models: use threat catalog
            info = get_threat_info(label_str)
            sev = info.get('severity', 'medium')
            results.append({
                'label': label_str,
                'name_uk': info.get('name_uk', label_str),
                'severity': sev,
                'severity_label': get_severity_label(sev),
                'severity_color': get_severity_color(sev),
                'description': info.get('description', ''),
                'is_threat': True,
            })

    return results
