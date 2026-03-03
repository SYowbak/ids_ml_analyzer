"""
IDS ML Analyzer - Gemini AI Service

Сервіс AI-аналізу загроз та генерації пояснень для звітів IDS.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import google.generativeai as genai

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Рівні критичності загроз."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ThreatAnalysis:
    """Структурований результат аналізу загрози."""

    threat_type: str
    severity: SeverityLevel
    description: str
    indicators: list[str]
    recommendations: list[str]
    technical_details: str
    raw_markdown: str


class GeminiService:
    """
    Інтеграція з Google Gemini API.
    Надає короткі та детальні пояснення для результатів сканування.
    """

    SEVERITY_KEYWORDS = {
        SeverityLevel.CRITICAL: ["ddos", "dos", "brute force", "sql injection", "xss", "rce", "exploit"],
        SeverityLevel.HIGH: ["infiltration", "botnet", "malware", "backdoor", "heartbleed"],
        SeverityLevel.MEDIUM: ["portscan", "port scan", "probe", "reconnaissance", "scan"],
        SeverityLevel.LOW: ["benign", "normal", "legitimate"],
    }

    MODEL_CANDIDATES = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-flash-latest",
        "gemini-pro",
    ]

    def __init__(self, api_key: str | None = None):
        self.model = None
        self.model_name = ""
        self.available = False
        self._available_model_names: list[str] = []

        if not api_key:
            logger.warning("Gemini API key not provided. AI features disabled.")
            return

        try:
            genai.configure(api_key=api_key)
        except Exception as e:
            logger.error(f"Gemini configure error: {e}")
            return

        self._available_model_names = self._list_generate_content_models()

        for candidate in self._candidate_model_sequence():
            if self._bind_model(candidate):
                self.available = True
                logger.info(f"Gemini model selected: {candidate}")
                break

        if not self.available:
            logger.error("Failed to initialize any Gemini model candidate.")

    def _list_generate_content_models(self) -> list[str]:
        """Best-effort discovery of models that support generateContent."""
        try:
            discovered = []
            for model in genai.list_models():
                model_name = str(getattr(model, "name", "") or "")
                if not model_name:
                    continue
                if model_name.startswith("models/"):
                    model_name = model_name.split("/", 1)[1]

                methods = [str(m).lower() for m in (getattr(model, "supported_generation_methods", None) or [])]
                if methods and not any("generatecontent" in m for m in methods):
                    continue

                discovered.append(model_name)

            return list(dict.fromkeys(discovered))
        except Exception as e:
            logger.debug(f"Gemini list_models unavailable: {e}")
            return []

    def _candidate_model_sequence(self) -> list[str]:
        if not self._available_model_names:
            return list(self.MODEL_CANDIDATES)

        available_set = set(self._available_model_names)
        preferred = [m for m in self.MODEL_CANDIDATES if m in available_set]
        discovered_tail = [m for m in self._available_model_names if m not in preferred]
        return preferred + discovered_tail

    def _bind_model(self, model_name: str) -> bool:
        try:
            self.model = genai.GenerativeModel(model_name)
            self.model_name = model_name
            return True
        except Exception as e:
            logger.warning(f"Gemini model init failed for {model_name}: {e}")
            return False

    def _generate_with_fallback(self, prompt: str) -> str:
        if not self.available:
            raise RuntimeError("Gemini is not available")

        candidate_order: list[str] = []
        if self.model_name:
            candidate_order.append(self.model_name)
        for candidate in self._candidate_model_sequence():
            if candidate not in candidate_order:
                candidate_order.append(candidate)

        last_error: Exception | None = None
        for candidate in candidate_order:
            if candidate != self.model_name and not self._bind_model(candidate):
                continue
            try:
                response = self.model.generate_content(prompt)
                text = (getattr(response, "text", "") or "").strip()
                if text:
                    return text
                return str(response)
            except Exception as e:
                last_error = e
                logger.warning(f"Gemini request failed for model {candidate}: {e}")

        self.available = False
        raise RuntimeError(f"All Gemini model candidates failed: {last_error}")

    def _determine_severity(self, threat_type: str) -> SeverityLevel:
        threat_lower = threat_type.lower()
        for severity, keywords in self.SEVERITY_KEYWORDS.items():
            if any(kw in threat_lower for kw in keywords):
                return severity
        return SeverityLevel.MEDIUM

    def _format_features(self, features: dict) -> str:
        if not features:
            return "Дані відсутні"

        lines = []
        for key, value in list(features.items())[:12]:
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.4f}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _format_network_context(scan_summary: dict) -> str:
        context = scan_summary.get("network_context", {})
        if not isinstance(context, dict) or not context:
            return "Немає додаткових мережевих деталей."

        field_map = {
            "top_dst_ports": "Найчастіші порти призначення",
            "top_src_ports": "Найчастіші порти джерела",
            "top_protocols": "Найчастіші протоколи",
            "top_src_ips": "Найчастіші IP джерела",
            "top_dst_ips": "Найчастіші IP призначення",
        }

        lines: list[str] = []
        for key, title in field_map.items():
            values = context.get(key)
            if not isinstance(values, list) or not values:
                continue
            pretty_values = ", ".join(str(v) for v in values[:8])
            lines.append(f"- {title}: {pretty_values}")

        return "\n".join(lines) if lines else "Немає додаткових мережевих деталей."

    def analyze_threat(self, threat_type: str, features: dict) -> str:
        """Коротке пояснення для однієї загрози."""
        if not self.available:
            return "Gemini API недоступний. Додайте API ключ у налаштуваннях."

        severity = self._determine_severity(threat_type)
        prompt = f"""
Ти аналітик SOC. Дай коротке пояснення українською.
Не використовуй markdown-посилання, emoji, символи типу 🔗/📎/🖇.

Тип загрози: {threat_type}
Рівень критичності: {severity.value.upper()}
Ознаки трафіку:
{self._format_features(features)}

Формат:
## Опис загрози
[2-3 речення]

## Індикатори
- [...]
- [...]

## Рекомендовані дії
1. [...]
2. [...]
3. [...]
"""
        try:
            logger.info("Calling Gemini API for threat analysis")
            return self._generate_with_fallback(prompt)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Не вдалося отримати пояснення від AI: {e}"

    def analyze_threat_detailed(self, threat_type: str, features: dict, similar_count: int = 0) -> ThreatAnalysis:
        """Детальний структурований аналіз загрози."""
        severity = self._determine_severity(threat_type)

        if not self.available:
            return ThreatAnalysis(
                threat_type=threat_type,
                severity=severity,
                description="AI-аналіз недоступний. Додайте Gemini API ключ.",
                indicators=[],
                recommendations=["Налаштуйте API ключ Gemini та повторіть аналіз."],
                technical_details="",
                raw_markdown="",
            )

        prompt = f"""
Ти провідний SOC-аналітик. Проаналізуй загрозу українською.
Не використовуй markdown-посилання, emoji, символи типу 🔗/📎/🖇.

Загроза: {threat_type}
Кількість подібних інцидентів: {similar_count}
Ознаки:
{self._format_features(features)}

Строгий формат:
### ОПИС
[2-3 речення]

### ІНДИКАТОРИ
- Індикатор 1
- Індикатор 2
- Індикатор 3

### РЕКОМЕНДАЦІЇ
1. Рекомендація 1
2. Рекомендація 2
3. Рекомендація 3

### ТЕХНІЧНІ_ДЕТАЛІ
[Короткий технічний опис]
"""
        try:
            logger.info("Calling Gemini API for detailed threat analysis")
            raw_text = self._generate_with_fallback(prompt)
            return self._parse_detailed_response(raw_text, threat_type, severity)
        except Exception as e:
            logger.error(f"Gemini detailed analysis error: {e}")
            return ThreatAnalysis(
                threat_type=threat_type,
                severity=severity,
                description=f"Помилка AI-аналізу: {e}",
                indicators=[],
                recommendations=[],
                technical_details="",
                raw_markdown="",
            )

    def _parse_detailed_response(self, text: str, threat_type: str, severity: SeverityLevel) -> ThreatAnalysis:
        """Парсить відповідь Gemini у структурований формат."""
        description = ""
        indicators: list[str] = []
        recommendations: list[str] = []
        technical_details = ""

        current_section: Optional[str] = None
        for raw_line in text.split("\n"):
            line = raw_line.strip()
            upper = line.upper()

            if "### ОПИС" in upper or "### DESCRIPTION" in upper:
                current_section = "description"
                continue
            if "### ІНДИКАТОРИ" in upper or "### INDICATORS" in upper:
                current_section = "indicators"
                continue
            if "### РЕКОМЕНДАЦІЇ" in upper or "### RECOMMENDATIONS" in upper:
                current_section = "recommendations"
                continue
            if "### ТЕХНІЧНІ" in upper or "### TECHNICAL" in upper:
                current_section = "technical"
                continue

            if not line:
                continue

            if current_section == "description":
                description += f"{line} "
            elif current_section == "indicators" and line.startswith("-"):
                indicators.append(line[1:].strip())
            elif current_section == "recommendations":
                clean = line.lstrip("0123456789.-) ").strip()
                if clean:
                    recommendations.append(clean)
            elif current_section == "technical":
                technical_details += f"{line} "

        return ThreatAnalysis(
            threat_type=threat_type,
            severity=severity,
            description=description.strip() or f"Виявлено {threat_type}",
            indicators=indicators[:5] if indicators else ["Аномальні мережеві патерни"],
            recommendations=recommendations[:5] if recommendations else ["Провести детальне розслідування інциденту"],
            technical_details=technical_details.strip(),
            raw_markdown=text,
        )

    def generate_executive_summary(self, scan_summary: dict, top_threats: list[dict]) -> str:
        """Коротке пояснення для керівництва."""
        if not self.available:
            return self._generate_fallback_summary(scan_summary, top_threats)

        threats_text = "\n".join(
            f"- {t.get('type', 'Unknown')}: {int(t.get('count', 0) or 0):,}"
            for t in top_threats[:6]
        )
        network_context_text = self._format_network_context(scan_summary)

        prompt = f"""
Ти CISO великої компанії. Підготуй змістовний короткий звіт для керівництва українською.
Не використовуй markdown-посилання, emoji, символи типу 🔗/📎/🖇.

Дані:
- Всього записів: {scan_summary.get('total', 0)}
- Аномалій: {scan_summary.get('anomalies', 0)}
- Рівень ризику: {scan_summary.get('risk_score', 0)}%
- Модель: {scan_summary.get('model_name', 'N/A')}
- Алгоритм: {scan_summary.get('algorithm', 'N/A')}
- Файл: {scan_summary.get('filename', 'N/A')}

Найчастіші загрози:
{threats_text}

Мережевий контекст (якщо доступний):
{network_context_text}

Вимоги:
- 260-420 слів.
- Конкретні цифри з даних.
- Пояснюй простою мовою без зайвого жаргону.
- Якщо є порти/протоколи/IP, явно вкажи найкритичніші значення.

Формат:
## Загальна оцінка
[2-4 речення]

## Ключові ризики
- [...]
- [...]
- [...]

## Рекомендовані дії
### Негайно (0-4 години)
1. [...]
2. [...]

### Протягом доби
1. [...]
2. [...]

## Ключові технічні індикатори
- [...]
- [...]

## Висновок
[1-2 речення]
"""
        try:
            logger.info("Calling Gemini API for executive summary")
            return self._generate_with_fallback(prompt)
        except Exception as e:
            logger.error(f"Executive summary generation error: {e}")
            return self._generate_fallback_summary(scan_summary, top_threats)

    def _generate_fallback_summary(self, scan_summary: dict, top_threats: list[dict]) -> str:
        """Детальний локальний звіт, якщо Gemini недоступний."""
        total = int(scan_summary.get("total", 0) or 0)
        anomalies = int(scan_summary.get("anomalies", 0) or 0)
        risk = float(scan_summary.get("risk_score", 0) or 0)
        model_name = str(scan_summary.get("model_name", "N/A") or "N/A")
        algorithm = str(scan_summary.get("algorithm", "N/A") or "N/A")
        filename = str(scan_summary.get("filename", "N/A") or "N/A")
        network_context_text = self._format_network_context(scan_summary)

        anomaly_ratio = (anomalies / total * 100.0) if total > 0 else 0.0

        if risk < 10:
            status = "Низький"
            status_note = "Критичних ознак інциденту не зафіксовано."
        elif risk < 30:
            status = "Помірний"
            status_note = "Є аномальна активність, потрібна додаткова перевірка."
        elif risk < 60:
            status = "Підвищений"
            status_note = "Висока інтенсивність підозрілого трафіку, потрібні активні дії."
        else:
            status = "Критичний"
            status_note = "Ситуація високого ризику, потрібне негайне реагування."

        prepared_threats: list[tuple[str, int, float]] = []
        for item in top_threats[:8]:
            t_name = str(item.get("type", "Unknown") or "Unknown")
            t_count = int(item.get("count", 0) or 0)
            t_share = (t_count / anomalies * 100.0) if anomalies > 0 else 0.0
            prepared_threats.append((t_name, t_count, t_share))

        if not prepared_threats:
            prepared_threats.append(("Немає класифікованих загроз", 0, 0.0))

        threats_block = "\n".join(
            f"- **{name}**: {count:,} ({share:.2f}% від усіх аномалій)"
            for name, count, share in prepared_threats
        )
        top_name, top_count, top_share = prepared_threats[0]

        return f"""## Загальна оцінка
**Рівень ризику: {status} ({risk:.2f}%)**. {status_note}

Проаналізовано **{total:,}** записів у файлі **{filename}**.
Виявлено **{anomalies:,}** аномалій (**{anomaly_ratio:.2f}%** від загального трафіку).

Модель: **{model_name}**  
Алгоритм: **{algorithm}**

## Ключові ризики
Наймасовіша загроза: **{top_name}** ({top_count:,}, {top_share:.2f}%).

{threats_block}

## Рекомендовані дії
### Негайно (0-4 години)
1. Заблокувати або обмежити джерела трафіку з наймасовішою аномалією.
2. Перевірити логи firewall/IDS/проксі за останні 24 години по ключових IOC.
3. Увімкнути підвищений моніторинг і алерти для критичних сервісів.

### Сьогодні (до 24 годин)
1. Перевірити відкриті порти та правила доступу, прибрати зайві публічні сервіси.
2. Валідовати облікові записи з підозрілою активністю (SSH/RDP/VPN).
3. Оновити сигнатури IDS/IPS та правила rate-limit.

## Що перевірити в логах
- Пікові джерела IP за кількістю подій.
- Нетипові порти/endpoint-и з повторюваними запитами.
- Серії невдалих аутентифікацій.

## Ключові технічні індикатори
{network_context_text}

## Висновок
Навіть без відповіді Gemini система сформувала базовий план реагування. Почніть зі стримування трафіку, далі проведіть валідацію джерел і посильте правила захисту.
"""

    def generate_security_report(self, summary: dict) -> str:
        """Повний технічний звіт для документації."""
        if not self.available:
            return "Технічний звіт недоступний без Gemini API."

        prompt = f"""
Ти старший SOC-аналітик. Підготуй технічний звіт українською.
Не використовуй markdown-посилання, emoji, символи типу 🔗/📎/🖇.

Дані:
- Всього пакетів: {summary.get('total', 0)}
- Аномалій: {summary.get('anomalies', 0)}
- Рівень ризику: {summary.get('risk_score', 0)}%
- Найчастіші загрози: {summary.get('top_threats', [])}
- Модель: {summary.get('model_name', 'N/A')}

Формат:
## Аналіз інцидентів
## Вектори атак
## План реагування (0-4 години / до 24 годин / до 7 днів)
## Технічні рекомендації
## Висновок
"""
        try:
            logger.info("Calling Gemini API for security report")
            return self._generate_with_fallback(prompt)
        except Exception as e:
            logger.error(f"Security report generation error: {e}")
            return f"Помилка генерації технічного звіту: {e}"

    def analyze_multiple_threats(self, threats: list[dict], max_threats: int = 5) -> list[ThreatAnalysis]:
        """Аналізує декілька загроз послідовно."""
        results: list[ThreatAnalysis] = []
        for threat in threats[:max_threats]:
            results.append(
                self.analyze_threat_detailed(
                    threat_type=threat.get("type", "Unknown"),
                    features=threat.get("features", {}),
                    similar_count=int(threat.get("count", 1) or 1),
                )
            )
        return results

    def generate_comprehensive_analysis(self, scan_summary: dict, all_threats: dict, sample_data: dict | None = None) -> str:
        """Детальний комплексний SOC-аналіз усіх виявлених загроз."""
        if not self.available:
            return "Gemini API недоступний для детального аналізу."

        sorted_threats = sorted(all_threats.items(), key=lambda x: x[1], reverse=True)
        threats_info = "\n".join(f"- **{name}**: {int(count):,}" for name, count in sorted_threats[:15])

        sample_anomalies = (sample_data or {}).get("sample_anomalies", []) if sample_data else []
        sample_count = len(sample_anomalies)
        network_context_text = self._format_network_context(scan_summary)

        prompt = f"""
Ти SOC-аналітик рівня 3. Зроби повний технічний аналіз інциденту українською.
Не використовуй markdown-посилання, emoji, символи типу 🔗/📎/🖇.
Пиши чітко, предметно, із прив'язкою до чисел.

Вхідні дані:
- Всього записів: {scan_summary.get('total', 0):,}
- Аномалій: {scan_summary.get('anomalies', 0):,}
- Рівень ризику: {scan_summary.get('risk_score', 0)}%
- Модель: {scan_summary.get('model_name', 'N/A')}
- Алгоритм: {scan_summary.get('algorithm', 'N/A')}
- Файл: {scan_summary.get('filename', 'N/A')}
- Кількість прикладів аномалій у вибірці: {sample_count}

Мережевий контекст:
{network_context_text}

Типи загроз:
{threats_info}

Формат відповіді:
## Звіт з аналізу мережевої безпеки (SOC LEVEL 3)

## Загальна оцінка ситуації
[2-4 речення, тільки по суті]

## Критичні загрози (потребують негайної реакції)
[перелік + конкретні ризики]

## Загрози середнього пріоритету (24-48 годин)
[перелік + конкретні дії]

## Загрози низького пріоритету
[що лишити на моніторинг]

## Патерни та тренди
[висновки по поведінці трафіку]

## План дій (Action Plan)
### Негайно (0-4 години)
1. [...]
2. [...]

### Протягом доби
1. [...]
2. [...]

### Протягом тижня
1. [...]
2. [...]

## Технічні рекомендації
- [...]
- [...]

## Що саме перевірити в логах
- Конкретні порти/протоколи/IP з найвищим ризиком.
- Які джерела трафіку відсікти негайно, які взяти на моніторинг.

## Висновок
[1-2 речення]
"""
        try:
            logger.info("Calling Gemini API for comprehensive analysis")
            return self._generate_with_fallback(prompt)
        except Exception as e:
            logger.error(f"Comprehensive analysis error: {e}")
            top_list = ", ".join(str(k) for k, _ in sorted_threats[:5]) or "немає класифікованих загроз"
            return f"""## Аналіз недоступний

Не вдалося виконати AI-аналіз через помилку: {str(e)[:140]}.

### Базова інформація:
- Виявлено {scan_summary.get('anomalies', 0)} аномалій із {scan_summary.get('total', 0)} записів
- Рівень ризику: {scan_summary.get('risk_score', 0)}%
- Типи загроз: {top_list}

Рекомендація: перевірте API ключ Gemini та повторіть запит.
"""


if __name__ == "__main__":
    print("GeminiService loaded successfully.")
