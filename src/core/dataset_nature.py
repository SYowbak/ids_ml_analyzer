from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.core.domain_schemas import DATASET_SCHEMAS, normalize_column_name


NATURE_NETWORK_INTRUSION = "network_intrusion"
NATURE_CLASSIC_IDS = "classic_ids_benchmark"
NATURE_MODERN_NETWORK = "modern_network_dataset"


@dataclass(frozen=True)
class NatureDefinition:
    nature_id: str
    label: str
    description: str
    dataset_types: tuple[str, ...]
    dataset_display_names: tuple[str, ...]
    features_description: str
    supported_models: tuple[str, ...]


NATURE_DEFINITIONS: dict[str, NatureDefinition] = {
    NATURE_NETWORK_INTRUSION: NatureDefinition(
        nature_id=NATURE_NETWORK_INTRUSION,
        label="Мережева інфільтрація",
        description="Flow-орієнтований аналіз мережевого трафіку (сценарії NIDS).",
        dataset_types=("CIC-IDS",),
        dataset_display_names=("CIC-IDS2017", "CIC-IDS2018"),
        features_description="Flow-ознаки: тривалість, байти, пакети, прапори, порти, IP.",
        supported_models=("Random Forest", "XGBoost", "Isolation Forest"),
    ),
    NATURE_CLASSIC_IDS: NatureDefinition(
        nature_id=NATURE_CLASSIC_IDS,
        label="Класичний IDS-бенчмарк",
        description="Класичні еталонні датасети IDS з connection-ознаками.",
        dataset_types=("NSL-KDD",),
        dataset_display_names=("NSL-KDD",),
        features_description="Connection-ознаки: protocol_type, service, flag, src_bytes та інші.",
        supported_models=("Random Forest", "XGBoost"),
    ),
    NATURE_MODERN_NETWORK: NatureDefinition(
        nature_id=NATURE_MODERN_NETWORK,
        label="Сучасний мережевий датасет",
        description="Сучасний гібридний мережевий датасет (мережеві + контентні + часові ознаки).",
        dataset_types=("UNSW-NB15",),
        dataset_display_names=("UNSW-NB15",),
        features_description="Гібридні ознаки: мережеві + контентні + часові.",
        supported_models=("Random Forest", "XGBoost"),
    ),
}


DATASET_TO_NATURE: dict[str, str] = {
    "CIC-IDS": NATURE_NETWORK_INTRUSION,
    "NSL-KDD": NATURE_CLASSIC_IDS,
    "UNSW-NB15": NATURE_MODERN_NETWORK,
}


def list_natures() -> list[NatureDefinition]:
    return list(NATURE_DEFINITIONS.values())


def get_nature(nature_id: str) -> NatureDefinition:
    try:
        return NATURE_DEFINITIONS[nature_id]
    except KeyError as exc:
        raise ValueError(f"Невідома природа датасету: {nature_id}") from exc


def nature_for_dataset(dataset_type: str | None) -> str | None:
    if not dataset_type:
        return None
    return DATASET_TO_NATURE.get(str(dataset_type).strip())


def nature_label(nature_id: str | None) -> str:
    if not nature_id:
        return "Невідомо"
    definition = NATURE_DEFINITIONS.get(nature_id)
    return definition.label if definition else "Невідомо"


def are_natures_compatible(model_nature: str | None, file_nature: str | None) -> bool:
    return bool(model_nature and file_nature and model_nature == file_nature)


def detect_nature_from_columns(columns: Iterable[object]) -> tuple[str | None, float]:
    normalized = {normalize_column_name(column) for column in columns}
    if not normalized:
        return None, 0.0

    best_dataset: str | None = None
    best_score = 0.0
    for dataset_type, schema in DATASET_SCHEMAS.items():
        markers = set(schema.detection_markers)
        if not markers:
            continue
        score = len(markers.intersection(normalized)) / len(markers)
        if score > best_score:
            best_score = float(score)
            best_dataset = dataset_type

    if best_dataset is None or best_score < 0.50:
        return None, float(best_score)

    return nature_for_dataset(best_dataset), float(best_score)


def describe_nature_mismatch(model_nature: str, file_nature: str) -> str:
    model_def = NATURE_DEFINITIONS.get(model_nature)
    file_def = NATURE_DEFINITIONS.get(file_nature)
    model_label = model_def.label if model_def else nature_label(model_nature)
    file_label = file_def.label if file_def else nature_label(file_nature)
    model_datasets = "/".join(model_def.dataset_display_names) if model_def else model_label
    file_datasets = "/".join(file_def.dataset_display_names) if file_def else file_label
    return (
        "УВАГА: Можлива несумісність моделі та даних\n\n"
        f"Завантажена модель навчалась на датасетах природи \"{model_label}\" "
        f"({model_datasets}).\n\n"
        f"Ваш файл, схоже, має структуру \"{file_label}\" ({file_datasets}).\n\n"
        "Це може призвести до:\n"
        "- Некоректних або хибних детекцій\n"
        "- Пропуску реальних атак\n"
        "- Неправильної класифікації трафіку\n\n"
        "Рекомендація: Навчіть окрему модель на датасетах відповідної природи."
    )


def supported_algorithms_for_nature(nature_id: str) -> tuple[str, ...]:
    return get_nature(nature_id).supported_models
