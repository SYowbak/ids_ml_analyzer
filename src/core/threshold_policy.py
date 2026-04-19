from __future__ import annotations

from typing import Any


THRESHOLD_POLICY_ID = "ids.threshold.policy.v1"
LEGACY_POLICY_ID = "ids.threshold.policy.legacy.v0"


def clamp_threshold(value: float, lower: float = 0.01, upper: float = 0.99) -> float:
    return float(min(max(float(value), float(lower)), float(upper)))


def _to_float_or_none(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None

def build_threshold_provenance(
    *,
    dataset_type: str,
    algorithm: str,
    recommended_threshold: float,
    recommended_metrics: dict[str, Any] | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_threshold = clamp_threshold(recommended_threshold)
    metadata = model_metadata or {}
    metrics = recommended_metrics or {}
    selection_policy = str(metrics.get("selection_policy") or "unspecified")

    thresholds_by_input_type: dict[str, float] = {}
    notes_by_input_type: dict[str, str] = {}

    if dataset_type == "CIC-IDS":
        if algorithm == "Random Forest":
            thresholds_by_input_type["pcap"] = clamp_threshold(max(base_threshold, 0.20))
            notes_by_input_type["pcap"] = "cic_pcap_calibrated_random_forest_floor"
            thresholds_by_input_type["csv"] = clamp_threshold(max(base_threshold, 0.30))
            notes_by_input_type["csv"] = "cic_csv_random_forest_guardrail_030"
        elif algorithm == "XGBoost":
            hard_case_sources = metadata.get("cic_hard_case_reference_sources")
            if isinstance(hard_case_sources, list) and hard_case_sources:
                thresholds_by_input_type["pcap"] = clamp_threshold(max(base_threshold, 0.20))
                notes_by_input_type["pcap"] = "cic_pcap_stability_xgboost_hard_case"
            else:
                thresholds_by_input_type["pcap"] = 0.05
                notes_by_input_type["pcap"] = "cic_pcap_floor_xgboost"
            if selection_policy == "holdout_rate_calibrated":
                thresholds_by_input_type["csv"] = clamp_threshold(base_threshold)
                notes_by_input_type["csv"] = "cic_csv_holdout_rate_calibrated"
            else:
                thresholds_by_input_type["csv"] = clamp_threshold(min(base_threshold, 0.02))
                notes_by_input_type["csv"] = "cic_csv_floor_rare_attack_coverage"
    elif dataset_type == "NSL-KDD":
        # NSL: баланс чутливості для уникнення FP-сплесків.
        thresholds_by_input_type["csv"] = 0.30
        notes_by_input_type["csv"] = "nsl_csv_balanced_guardrail_030"
    elif dataset_type == "UNSW-NB15":
        thresholds_by_input_type["csv"] = clamp_threshold(base_threshold)
        notes_by_input_type["csv"] = "unsw_csv_model_threshold"

    return {
        "policy_id": THRESHOLD_POLICY_ID,
        "policy_version": 1,
        "source": "training_calibration",
        "dataset_type": str(dataset_type),
        "algorithm": str(algorithm),
        "base_threshold": float(base_threshold),
        "thresholds_by_input_type": {
            key: float(clamp_threshold(value))
            for key, value in thresholds_by_input_type.items()
            if isinstance(value, (int, float))
        },
        "notes_by_input_type": {key: str(value) for key, value in notes_by_input_type.items()},
        "selection_policy": selection_policy,
    }

def _resolve_legacy_threshold(manifest: dict[str, Any], inspection: Any) -> tuple[float, str, dict[str, Any]]:
    metadata = manifest.get("metadata") or {}
    recommended_raw = metadata.get("recommended_threshold")
    threshold_value = _to_float_or_none(recommended_raw)
    if threshold_value is None:
        threshold_value = 0.30
    threshold_value = clamp_threshold(threshold_value)

    dataset_type = str(manifest.get("dataset_type") or "")
    algorithm = str(manifest.get("algorithm") or "")
    input_type = str(getattr(inspection, "input_type", "") or "")

    if input_type == "pcap" and dataset_type == "CIC-IDS":
        if algorithm == "Random Forest":
            adjusted = clamp_threshold(max(threshold_value, 0.20))
            return (
                adjusted,
                "Рекомендований поріг для цієї моделі: "
                f"{adjusted:.2f} (legacy PCAP calibrated floor для CIC Random Forest)",
                {
                    "policy_id": LEGACY_POLICY_ID,
                    "source": "legacy_runtime_override",
                    "rule": "cic_pcap_random_forest_calibrated_floor",
                    "input_type": input_type,
                    "threshold": float(adjusted),
                },
            )
        if algorithm == "XGBoost":
            hard_case_sources = metadata.get("cic_hard_case_reference_sources")
            if isinstance(hard_case_sources, list) and hard_case_sources:
                adjusted = clamp_threshold(max(threshold_value, 0.20))
                return (
                    adjusted,
                    "Рекомендований поріг для цієї моделі: "
                    f"{adjusted:.2f} (legacy PCAP stability override для CIC XGBoost)",
                    {
                        "policy_id": LEGACY_POLICY_ID,
                        "source": "legacy_runtime_override",
                        "rule": "cic_pcap_xgboost_hard_case",
                        "input_type": input_type,
                        "threshold": float(adjusted),
                    },
                )
            return (
                0.05,
                "Рекомендований поріг для цієї моделі: 0.05 "
                "(legacy PCAP override для CIC XGBoost)",
                {
                    "policy_id": LEGACY_POLICY_ID,
                    "source": "legacy_runtime_override",
                    "rule": "cic_pcap_xgboost",
                    "input_type": input_type,
                    "threshold": 0.05,
                },
            )

    if input_type == "csv":
        if dataset_type == "CIC-IDS" and algorithm == "Random Forest":
            adjusted = clamp_threshold(max(threshold_value, 0.30))
            return (
                adjusted,
                "Рекомендований поріг для цієї моделі: "
                f"{adjusted:.2f} (legacy CIC RF CSV guardrail)",
                {
                    "policy_id": LEGACY_POLICY_ID,
                    "source": "legacy_runtime_override",
                    "rule": "cic_csv_random_forest_guardrail_030",
                    "input_type": input_type,
                    "threshold": float(adjusted),
                },
            )
        if dataset_type == "CIC-IDS" and algorithm == "XGBoost":
            adjusted = clamp_threshold(min(threshold_value, 0.02))
            return (
                adjusted,
                "Рекомендований поріг для цієї моделі: "
                f"{adjusted:.2f} (legacy CIC XGBoost CSV override)",
                {
                    "policy_id": LEGACY_POLICY_ID,
                    "source": "legacy_runtime_override",
                    "rule": "cic_csv_xgboost",
                    "input_type": input_type,
                    "threshold": float(adjusted),
                },
            )
        if dataset_type == "NSL-KDD":
            adjusted = 0.30
            return (
                adjusted,
                "Рекомендований поріг для цієї моделі: "
                f"{adjusted:.2f} (legacy NSL CSV guardrail)",
                {
                    "policy_id": LEGACY_POLICY_ID,
                    "source": "legacy_runtime_override",
                    "rule": "nsl_csv_guardrail_030",
                    "input_type": input_type,
                    "threshold": float(adjusted),
                },
            )
        if dataset_type == "UNSW-NB15":
            model_name = str(manifest.get("name") or "").lower()
            if "seed_balanced" in model_name:
                adjusted = clamp_threshold(min(threshold_value, 0.20))
                return (
                    adjusted,
                    "Рекомендований поріг для цієї моделі: "
                    f"{adjusted:.2f} (legacy UNSW seed_balanced override)",
                    {
                        "policy_id": LEGACY_POLICY_ID,
                        "source": "legacy_runtime_override",
                        "rule": "unsw_csv_seed_balanced",
                        "input_type": input_type,
                        "threshold": float(adjusted),
                    },
                )
            adjusted = clamp_threshold(threshold_value)
            return (
                adjusted,
                "Рекомендований поріг для цієї моделі: "
                f"{adjusted:.2f} (legacy UNSW metadata threshold)",
                {
                    "policy_id": LEGACY_POLICY_ID,
                    "source": "legacy_metadata",
                    "rule": "unsw_csv_metadata",
                    "input_type": input_type,
                    "threshold": float(adjusted),
                },
            )

    return (
        threshold_value,
        f"Рекомендований поріг для цієї моделі: {threshold_value:.2f} (legacy metadata threshold)",
        {
            "policy_id": LEGACY_POLICY_ID,
            "source": "legacy_metadata",
            "rule": "default",
            "input_type": input_type or "unknown",
            "threshold": float(threshold_value),
        },
    )

def resolve_threshold_for_scan(
    manifest: dict[str, Any],
    inspection: Any,
) -> tuple[float, str, dict[str, Any]]:
    algorithm = str(manifest.get("algorithm") or "")
    dataset_type = str(manifest.get("dataset_type") or "")
    metadata = manifest.get("metadata") or {}
    provenance = metadata.get("threshold_provenance")
    input_type = str(getattr(inspection, "input_type", "") or "")

    if isinstance(provenance, dict) and str(provenance.get("policy_id") or "") == THRESHOLD_POLICY_ID:
        base_threshold = _to_float_or_none(provenance.get("base_threshold"))
        if base_threshold is None:
            base_threshold = _to_float_or_none(metadata.get("recommended_threshold"))
        if base_threshold is None:
            base_threshold = 0.30

        resolved = clamp_threshold(base_threshold)
        selected_rule = "base_threshold"
        notes = provenance.get("notes_by_input_type")
        by_input = provenance.get("thresholds_by_input_type")
        if isinstance(by_input, dict) and input_type:
            candidate = _to_float_or_none(by_input.get(input_type))
            if candidate is not None:
                resolved = clamp_threshold(candidate)
                selected_rule = str(input_type)
        note = ""
        if isinstance(notes, dict):
            note = str(notes.get(selected_rule) or notes.get(input_type) or "")

        if (
            input_type == "pcap"
            and dataset_type == "CIC-IDS"
            and algorithm == "Random Forest"
            and selected_rule == "pcap"
            and float(resolved) <= 0.20
            and float(base_threshold) > float(resolved)
        ):
            resolved = clamp_threshold(max(float(base_threshold), 0.20))
            selected_rule = "pcap_calibrated_floor"
            note = "runtime_upgrade_from_static_pcap_floor"

        if (
            input_type == "csv"
            and dataset_type == "NSL-KDD"
            and float(resolved) < 0.30
        ):
            resolved = 0.30
            selected_rule = "nsl_csv_runtime_guardrail_030"
            note = "runtime_upgrade_from_low_nsl_threshold"

        if (
            input_type == "csv"
            and dataset_type == "CIC-IDS"
            and algorithm == "Random Forest"
            and float(resolved) < 0.30
        ):
            resolved = 0.30
            selected_rule = "cic_csv_rf_runtime_guardrail_030"
            note = "runtime_upgrade_from_low_cic_rf_threshold"

        caption = (
            f"Рекомендований поріг для цієї моделі: {resolved:.2f} "
            f"(policy={THRESHOLD_POLICY_ID}, rule={selected_rule or 'default'})"
        )
        return (
            resolved,
            caption,
            {
                "policy_id": THRESHOLD_POLICY_ID,
                "source": "training_provenance",
                "input_type": input_type or "unknown",
                "rule": selected_rule,
                "note": note,
                "threshold": float(resolved),
            },
        )

    return _resolve_legacy_threshold(manifest=manifest, inspection=inspection)
