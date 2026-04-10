from types import SimpleNamespace

from src.core.threshold_policy import (
    THRESHOLD_POLICY_ID,
    build_threshold_provenance,
    resolve_threshold_for_scan,
)


def test_build_threshold_provenance_for_cic_random_forest_has_input_specific_thresholds() -> None:
    provenance = build_threshold_provenance(
        dataset_type="CIC-IDS",
        algorithm="Random Forest",
        recommended_threshold=0.82,
        recommended_metrics={"selection_policy": "fpr<=1%"},
    )

    assert provenance["policy_id"] == THRESHOLD_POLICY_ID
    assert float(provenance["base_threshold"]) == 0.82
    assert float(provenance["thresholds_by_input_type"]["pcap"]) == 0.82
    assert float(provenance["thresholds_by_input_type"]["csv"]) == 0.02


def test_resolve_threshold_for_scan_prefers_training_provenance_over_legacy_rules() -> None:
    manifest = {
        "name": "cic_ids_random_forest_test.joblib",
        "algorithm": "Random Forest",
        "dataset_type": "CIC-IDS",
        "metadata": {
            "recommended_threshold": 0.82,
            "threshold_provenance": {
                "policy_id": THRESHOLD_POLICY_ID,
                "policy_version": 1,
                "base_threshold": 0.82,
                "thresholds_by_input_type": {"pcap": 0.23, "csv": 0.04},
                "notes_by_input_type": {"pcap": "explicit_training_rule"},
            },
        },
    }

    threshold, caption, details = resolve_threshold_for_scan(
        manifest=manifest,
        inspection=SimpleNamespace(input_type="pcap"),
    )

    assert threshold == 0.23
    assert "policy=ids.threshold.policy.v1" in caption
    assert details["source"] == "training_provenance"


def test_resolve_threshold_for_scan_upgrades_old_static_rf_pcap_rule_to_calibrated_base() -> None:
    manifest = {
        "name": "cic_ids_random_forest_old.joblib",
        "algorithm": "Random Forest",
        "dataset_type": "CIC-IDS",
        "metadata": {
            "recommended_threshold": 0.39,
            "threshold_provenance": {
                "policy_id": THRESHOLD_POLICY_ID,
                "policy_version": 1,
                "base_threshold": 0.39,
                "thresholds_by_input_type": {"pcap": 0.20, "csv": 0.02},
                "notes_by_input_type": {"pcap": "cic_pcap_floor_random_forest"},
            },
        },
    }

    threshold, _caption, details = resolve_threshold_for_scan(
        manifest=manifest,
        inspection=SimpleNamespace(input_type="pcap"),
    )

    assert threshold == 0.39
    assert details["rule"] == "pcap_calibrated_floor"
    assert details["source"] == "training_provenance"


def test_resolve_threshold_for_scan_keeps_legacy_behavior_for_old_models() -> None:
    legacy_manifest = {
        "name": "legacy_rf.joblib",
        "algorithm": "Random Forest",
        "dataset_type": "CIC-IDS",
        "metadata": {"recommended_threshold": 0.85},
    }

    threshold, caption, details = resolve_threshold_for_scan(
        manifest=legacy_manifest,
        inspection=SimpleNamespace(input_type="pcap"),
    )

    assert threshold == 0.85
    assert "legacy" in caption.lower()
    assert details["policy_id"].endswith("legacy.v0")
