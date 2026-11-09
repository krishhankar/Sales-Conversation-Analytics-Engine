import pytest

from scripts.evaluate import compute_metrics


def test_compute_metrics_includes_binary_and_per_class_metrics():
    metrics = compute_metrics([1, 1, 0, 0], [1, 0, 1, 0])

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["balanced_accuracy"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["macro_f1"] == 0.5
    assert metrics["mcc"] == 0.0
    assert metrics["per_class"]["conversion"]["support"] == 2
    assert metrics["per_class"]["no_conversion"]["support"] == 2


def test_compute_metrics_handles_empty_inputs():
    metrics = compute_metrics([], [])

    assert metrics["n"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["per_class"]["conversion"]["support"] == 0


def test_compute_metrics_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="equal lengths"):
        compute_metrics([1], [])


def test_compute_metrics_rejects_non_binary_values():
    with pytest.raises(ValueError, match="only 0 or 1"):
        compute_metrics([2], [1])
