import pytest

from src.predict_readmission import predict_readmission


def test_prediction_rejects_non_mapping_input():
    with pytest.raises(TypeError, match="mapping"):
        predict_readmission(None)


def test_prediction_handles_missing_features_with_production_defaults():
    result = predict_readmission({"age_at_admission": 67, "length_of_stay": 5.0})
    assert 0 <= result["probability"] <= 1
    assert result["normalized_input"]["age_at_admission"] == 67
