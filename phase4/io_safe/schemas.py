from typing import Dict, Iterable


REQUIRED_FIELDS = {
    "doc_id",
    "split",
    "subset_domain",
    "sample_id",
    "input_noisy",
    "prediction",
    "changed_flag",
}


def validate_prediction_record(record: Dict[str, object], *, blind_test: bool) -> None:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"Prediction record missing fields: {sorted(missing)}")
    if blind_test and "test_metric" in record:
        raise ValueError("Blind test record must not include derived test metrics.")


def validate_prediction_records(records: Iterable[Dict[str, object]], *, blind_test: bool) -> None:
    for record in records:
        validate_prediction_record(record, blind_test=blind_test)

