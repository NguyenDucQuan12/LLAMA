from __future__ import annotations

"""
Kiểm tra registry SQL chỉ chứa query read-only và validate parameter đúng.
"""

import pytest

from app.sql.query_registry import PredefinedSqlQueryRegistry


def test_registry_contains_expected_queries() -> None:
    registry = PredefinedSqlQueryRegistry()

    assert "agv_tasks_today" in registry.keys()
    assert "pallet_by_location" in registry.keys()
    assert "fabric_roll_by_qr" in registry.keys()


def test_pallet_query_requires_location_code() -> None:
    registry = PredefinedSqlQueryRegistry()
    query_definition = registry.get("pallet_by_location")

    missing_parameters = query_definition.missing_required_parameters({})

    assert missing_parameters == ["location_code"]


def test_pallet_query_rejects_unexpected_parameter() -> None:
    registry = PredefinedSqlQueryRegistry()
    query_definition = registry.get("pallet_by_location")

    with pytest.raises(ValueError):
        query_definition.validate_parameters(
            {
                "location_code": "F3-29",
                "unexpected": "value",
            }
        )


def test_pallet_query_accepts_valid_location_code() -> None:
    registry = PredefinedSqlQueryRegistry()
    query_definition = registry.get("pallet_by_location")

    validated_parameters = query_definition.validate_parameters(
        {"location_code": "F3-29"}
    )

    assert validated_parameters == {"location_code": "F3-29"}
