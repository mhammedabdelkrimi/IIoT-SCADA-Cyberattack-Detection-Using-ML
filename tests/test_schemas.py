import pytest
from pydantic import ValidationError

from src.api.main import Flow


def test_valid_payload_accepted():
    Flow(data={"Duration": 1.5, "Protocol": "TCP", "anomaly_alert": True, "Service": None})


def test_empty_data_rejected():
    with pytest.raises(ValidationError):
        Flow(data={})


def test_string_instead_of_dict_rejected():
    with pytest.raises(ValidationError):
        Flow(data="abc")


def test_list_value_rejected():
    with pytest.raises(ValidationError):
        Flow(data={"Duration": [1, 2]})


def test_missing_field_rejected():
    with pytest.raises(ValidationError):
        Flow()


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        Flow(data={"Duration": 1}, extra=1)