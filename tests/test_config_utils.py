import pytest

from ragbench_core.utils.config import parse_yaml_text


def test_parse_yaml_text_accepts_mapping():
    assert parse_yaml_text('experiment:\n  name: ui_edit\n') == {'experiment': {'name': 'ui_edit'}}


def test_parse_yaml_text_rejects_non_mapping_root():
    with pytest.raises(ValueError, match='mapping'):
        parse_yaml_text('- not\n- a\n- config\n')
