from __future__ import annotations

import json

import pytest
from common import PROJECT_DIR, build_project_cli_parser


def test_cli_help_exits_cleanly():
    parser, _ = build_project_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


@pytest.mark.parametrize(
    "path", sorted((PROJECT_DIR / "configs").glob("*.json")), ids=lambda p: p.name
)
def test_configs_are_json_with_description(path):
    cfg = json.loads(path.read_text())
    assert cfg.get("description")
