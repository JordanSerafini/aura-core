"""`aura-core-heartbeat init` : chaque modele s'ecrit, se relit sans avertissement, ne s'ecrase pas."""

import pytest

from aura_core.heartbeat.cli import main
from aura_core.heartbeat.parser import parse_heartbeat
from aura_core.heartbeat.templates import TEMPLATES


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_parses_cleanly(name, tmp_path):
  target = tmp_path / "HEARTBEAT.md"
  assert main(["--file", str(target), "init", "--template", name]) == 0
  tasks = parse_heartbeat(target)
  assert len(tasks) >= 1
  assert not tasks.errors


def test_init_refuses_to_overwrite_without_force(tmp_path, capsys):
  target = tmp_path / "HEARTBEAT.md"
  target.write_text("keep me", encoding="utf-8")
  assert main(["--file", str(target), "init"]) == 2
  assert target.read_text(encoding="utf-8") == "keep me"
  assert "already exists" in capsys.readouterr().err
  assert main(["--file", str(target), "init", "--force"]) == 0
  assert target.read_text(encoding="utf-8") == TEMPLATES["minimal"]
