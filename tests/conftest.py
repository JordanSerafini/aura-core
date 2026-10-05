import os

import pytest

# Keep tests off the network and off the user's real state.
os.environ.setdefault("AURA_CORE_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
  monkeypatch.setenv("AURA_CORE_HOME", str(tmp_path / "home"))
  monkeypatch.delenv("AURA_CORE_ENV_FILE", raising=False)
