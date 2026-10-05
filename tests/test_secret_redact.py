from aura_core.memory.secret_redact import MARK, has_secret, mask_env_values, redact


def _fake(prefix: str, body: str) -> str:
  # Built at runtime so no secret-shaped literal sits in the repository.
  return prefix + body


def test_prefixed_tokens_are_masked():
  for token in (_fake("sk-", "A1b2C3d4E5f6G7h8I9j0"), _fake("ghp_", "A1b2C3d4E5f6G7h8I9j0K1l2"),
                _fake("AKIA", "ABCDEFGHIJKLMNOP")):
    out = redact(f"value is {token} here")
    assert token not in out and MARK in out


def test_assignments_and_bearer_and_url_credentials():
  assert "hunter22secret" not in redact("password=hunter22secret")
  assert "hunter22secret" not in redact('API_TOKEN: "hunter22secret"')
  assert "abcdef0123456789" not in redact("Authorization: Bearer abcdef0123456789")
  assert "s3cretpw" not in redact("postgres://app:s3cretpw@db.internal/x")
  assert "app:" in redact("postgres://app:s3cretpw@db.internal/x")  # structure kept


def test_pem_block_masked():
  pem = "-----BEGIN " + "PRIVATE KEY-----\nMIIabc\n-----END " + "PRIVATE KEY-----"
  assert "MIIabc" not in redact(f"before\n{pem}\nafter")


def test_references_and_prose_are_left_alone():
  for text in ("token=$TOKEN", "key=<your key>", "the token is stored in a vault",
               "password policy: rotate quarterly", "monkey=banana"):
    assert redact(text) == text, text


def test_redact_is_idempotent():
  once = redact("password=hunter22secret")
  assert redact(once) == once
  assert has_secret("password=hunter22secret") and not has_secret("plain text")


def test_exact_value_masking_from_env_file(tmp_path, monkeypatch):
  value = "v4lue-" + "x9Zq81LmN2"
  env = tmp_path / "local.env"
  env.write_text(f"# comment\nSERVICE_TOKEN={value}\nPLAIN=hello\nSHORT_KEY=abc\nexport DB_PASSWORD='{value}2'\n")
  monkeypatch.setenv("AURA_CORE_ENV_FILE", str(env))
  text, n = mask_env_values(f"a {value} b {value}2 c hello abc")
  assert n == 2 and value not in text and "hello abc" in text  # non-credential and short values untouched


def test_no_env_file_means_no_masking():
  assert mask_env_values("anything") == ("anything", 0)
