import utils.storage.migrations as m


def test_cli_migrate_and_version(tmp_path, monkeypatch, capsys):
    import utils.storage as storage_mod
    monkeypatch.setattr(storage_mod, "DEFAULT_DB_PATH", str(tmp_path / "races.db"))
    assert m._main(["migrate"]) == 0
    out = capsys.readouterr().out
    assert "migrated to version" in out
    assert m._main(["version"]) == 0
    assert "target=" in capsys.readouterr().out
