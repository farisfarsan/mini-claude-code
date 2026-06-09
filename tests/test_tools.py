import pytest

import agent_cli.tools as tools
from agent_cli.tools import WORKSPACE, _workspace_path, truncate_result


class TestWorkspacePath:
    def test_resolves_plain_filename(self):
        assert _workspace_path("hello.py") == f"{WORKSPACE}/hello.py"

    def test_strips_workspace_prefix(self):
        assert _workspace_path("workspace/hello.py") == _workspace_path("hello.py")

    def test_rejects_parent_traversal(self):
        with pytest.raises(ValueError):
            _workspace_path("../../etc/passwd")

    def test_rejects_single_parent(self):
        with pytest.raises(ValueError):
            _workspace_path("../outside.txt")


class TestTruncateResult:
    def test_short_text_unchanged(self):
        text = "hello world"
        assert truncate_result(text) == text

    def test_at_limit_unchanged(self):
        assert truncate_result("x" * 5000) == "x" * 5000

    def test_long_text_is_shorter(self):
        text = "A" * 3000 + "B" * 3000
        assert len(truncate_result(text)) < len(text)

    def test_head_and_tail_preserved(self):
        text = "A" * 3000 + "M" * 1000 + "B" * 3000
        result = truncate_result(text)
        assert result.startswith("A" * 2500)
        assert result.endswith("B" * 2500)

    def test_marker_shows_dropped_count(self):
        result = truncate_result("x" * 6000)
        assert "1000" in result
        assert "truncated" in result


class TestStrReplace:
    def test_not_found_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "WORKSPACE", str(tmp_path))
        (tmp_path / "f.txt").write_text("hello world")
        result = tools.str_replace("f.txt", "missing", "x")
        assert result.startswith("ERROR")
        assert (tmp_path / "f.txt").read_text() == "hello world"

    def test_multiple_matches_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "WORKSPACE", str(tmp_path))
        (tmp_path / "f.txt").write_text("foo foo foo")
        result = tools.str_replace("f.txt", "foo", "bar")
        assert result.startswith("ERROR")
        assert "3 times" in result
        assert (tmp_path / "f.txt").read_text() == "foo foo foo"

    def test_single_match_replaces(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "WORKSPACE", str(tmp_path))
        (tmp_path / "f.txt").write_text("hello world")
        result = tools.str_replace("f.txt", "world", "Python")
        assert "Successfully" in result
        assert (tmp_path / "f.txt").read_text() == "hello Python"

    def test_file_not_found_returns_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tools, "WORKSPACE", str(tmp_path))
        result = tools.str_replace("missing.txt", "old", "new")
        assert result.startswith("ERROR")
