"""CHANGELOG 解析与展示测试。"""

from __future__ import annotations

from pi_coding_agent.changelog import (
    compare_versions,
    find_changelog_path,
    format_changelog,
    get_new_entries,
    parse_changelog,
)


def _write(tmp_path, text):
    path = tmp_path / "CHANGELOG.md"
    path.write_text(text, encoding="utf-8")
    return path


class TestParseChangelog:
    def test_parses_versioned_entries(self, tmp_path):
        path = _write(
            tmp_path,
            "# Changelog\n\n"
            "## [0.2.0]\n\n### Added\n\n- feature x\n\n"
            "## [0.1.0]\n\n### Fixed\n\n- bug y\n",
        )
        entries = parse_changelog(path)
        assert [entry.version for entry in entries] == ["0.2.0", "0.1.0"]
        assert "feature x" in entries[0].content
        assert "bug y" in entries[1].content

    def test_skips_unversioned_sections(self, tmp_path):
        path = _write(
            tmp_path,
            "## [Unreleased]\n\n### Added\n\n- wip\n\n## [0.1.0]\n\n- released\n",
        )
        entries = parse_changelog(path)
        assert [entry.version for entry in entries] == ["0.1.0"]

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_changelog(tmp_path / "nope.md") == []

    def test_invalid_content_returns_empty(self, tmp_path):
        path = _write(tmp_path, "no headers here")
        assert parse_changelog(path) == []


class TestVersionHelpers:
    def test_compare(self):
        from pi_coding_agent.changelog import ChangelogEntry

        older = ChangelogEntry(0, 1, 0, "")
        newer = ChangelogEntry(0, 2, 0, "")
        assert compare_versions(older, newer) < 0
        assert compare_versions(newer, older) > 0
        assert compare_versions(newer, newer) == 0

    def test_get_new_entries(self):
        from pi_coding_agent.changelog import ChangelogEntry

        entries = [
            ChangelogEntry(0, 1, 0, "a"),
            ChangelogEntry(0, 2, 0, "b"),
            ChangelogEntry(1, 0, 0, "c"),
        ]
        assert [entry.version for entry in get_new_entries(entries, "0.1.0")] == [
            "0.2.0",
            "1.0.0",
        ]
        assert get_new_entries(entries, "9.0.0") == []


class TestFindAndFormat:
    def test_find_from_cwd_ancestor(self, tmp_path):
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("## [0.1.0]\n\nx\n", encoding="utf-8")
        nested = tmp_path / "a" / "b"
        nested.mkdir(parents=True)
        assert find_changelog_path(nested) == changelog

    def test_format_orders_newest_first(self, tmp_path):
        from pi_coding_agent.changelog import ChangelogEntry

        entries = [
            ChangelogEntry(0, 1, 0, "## [0.1.0]\n\nold"),
            ChangelogEntry(0, 2, 0, "## [0.2.0]\n\nnew"),
        ]
        rendered = format_changelog(entries)
        assert rendered.index("0.2.0") < rendered.index("0.1.0")

    def test_format_limit(self):
        from pi_coding_agent.changelog import ChangelogEntry

        entries = [
            ChangelogEntry(0, 1, 0, "## [0.1.0]\n\na"),
            ChangelogEntry(0, 2, 0, "## [0.2.0]\n\nb"),
        ]
        assert "0.1.0" not in format_changelog(entries, limit=1)

    def test_format_empty(self):
        assert format_changelog([]) == "No changelog entries found."
