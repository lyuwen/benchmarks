from benchmarks.scaleswe_interactive.file_reference import (
    extract_file_paths,
    inject_files,
)


def test_extract_paths_from_prose_and_backticks_and_lineref():
    text = (
        "The bug is in src/pkg/mod.py and also `lib/util.js`.\n"
        "See tests/test_x.py:42 for the failing case. Ignore http://x.com/a.py"
    )
    paths = extract_file_paths(text)
    assert "src/pkg/mod.py" in paths
    assert "lib/util.js" in paths
    assert "tests/test_x.py" in paths          # line suffix stripped
    assert all(not p.startswith("http") for p in paths)
    # order-preserving, de-duplicated
    assert paths == list(dict.fromkeys(paths))


def test_github_blob_url_is_not_extracted():
    text = "See https://github.com/org/repo/blob/main/src/foo.py for context"
    paths = extract_file_paths(text)
    assert "com/org/repo/blob/main/src/foo.py" not in paths
    assert all(not p.endswith("foo.py") for p in paths)


def test_plain_http_url_yields_no_paths():
    assert extract_file_paths("http://example.com/a/b/c.py") == []


def test_version_string_yields_no_paths():
    assert extract_file_paths("upgrade to 1.2.3 today") == []


def test_dotted_attribute_without_slash_yields_no_paths():
    assert extract_file_paths("call module.attr here") == []


class _FakeWS:
    def __init__(self, files):
        self._files = files

    def execute_command(self, cmd):
        # cmd looks like: cat -- '<repo>/<path>'
        class R:  # minimal CommandResult stand-in
            pass
        r = R()
        for path, content in self._files.items():
            if path in cmd:
                r.exit_code, r.stdout, r.stderr = 0, content, ""
                return r
        r.exit_code, r.stdout, r.stderr = 1, "", "No such file"
        return r


def test_inject_files_respects_caps_and_marks_missing():
    ws = _FakeWS({"a.py": "print(1)\n", "big.py": "x" * 100})
    out = inject_files(ws, "/repo", ["a.py", "big.py", "missing.py"],
                       max_files=5, max_bytes=10)
    by_path = {o["path"]: o for o in out}
    assert by_path["a.py"]["content"] == "print(1)\n"
    assert by_path["a.py"]["skipped"] is None
    assert by_path["big.py"]["skipped"] == "too_large"
    assert by_path["missing.py"]["skipped"] == "not_found"


def test_inject_files_caps_file_count():
    ws = _FakeWS({f"f{i}.py": "y" for i in range(10)})
    out = inject_files(ws, "/repo", [f"f{i}.py" for i in range(10)],
                       max_files=3, max_bytes=1000)
    injected = [o for o in out if o["skipped"] is None]
    assert len(injected) == 3
