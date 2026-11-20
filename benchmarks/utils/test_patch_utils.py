from benchmarks.utils.patch_utils import remove_binary_diffs, remove_files_from_patch


PATCH = """diff --git a/src/app.go b/src/app.go
--- a/src/app.go
+++ b/src/app.go
@@ -1,1 +1,1 @@
-old
+new
diff --git a/go1.23.0.linux-amd64.tar.gz b/go1.23.0.linux-amd64.tar.gz
new file mode 100644
index 0000000..1234567
Binary files /dev/null and b/go1.23.0.linux-amd64.tar.gz differ
"""


def test_remove_files_from_patch_drops_matching_files() -> None:
    patched = remove_files_from_patch(PATCH, {"go1.23.0.linux-amd64.tar.gz"})

    assert "src/app.go" in patched
    assert "go1.23.0.linux-amd64.tar.gz" not in patched


def test_remove_files_from_patch_keeps_patch_without_matches() -> None:
    patched = remove_files_from_patch(PATCH, {"unrelated.txt"})

    assert "src/app.go" in patched
    assert "go1.23.0.linux-amd64.tar.gz" in patched


def test_remove_binary_diffs_drops_only_binary_blocks() -> None:
    patched = remove_binary_diffs(PATCH)

    assert "src/app.go" in patched
    assert "go1.23.0.linux-amd64.tar.gz" not in patched


def test_remove_binary_diffs_preserves_trailing_newline() -> None:
    patched = remove_binary_diffs(PATCH)

    assert patched.endswith("\n")


def test_remove_binary_diffs_handles_git_binary_patch() -> None:
    patch = PATCH.replace(
        "Binary files /dev/null and b/go1.23.0.linux-amd64.tar.gz differ",
        "GIT binary patch\nliteral 0\nHcmV?d00001\n",
    )

    patched = remove_binary_diffs(patch)

    assert "src/app.go" in patched
    assert "go1.23.0.linux-amd64.tar.gz" not in patched
