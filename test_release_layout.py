from pathlib import Path


ROOT = Path(__file__).parent
LEGACY_PACKAGE = "PE" + "MA"
OLD_PACKAGE = "phyrc" + "_gzsl"
ROOT_ENTRIES = {"baseline", "checkpoints", "configs", "data", "models", "phyrc", "tests"}
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".ps1"}
FORBIDDEN_SUFFIXES = {".mat", ".pt", ".pth", ".pyc", ".log"}


def test_release_layout():
    assert not (ROOT / OLD_PACKAGE).exists()
    assert ROOT_ENTRIES <= {path.name for path in ROOT.iterdir()}
    assert not (ROOT / LEGACY_PACKAGE).exists()

    files = [path for path in ROOT.rglob("*") if path.is_file()]
    assert not [path for path in files if path.suffix.lower() in FORBIDDEN_SUFFIXES]
    assert not [path for path in files if path.stat().st_size >= 100 * 1024 * 1024]

    stale = []
    for path in files:
        if path.is_relative_to(ROOT / "docs" / "superpowers"):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if LEGACY_PACKAGE in text:
                stale.append(path.relative_to(ROOT))
            if OLD_PACKAGE in text:
                stale.append(path.relative_to(ROOT))
    assert not stale, stale


if __name__ == "__main__":
    test_release_layout()
