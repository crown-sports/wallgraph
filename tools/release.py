"""Export an explicit public source allowlist; never archive the parent repository."""

import argparse
import re
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = {
    "README.md",
    "README.zh-CN.md",
    "LICENSE",
    "NOTICE.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "CITATION.cff",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "pyproject.toml",
    "MANIFEST.in",
    ".gitignore",
}
PATTERNS = (
    r"\b(?:10|192\.168)\.\d+\.\d+(?:\.\d+)?\b",
    r"\b172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+\b",
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
    r"(?i)(?:password|passwd|api[_-]?key|access[_-]?secret)\s*[=:]\s*['\"]?\w{6,}",
)


def public_files() -> list[Path]:
    files = [ROOT / name for name in sorted(TOP_LEVEL)]
    for directory, suffixes in (
        ("src", {".py"}),
        ("tests", {".py"}),
        ("tools", {".py"}),
        ("docs", {".md"}),
        (".github/workflows", {".yml", ".yaml"}),
        (".github/ISSUE_TEMPLATE", {".yml", ".yaml"}),
        ("examples", {".png"}),
    ):
        for path in sorted((ROOT / directory).rglob("*")):
            if (
                "__pycache__" in path.parts
                or path.suffix == ".pyc"
                or any(part.endswith(".egg-info") for part in path.parts)
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"symlink in public directory: {path.relative_to(ROOT)}")
            if path.is_file():
                if path.suffix not in suffixes:
                    raise ValueError(f"unexpected public file: {path.relative_to(ROOT)}")
                files.append(path)
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError("required public file missing or symlinked")
        if path.suffix == ".png":
            if path.relative_to(ROOT).as_posix() != "examples/simple.png":
                raise ValueError("only the single reviewed demo image can be published")
            with Image.open(path) as image:
                if max(image.size) > 2048:
                    raise ValueError("demo image is larger than the public limit")
                if image.format != "PNG" or image.mode != "RGB" or image.info:
                    raise ValueError("demo image must be RGB PNG with no metadata")
                image.load()
        else:
            source = path.read_text(encoding="utf-8")
            if any(re.search(pattern, source) for pattern in PATTERNS):
                raise ValueError(f"private-looking content: {path.relative_to(ROOT)}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.check and args.output is None:
        parser.error("--check or --output required")
    files = public_files()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}")
    print(f"Public source check passed: {len(files)} files")


if __name__ == "__main__":
    main()
