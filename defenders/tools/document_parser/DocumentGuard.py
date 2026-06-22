import os
import re
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

ALLOWED_INPUT_DIR = _PROJECT_ROOT / "documents" / "input"
ALLOWED_OUTPUT_DIR = _PROJECT_ROOT / "documents" / "output"

_ALPHANUMERIC_PDF = re.compile(r'^[\w\-]+\.pdf$', re.IGNORECASE)

_SENSITIVE = [
    r'/etc/', r'/sys/', r'/proc/', r'/root/', r'\.ssh',
    r'\.env', r'passwd', r'shadow', r'id_rsa', r'\.pem',
    r'\.key', r'config\.', r'secrets\.', r'credentials',
    r'windows[/\\]system32',
]

def _normalize_ext(file_path: str) -> tuple[str, str | None]:
    suffix = Path(file_path).suffix.lower()
    if suffix == "":
        return file_path + ".pdf", None
    if suffix == ".pdf":
        return file_path, None
    return file_path, f"only .pdf files are allowed. i got '{suffix}' remove the extension and try again"


class DocumentGuard:

    def check_read(self, file_path: str) -> tuple[bool, str, str]:
        file_path, ext_err = _normalize_ext(file_path)
        if ext_err:
            return False, ext_err, file_path

        blocked, reason = self._check_path_rules(file_path, ALLOWED_INPUT_DIR)
        if blocked:
            return False, reason, file_path

        resolved = (ALLOWED_INPUT_DIR / file_path).resolve()
        if not resolved.exists():
            return False, f"'{file_path}' was not found in the input directory", file_path
        if resolved.stat().st_size > 10_000_000:
            return False, "file exceeds 10 MB size cap", file_path

        return True, "ok", file_path

    def check_write(self, file_path: str, documentation: str) -> tuple[bool, str, str]:
        file_path, ext_err = _normalize_ext(file_path)
        if ext_err:
            return False, ext_err, file_path

        blocked, reason = self._check_path_rules(file_path, ALLOWED_OUTPUT_DIR)
        if blocked:
            return False, reason, file_path

        resolved = (ALLOWED_OUTPUT_DIR / file_path).resolve()

        if resolved.exists():
            return False, "file already exists! overwriting is not allowed!", file_path

        if not str(resolved).startswith(str(ALLOWED_OUTPUT_DIR.resolve())):
            return False, "path escapes the allowed output directory", file_path

        if str(resolved).startswith(str(ALLOWED_INPUT_DIR.resolve())):
            return False, "cannot write into the input directory", file_path

        if len(documentation) > 50_000:
            return False, f"documentation content exceeds 50 000 character cap ({len(documentation)} chars)", file_path

        if any(ord(c) < 32 and c not in '\n\r\t' for c in documentation):
            return False, "non-printable characters detected in content", file_path

        return True, "ok", file_path

    def _check_path_rules(self, file_path: str, allowed_dir: Path) -> tuple[bool, str]:
        if "../" in file_path or "..\\" in file_path:
            return True, "path traversal sequence detected"

        if os.path.isabs(file_path):
            return True, "absolute paths are not allowed! use a filename only!"

        for pattern in _SENSITIVE:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True, f"sensitive path pattern detected: {pattern}"

        name = Path(file_path).name
        if name.startswith("."):
            return True, "hidden files are not allowed!"

        if not _ALPHANUMERIC_PDF.match(name):
            return True, f"filename '{name}' must be alphanumeric with hyphens only (like my-doc.pdf)"

        resolved = (allowed_dir / file_path).resolve()
        if not str(resolved).startswith(str(allowed_dir.resolve())):
            return True, "resolved path escapes the allowed directory"

        return False, "ok"
