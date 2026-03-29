"""Mount policy enforcement: blocks sensitive paths from being bind-mounted."""

from __future__ import annotations

import os
from pathlib import Path

# Subdirectories of $HOME that are always blocked
_BLOCKED_HOME_DIRS = {
    ".ssh",
    ".gnupg",
    ".aws",
    ".kube",
    ".docker",
    ".config/leuk",
    ".netrc",
    ".pgpass",
    ".git-credentials",
}

# Individual file names that are always blocked regardless of location
_BLOCKED_FILENAMES = {
    ".env",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "id_ecdsa_sk",
    "id_ed25519_sk",
}

# File suffixes that are always blocked
_BLOCKED_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".cer",
}


def _is_blocked(path: Path) -> bool:
    """Return True if path should never be mounted into a container."""
    home = Path.home()

    # Block by filename or suffix
    if path.name in _BLOCKED_FILENAMES:
        return True
    if path.suffix in _BLOCKED_SUFFIXES:
        return True

    # Block sensitive home subdirectories
    try:
        rel = path.relative_to(home)
        rel_str = str(rel)
        for blocked in _BLOCKED_HOME_DIRS:
            if rel_str == blocked or rel_str.startswith(blocked + os.sep):
                return True
    except ValueError:
        pass  # Not under home — path-level checks still apply

    return False


def validate_mounts(allowed_mounts: list[str]) -> list[tuple[str, str, bool]]:
    """Parse and validate a list of mount specifications.

    Each entry has the form ``host_path[:container_path][:rw]``.  The default
    access mode is **read-only** unless ``:rw`` is explicitly appended.

    Returns a list of ``(host_path, container_path, read_only)`` tuples.

    Raises ``ValueError`` if any mount targets a blocked path.
    """
    result: list[tuple[str, str, bool]] = []

    for spec in allowed_mounts:
        # Split on ":" but be careful with Windows-style absolute paths (not a
        # concern in practice for Docker hosts, but let's be safe).
        parts = spec.split(":")

        host_raw = parts[0]
        host_path = Path(host_raw).expanduser().resolve()

        if _is_blocked(host_path):
            raise ValueError(
                f"Mount refused — path contains sensitive credentials: {host_path}"
            )

        # Determine container path and read-only flag
        read_only = True
        container_path = str(host_path)

        if len(parts) >= 2:
            # Last token may be an access flag
            if parts[-1] in ("ro", "rw"):
                read_only = parts[-1] != "rw"
                if len(parts) >= 3:
                    container_path = parts[1]
                # else container_path stays as host_path (already set above)
            else:
                container_path = parts[1]

        result.append((str(host_path), container_path, read_only))

    return result
