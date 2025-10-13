import subprocess
from typing import NamedTuple, Optional

class GitInfo(NamedTuple):
    commit: str
    branch: str
    dirty: bool


def get_git_info() -> Optional[GitInfo]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).strip())
        return GitInfo(commit, branch, dirty)
    except Exception:
        return None