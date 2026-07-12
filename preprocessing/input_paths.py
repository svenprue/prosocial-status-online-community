"""Resolve data/input parquet paths, preferring *_revision files when present."""
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = _PROJECT_ROOT / "data" / "input"

# Base stem -> preferred filename (revision first, then canonical)
_INPUT_FILES = {
    "posts_questions": ("posts_questions_revision.parquet", "posts_questions.parquet"),
    "posts_answers": ("posts_answers_revision.parquet", "posts_answers.parquet"),
    "users": ("Users_revision.parquet", "Users.parquet"),
    "votes": ("Votes_revision.parquet", "Votes.parquet"),
    "comments": ("Comments_revision.parquet", "Comments.parquet"),
    "posthistory": ("PostHistory_revision.parquet", "PostHistory.parquet"),
    "badges": ("Badges.parquet", "Badges.parquet"),
}


def resolve_input_file(key: str) -> str:
    """Return absolute path string for a logical input key."""
    if key not in _INPUT_FILES:
        raise KeyError(f"Unknown input key: {key}. Known: {sorted(_INPUT_FILES)}")
    revision_name, canonical_name = _INPUT_FILES[key]
    revision_path = INPUT_DIR / revision_name
    if revision_path.exists():
        return str(revision_path)
    canonical_path = INPUT_DIR / canonical_name
    if canonical_path.exists():
        return str(canonical_path)
    raise FileNotFoundError(
        f"No input file for '{key}' in {INPUT_DIR} "
        f"(tried {revision_name}, {canonical_name})"
    )


def input_file_exists(key: str) -> bool:
    try:
        resolve_input_file(key)
        return True
    except FileNotFoundError:
        return False
