"""CSV manifest loading and leakage checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "image_path",
    "product_code",
    "model_family",
    "physical_part_id",
    "capture_session",
    "split",
    "orientation",
    "source",
}
VALID_SPLITS = {"gallery", "query", "unknown"}


@dataclass(frozen=True)
class ManifestEntry:
    image_path: Path
    product_code: str
    model_family: str = ""
    physical_part_id: str = ""
    capture_session: str = ""
    split: str = "query"
    orientation: str = ""
    source: str = "local"

    def as_metadata(self) -> dict[str, str]:
        return {
            "image_path": str(self.image_path),
            "product_code": self.product_code,
            "model_family": self.model_family,
            "physical_part_id": self.physical_part_id,
            "capture_session": self.capture_session,
            "split": self.split,
            "orientation": self.orientation,
            "source": self.source,
        }


def _resolve_image_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path.strip()).expanduser()
    if path.is_absolute():
        return path.resolve()

    # A manifest normally contains paths relative to the project working dir.
    cwd_candidate = (Path.cwd() / path).resolve()
    manifest_candidate = (manifest_path.parent / path).resolve()
    return cwd_candidate if cwd_candidate.exists() else manifest_candidate


def load_manifest(
    manifest_path: str | Path,
    *,
    split: str | None = None,
    check_files: bool = True,
) -> list[ManifestEntry]:
    """Load and validate a UTF-8 CSV manifest.

    Empty rows are ignored. Relative image paths are first resolved from the
    current working directory and then from the manifest directory.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    if split is not None and split not in VALID_SPLITS:
        raise ValueError(f"Invalid split {split!r}; expected one of {sorted(VALID_SPLITS)}")

    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")

        for line_number, row in enumerate(reader, start=2):
            if not any((value or "").strip() for value in row.values()):
                continue
            product_code = (row["product_code"] or "").strip()
            row_split = (row["split"] or "").strip().lower()
            raw_image_path = (row["image_path"] or "").strip()
            if not product_code:
                raise ValueError(f"Line {line_number}: product_code is empty")
            if row_split not in VALID_SPLITS:
                raise ValueError(f"Line {line_number}: invalid split {row_split!r}")
            if not raw_image_path:
                raise ValueError(f"Line {line_number}: image_path is empty")

            image_path = _resolve_image_path(raw_image_path, path)
            if check_files and not image_path.is_file():
                raise FileNotFoundError(f"Line {line_number}: image does not exist: {image_path}")
            if split is None or row_split == split:
                entries.append(
                    ManifestEntry(
                        image_path=image_path,
                        product_code=product_code,
                        model_family=(row["model_family"] or "").strip(),
                        physical_part_id=(row["physical_part_id"] or "").strip(),
                        capture_session=(row["capture_session"] or "").strip(),
                        split=row_split,
                        orientation=(row["orientation"] or "").strip(),
                        source=(row["source"] or "").strip(),
                    )
                )
    return entries


def validate_independent_splits(entries: Iterable[ManifestEntry]) -> list[str]:
    """Return warnings for physical-part/session leakage across gallery/query."""

    locations: dict[tuple[str, str, str], set[str]] = {}
    paths: dict[Path, set[str]] = {}
    for entry in entries:
        paths.setdefault(entry.image_path, set()).add(entry.split)
        if entry.physical_part_id and entry.capture_session:
            key = (entry.product_code, entry.physical_part_id, entry.capture_session)
            locations.setdefault(key, set()).add(entry.split)

    warnings: list[str] = []
    for image_path, splits in paths.items():
        if "gallery" in splits and "query" in splits:
            warnings.append(f"Same image is in gallery and query: {image_path}")
    for key, splits in locations.items():
        if "gallery" in splits and "query" in splits:
            code, part, session = key
            warnings.append(
                f"Possible leakage for code={code}, physical_part_id={part}, "
                f"capture_session={session}"
            )
    return warnings
