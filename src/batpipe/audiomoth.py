from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re


AUDIOMOTH_NAME_RE = re.compile(
    r"(?P<date>\d{8})_(?P<time>\d{6})T(?:\.[A-Za-z0-9]+)?$",
    re.IGNORECASE,
)


def normalize_recording_name(path_or_name: str | Path) -> str:
    name = Path(path_or_name).name
    suffixes = Path(name).suffixes
    if suffixes[-2:] == [".WAV", ".json"] or suffixes[-2:] == [".wav", ".json"]:
        return name[: -len(".json")]
    return name


def parse_audiomoth_timestamp(path_or_name: str | Path) -> datetime:
    recording_name = normalize_recording_name(path_or_name)
    stem = Path(recording_name).stem
    match = AUDIOMOTH_NAME_RE.match(stem)
    if not match:
        raise ValueError(f"Unsupported AudioMoth filename: {recording_name}")
    date_part = match.group("date")
    time_part = match.group("time")
    return datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")


def build_night_window(night_token: str, night_start_hour: int, night_end_hour: int) -> tuple[datetime, datetime]:
    if not re.fullmatch(r"\d{8}", night_token):
        raise ValueError(f"Night token must be YYYYMMDD, got: {night_token}")
    if not 0 <= night_start_hour <= 23:
        raise ValueError(f"night_start_hour must be between 0 and 23, got: {night_start_hour}")
    if not 0 <= night_end_hour <= 23:
        raise ValueError(f"night_end_hour must be between 0 and 23, got: {night_end_hour}")

    start = datetime.strptime(night_token, "%Y%m%d").replace(
        hour=night_start_hour,
        minute=0,
        second=0,
        microsecond=0,
    )
    if night_end_hour <= night_start_hour:
        end = (start + timedelta(days=1)).replace(
            hour=night_end_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
    else:
        end = start.replace(
            hour=night_end_hour,
            minute=0,
            second=0,
            microsecond=0,
        )
    return start, end


def is_in_night_window(
    path_or_name: str | Path,
    night_token: str,
    night_start_hour: int,
    night_end_hour: int,
) -> bool:
    recording_time = parse_audiomoth_timestamp(path_or_name)
    start, end = build_night_window(night_token, night_start_hour, night_end_hour)
    return start <= recording_time < end
