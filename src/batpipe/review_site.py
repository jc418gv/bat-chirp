from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from batpipe.audiomoth import parse_audiomoth_timestamp
from batpipe.review_site_render import (
    render_cards_html,
    render_hour_card,
    render_hour_sections_html,
    render_html_document,
)

_ASSETS_DIR = Path(__file__).parent / "assets"


# ---------------------------------------------------------------------------
# CSV / data helpers
# ---------------------------------------------------------------------------

def _read_csv_rows(csv_path: Path | None) -> list[dict[str, str]]:
    if csv_path is None or not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_int(value: object, default: int = 0) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _to_float(value: object) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Entry enrichment
# ---------------------------------------------------------------------------

def _build_review_entries(
    review_items: list[dict[str, object]],
    review_queue_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rank_by_source = {str(row.get("source_file")): row for row in review_queue_rows}
    entries: list[dict[str, object]] = []

    for item in review_items:
        audio_file = str(item.get("audio_file"))
        audio_name = Path(audio_file).name
        recording_start = parse_audiomoth_timestamp(audio_name)
        ranking = rank_by_source.get(audio_name, {})
        rank_value = _to_int(ranking.get("rank"), default=999999)
        detection_count = _to_int(ranking.get("detection_count"), default=_to_int(item.get("detections_in_clip")))
        max_det_prob = _to_float(ranking.get("max_det_prob"))
        entries.append(
            {
                "audio_file": audio_file,
                "audio_name": audio_name,
                "recording_start": recording_start,
            "recording_hour_key": recording_start.strftime("%y%m%d%H"),
            "recording_hour_label": recording_start.strftime("%Y-%m-%d %H:00"),
                "sample_local_time": str(item.get("sample_local_time") or ""),
                "rank": rank_value,
                "detection_count": detection_count,
                "max_det_prob": max_det_prob,
                "clip_wav": str(item.get("clip_wav") or ""),
                "clip_mp3": str(item.get("clip_mp3") or ""),
                "audible_wav": str(item.get("audible_wav") or ""),
                "audible_mp3": str(item.get("audible_mp3") or ""),
                "spectrogram_png": str(item.get("spectrogram_png") or ""),
                "report_json": str(item.get("report_json") or ""),
                "clip_start_s": item.get("clip_start_s"),
                "clip_end_s": item.get("clip_end_s"),
                "activity_segment_count": item.get("activity_segment_count"),
                "detections_in_clip": item.get("detections_in_clip"),
            }
        )

    entries.sort(
        key=lambda entry: (
            int(entry["rank"]),
            -int(entry["detection_count"]),
            entry["recording_start"],
        )
    )
    return entries


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _write_style(summary_dir: Path) -> Path:
    style_path = summary_dir / "style.css"
    style_path.write_text(
        (_ASSETS_DIR / "style.css").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return style_path


def _write_hour_pages(
    entries_by_hour: dict[str, list[dict[str, object]]],
    summary_dir: Path,
) -> tuple[list[str], list[str]]:
    hour_page_paths: list[str] = []
    hour_cards: list[str] = []
    for hour_key in sorted(entries_by_hour):
        hour_entries = entries_by_hour[hour_key]
        hour_label = str(hour_entries[0].get("recording_hour_label") or hour_key)
        hour_filename = f"hour-{hour_key}.html"
        hour_page_path = summary_dir / hour_filename
        hour_page_paths.append(str(hour_page_path))
        hour_page_path.write_text(
            render_html_document(
                title=f"Review Hour {hour_label}",
                body=f'<section class="cards">{render_cards_html(hour_entries, summary_dir)}</section>',
                back_link="index.html",
            ),
            encoding="utf-8",
        )
        hour_cards.append(render_hour_card(hour_label, hour_filename, len(hour_entries)))
    return hour_page_paths, hour_cards


def _write_index(
    night_output_dir: Path,
    summary_dir: Path,
    hour_cards: list[str],
    entries_by_hour: dict[str, list[dict[str, object]]],
) -> Path:
    index_body = (
        f'<section class="hours">{"".join(hour_cards)}</section>'
        f'<section class="hour-groups">{render_hour_sections_html(entries_by_hour, summary_dir)}</section>'
    )
    index_path = summary_dir / "index.html"
    index_path.write_text(
        render_html_document(title=f"Night Review {night_output_dir.name}", body=index_body),
        encoding="utf-8",
    )
    return index_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_review_site(
    night_output_dir: Path,
    review_items: list[dict[str, object]],
    summary_outputs: dict[str, object] | None = None,
) -> dict[str, object]:
    summary_dir = night_output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)

    review_queue_path = None
    if summary_outputs:
        review_queue_value = summary_outputs.get("review_queue")
        if review_queue_value:
            review_queue_path = Path(str(review_queue_value))

    entries = _build_review_entries(review_items, _read_csv_rows(review_queue_path))

    entries_by_hour: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        entries_by_hour[str(entry["recording_hour_key"])].append(entry)

    style_path = _write_style(summary_dir)
    hour_page_paths, hour_cards = _write_hour_pages(entries_by_hour, summary_dir)
    index_path = _write_index(night_output_dir, summary_dir, hour_cards, entries_by_hour)

    return {
        "review_summary_dir": str(summary_dir),
        "review_index_html": str(index_path),
        "review_css": str(style_path),
        "review_hour_pages": hour_page_paths,
        "review_entry_count": len(entries),
    }

