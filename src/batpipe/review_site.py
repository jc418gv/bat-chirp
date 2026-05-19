from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
import csv
import os

from batpipe.audiomoth import parse_audiomoth_timestamp


def _read_csv_rows(csv_path: Path | None) -> list[dict[str, str]]:
    if csv_path is None or not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative_link(from_dir: Path, target_path: str | Path | None) -> str:
    if not target_path:
        return ""
    relative_path = os.path.relpath(str(target_path), start=str(from_dir))
    return relative_path.replace("\\", "/")


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
                "expanded_train_segment_count": item.get("expanded_train_segment_count"),
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


def _render_cards_html(entries: list[dict[str, object]], summary_dir: Path) -> str:
    cards: list[str] = []
    for entry in entries:
        spectrogram_href = _relative_link(summary_dir, str(entry["spectrogram_png"]))
        clip_mp3_href = _relative_link(summary_dir, str(entry["clip_mp3"]))
        clip_wav_href = _relative_link(summary_dir, str(entry["clip_wav"]))
        audible_mp3_href = _relative_link(summary_dir, str(entry["audible_mp3"]))
        audible_wav_href = _relative_link(summary_dir, str(entry["audible_wav"]))
        report_href = _relative_link(summary_dir, str(entry["report_json"]))
        probability = entry["max_det_prob"]
        probability_text = f"{probability:.3f}" if isinstance(probability, float) else "n/a"
        cards.append(
            f"""
            <article class=\"card\">
              <a class=\"image-link\" href=\"{escape(spectrogram_href)}\"><img class=\"spectrogram\" src=\"{escape(spectrogram_href)}\" alt=\"Spectrogram for {escape(str(entry['audio_name']))}\" loading=\"lazy\"></a>
              <div class=\"card-body\">
                <h3>{escape(str(entry['audio_name']))}</h3>
                <p class=\"meta\">Rank {escape(str(entry['rank']))} · {escape(str(entry['recording_start'].strftime('%Y-%m-%d %H:%M:%S')))} · sample {escape(str(entry['sample_local_time']))}</p>
                <p class=\"meta\">Detections {escape(str(entry['detection_count']))} · clip detections {escape(str(entry['detections_in_clip']))} · max det prob {escape(probability_text)}</p>
                <p class=\"meta\">Clip {escape(str(entry['clip_start_s']))}s to {escape(str(entry['clip_end_s']))}s · train segments {escape(str(entry['expanded_train_segment_count']))}</p>
                <div class=\"links\">
                  <a href=\"{escape(clip_wav_href)}\">clip wav</a>
                  <a href=\"{escape(clip_mp3_href)}\">clip mp3</a>
                  <a href=\"{escape(audible_wav_href)}\">x8 wav</a>
                  <a href=\"{escape(audible_mp3_href)}\">x8 mp3</a>
                  <a href=\"{escape(spectrogram_href)}\">spectrogram</a>
                  <a href=\"{escape(report_href)}\">report</a>
                </div>
              </div>
            </article>
            """.strip()
        )
    return "\n".join(cards)


def _render_html_document(title: str, body: str, *, back_link: str | None = None) -> str:
    back_link_html = f'<p class="back-link"><a href="{escape(back_link or "index.html")}">Back to nightly overview</a></p>' if back_link else ""
    return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <link rel=\"stylesheet\" href=\"style.css\">
</head>
<body>
  <main class=\"page\">
    <header class=\"hero\">
      <h1>{escape(title)}</h1>
      {back_link_html}
    </header>
    {body}
  </main>
</body>
</html>
"""


def _render_hour_sections_html(
    entries_by_hour: dict[str, list[dict[str, object]]],
    summary_dir: Path,
) -> str:
    sections: list[str] = []
    for index, hour_key in enumerate(sorted(entries_by_hour)):
        hour_entries = entries_by_hour[hour_key]
        hour_label = str(hour_entries[0].get("recording_hour_label") or hour_key)
        open_attr = " open" if index == 0 else ""
        sections.append(
            f"""
            <details class="hour-group"{open_attr}>
              <summary>
                <span class="hour-title">Hour {escape(hour_label)}</span>
                <span class="hour-count">{escape(str(len(hour_entries)))} detections</span>
                <a class="hour-page-link" href="{escape(f'hour-{hour_key}.html')}" onclick="event.stopPropagation();">open hour page</a>
              </summary>
              <section class="cards">{_render_cards_html(hour_entries, summary_dir)}</section>
            </details>
            """.strip()
        )
    return "\n".join(sections)


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

    style_path = summary_dir / "style.css"
    style_path.write_text(
        """
body {
  margin: 0;
  font-family: Georgia, \"Times New Roman\", serif;
  background: linear-gradient(180deg, #eef4ea 0%, #d7e5d1 100%);
  color: #1d2a17;
}

.page {
  max-width: 1400px;
  margin: 0 auto;
  padding: 24px;
}

.hero {
  margin-bottom: 24px;
}

.hero h1 {
  margin: 0 0 8px;
  font-size: 2rem;
}

.hours, .cards {
  display: grid;
  gap: 16px;
}

.hours {
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin-bottom: 28px;
}

.cards {
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
}

.hour-card, .card {
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(29, 42, 23, 0.12);
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 10px 24px rgba(29, 42, 23, 0.08);
}

.hour-card {
  padding: 18px;
}

.hour-card a, .links a, .back-link a {
  color: #234a1f;
  font-weight: 600;
}

.image-link {
  display: block;
  background: #0f140e;
}

.spectrogram {
  display: block;
  width: 100%;
  height: 220px;
  object-fit: cover;
}

.card-body {
  padding: 14px 16px 18px;
}

.card-body h3 {
  margin: 0 0 8px;
  font-size: 1rem;
  word-break: break-word;
}

.meta {
  margin: 0 0 8px;
  font-size: 0.92rem;
}

.links {
  display: flex;
  flex-wrap: wrap;
  gap: 10px 14px;
  margin-top: 12px;
}

.hour-groups {
  display: grid;
  gap: 18px;
}

.hour-group {
  background: rgba(255, 255, 255, 0.82);
  border: 1px solid rgba(29, 42, 23, 0.12);
  border-radius: 16px;
  box-shadow: 0 10px 24px rgba(29, 42, 23, 0.08);
  overflow: hidden;
}

.hour-group summary {
  display: flex;
  align-items: center;
  gap: 14px;
  justify-content: space-between;
  padding: 16px 18px;
  cursor: pointer;
  font-weight: 600;
  background: rgba(215, 229, 209, 0.65);
}

.hour-title {
  font-size: 1.05rem;
}

.hour-count {
  color: #35562d;
  margin-left: auto;
}

.hour-page-link {
  white-space: nowrap;
}

.hour-group .cards {
  padding: 18px;
}
        """.strip(),
        encoding="utf-8",
    )

    entries_by_hour: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        entries_by_hour[str(entry["recording_hour_key"])].append(entry)

    hour_page_paths: list[str] = []
    hour_cards: list[str] = []
    for hour_key in sorted(entries_by_hour):
        hour_entries = entries_by_hour[hour_key]
        hour_label = str(hour_entries[0].get("recording_hour_label") or hour_key)
        hour_filename = f"hour-{hour_key}.html"
        hour_page_path = summary_dir / hour_filename
        hour_page_paths.append(str(hour_page_path))
        hour_page_path.write_text(
            _render_html_document(
                title=f"Review Hour {hour_label}",
                body=f'<section class="cards">{_render_cards_html(hour_entries, summary_dir)}</section>',
                back_link="index.html",
            ),
            encoding="utf-8",
        )
        hour_cards.append(
            f"""
            <section class=\"hour-card\">
              <h2>Hour {escape(hour_label)}</h2>
              <p>{escape(str(len(hour_entries)))} review clips</p>
              <a href=\"{escape(hour_filename)}\">Open hour page</a>
            </section>
            """.strip()
        )

    index_path = summary_dir / "index.html"
    index_body = (
        f'<section class="hours">{"".join(hour_cards)}</section>'
      f'<section class="hour-groups">{_render_hour_sections_html(entries_by_hour, summary_dir)}</section>'
    )
    index_path.write_text(
        _render_html_document(title=f"Night Review {night_output_dir.name}", body=index_body),
        encoding="utf-8",
    )

    return {
        "review_summary_dir": str(summary_dir),
        "review_index_html": str(index_path),
        "review_css": str(style_path),
        "review_hour_pages": hour_page_paths,
        "review_entry_count": len(entries),
    }