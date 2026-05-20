from __future__ import annotations

from collections import defaultdict
from html import escape
from pathlib import Path
import csv
import os

from batpipe.audiomoth import parse_audiomoth_timestamp

_ASSETS_DIR = Path(__file__).parent / "assets"


# ---------------------------------------------------------------------------
# CSV / data helpers
# ---------------------------------------------------------------------------

def _read_csv_rows(csv_path: Path | None) -> list[dict[str, str]]:
    if csv_path is None or not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _relative_link(from_dir: Path, target_path: str | Path | None) -> str:
    if not target_path:
        return ""
    return os.path.relpath(str(target_path), start=str(from_dir)).replace("\\", "/")


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
# HTML rendering
# ---------------------------------------------------------------------------

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
                probability_text = f"p={probability:.3f}" if isinstance(probability, float) else "p=n/a"
                recording_start = entry["recording_start"]
                time_label = recording_start.strftime("%H:%M:%S") if hasattr(recording_start, "strftime") else str(recording_start)
                det_count = escape(str(entry["detection_count"]))
                segments = escape(str(entry["activity_segment_count"]))
                clip_range = f"{entry['clip_start_s']}s – {entry['clip_end_s']}s"
                audio_panel = f"""<details class="audio-panel">
            <summary>Audio</summary>
            <div class="audio-grid">
                <section class="audio-block">
                    <p class="audio-label">x1 clip</p>
                    <audio controls preload="none" src="{escape(clip_mp3_href)}"></audio>
                    <div class="mini-links">
                        <a class="pill subtle" href="{escape(clip_mp3_href)}" download>mp3</a>
                        <a class="pill subtle" href="{escape(clip_wav_href)}" download>wav</a>
                    </div>
                </section>
                <section class="audio-block">
                    <p class="audio-label">x8 audible</p>
                    <audio controls preload="none" src="{escape(audible_mp3_href)}"></audio>
                    <div class="mini-links">
                        <a class="pill subtle" href="{escape(audible_mp3_href)}" download>mp3</a>
                        <a class="pill subtle" href="{escape(audible_wav_href)}" download>wav</a>
                    </div>
                </section>
            </div>
        </details>"""
                cards.append(
                        f"""<article class="card">
    <button class="image-link" type="button" data-spectrogram-modal-trigger data-spectrogram-src="{escape(spectrogram_href)}" data-spectrogram-alt="Spectrogram {escape(time_label)}" data-spectrogram-title="{escape(str(entry['audio_name']))}">
        <img class="spectrogram" src="{escape(spectrogram_href)}" alt="Spectrogram {escape(time_label)}" loading="lazy">
        <span class="image-link-label">Expand spectrogram</span>
    </button>
    <div class="card-body">
        <p class="card-time">{escape(time_label)}<span class="card-rank">rank {escape(str(entry['rank']))}</span></p>
        <p class="card-filename">{escape(str(entry['audio_name']))}</p>
        <p class="card-stats">{det_count} detections · {escape(probability_text)} · {segments} activity segment{'s' if str(entry['activity_segment_count']) != '1' else ''} · {escape(clip_range)}</p>
        {audio_panel}
        <div class="links">
            <button class="pill primary pill-button" type="button" data-spectrogram-modal-trigger data-spectrogram-src="{escape(spectrogram_href)}" data-spectrogram-alt="Spectrogram {escape(time_label)}" data-spectrogram-title="{escape(str(entry['audio_name']))}">spectrogram</button>
            <a class="pill" href="{escape(spectrogram_href)}" target="_blank" rel="noreferrer">new tab</a>
            <a class="pill" href="{escape(audible_mp3_href)}" download>x8 mp3</a>
            <a class="pill" href="{escape(audible_wav_href)}" download>x8 wav</a>
            <a class="pill" href="{escape(clip_mp3_href)}" download>x1 mp3</a>
            <a class="pill" href="{escape(clip_wav_href)}" download>x1 wav</a>
            <a class="pill" href="{escape(report_href)}">json</a>
        </div>
    </div>
</article>"""
                )
        return "\n".join(cards)


def _render_modal_shell() -> str:
        return """<dialog class="spectrogram-modal" data-spectrogram-modal>
    <div class="spectrogram-modal-shell">
        <div class="spectrogram-modal-bar">
            <div>
                <p class="spectrogram-modal-kicker">Spectrogram</p>
                <h2 class="spectrogram-modal-title" data-spectrogram-modal-title>Preview</h2>
            </div>
            <div class="spectrogram-modal-actions">
                <a class="pill" data-spectrogram-modal-open href="#" target="_blank" rel="noreferrer">open full image</a>
                <button class="pill pill-button" type="button" data-spectrogram-modal-close>close</button>
            </div>
    </div>
        <div class="spectrogram-modal-frame">
            <img class="spectrogram-modal-image" data-spectrogram-modal-image src="" alt="">
        </div>
    </div>
</dialog>"""


def _render_modal_script() -> str:
        return """<script>
(() => {
    const modal = document.querySelector('[data-spectrogram-modal]');
    if (!modal) {
        return;
    }

    const modalImage = modal.querySelector('[data-spectrogram-modal-image]');
    const modalTitle = modal.querySelector('[data-spectrogram-modal-title]');
    const modalOpenLink = modal.querySelector('[data-spectrogram-modal-open]');
    const closeButton = modal.querySelector('[data-spectrogram-modal-close]');
    const triggers = document.querySelectorAll('[data-spectrogram-modal-trigger]');

    const openModal = (trigger) => {
        const src = trigger.getAttribute('data-spectrogram-src');
        const alt = trigger.getAttribute('data-spectrogram-alt') || 'Spectrogram preview';
        const title = trigger.getAttribute('data-spectrogram-title') || 'Spectrogram preview';
        if (!src || !modalImage || !modalTitle || !modalOpenLink) {
            window.open(src, '_blank', 'noopener');
            return;
        }

        modalImage.src = src;
        modalImage.alt = alt;
        modalTitle.textContent = title;
        modalOpenLink.href = src;

        if (typeof modal.showModal === 'function') {
            modal.showModal();
            return;
        }

        window.open(src, '_blank', 'noopener');
    };

    triggers.forEach((trigger) => {
        trigger.addEventListener('click', () => openModal(trigger));
    });

    closeButton?.addEventListener('click', () => {
        modal.close();
    });

    modal.addEventListener('click', (event) => {
        if (event.target === modal) {
            modal.close();
        }
    });

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && modal.open) {
            modal.close();
        }
    });
})();
</script>"""


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
    {_render_modal_shell()}
    {_render_modal_script()}
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
            _render_html_document(
                title=f"Review Hour {hour_label}",
                body=f'<section class="cards">{_render_cards_html(hour_entries, summary_dir)}</section>',
                back_link="index.html",
            ),
            encoding="utf-8",
        )
        hour_cards.append(
            f"""
            <section class="hour-card">
              <h2>Hour {escape(hour_label)}</h2>
              <p>{escape(str(len(hour_entries)))} review clips</p>
              <a href="{escape(hour_filename)}">Open hour page</a>
            </section>
            """.strip()
        )
    return hour_page_paths, hour_cards


def _write_index(
    night_output_dir: Path,
    summary_dir: Path,
    hour_cards: list[str],
    entries_by_hour: dict[str, list[dict[str, object]]],
) -> Path:
    index_body = (
        f'<section class="hours">{"".join(hour_cards)}</section>'
        f'<section class="hour-groups">{_render_hour_sections_html(entries_by_hour, summary_dir)}</section>'
    )
    index_path = summary_dir / "index.html"
    index_path.write_text(
        _render_html_document(title=f"Night Review {night_output_dir.name}", body=index_body),
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

