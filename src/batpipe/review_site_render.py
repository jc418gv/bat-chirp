from __future__ import annotations

from html import escape
from pathlib import Path
import os


def _relative_link(from_dir: Path, target_path: str | Path | None) -> str:
    if not target_path:
        return ""
    return os.path.relpath(str(target_path), start=str(from_dir)).replace("\\", "/")


def _render_audio_panel(*, clip_mp3_href: str, clip_wav_href: str, audible_mp3_href: str, audible_wav_href: str) -> str:
    return f"""<details class="audio-panel">
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


def _render_card_actions(
    *,
    spectrogram_href: str,
    time_label: str,
    audio_name: str,
    audible_mp3_href: str,
    audible_wav_href: str,
    clip_mp3_href: str,
    clip_wav_href: str,
    report_href: str,
) -> str:
    return f"""<div class="links">
      <button class="pill primary pill-button" type="button" data-spectrogram-modal-trigger data-spectrogram-src="{escape(spectrogram_href)}" data-spectrogram-alt="Spectrogram {escape(time_label)}" data-spectrogram-title="{escape(audio_name)}">spectrogram</button>
      <a class="pill" href="{escape(spectrogram_href)}" target="_blank" rel="noreferrer">new tab</a>
      <a class="pill" href="{escape(audible_mp3_href)}" download>x8 mp3</a>
      <a class="pill" href="{escape(audible_wav_href)}" download>x8 wav</a>
      <a class="pill" href="{escape(clip_mp3_href)}" download>x1 mp3</a>
      <a class="pill" href="{escape(clip_wav_href)}" download>x1 wav</a>
      <a class="pill" href="{escape(report_href)}">json</a>
    </div>"""


def render_cards_html(entries: list[dict[str, object]], summary_dir: Path) -> str:
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
        audio_name = str(entry["audio_name"])
        audio_panel = _render_audio_panel(
            clip_mp3_href=clip_mp3_href,
            clip_wav_href=clip_wav_href,
            audible_mp3_href=audible_mp3_href,
            audible_wav_href=audible_wav_href,
        )
        card_actions = _render_card_actions(
            spectrogram_href=spectrogram_href,
            time_label=time_label,
            audio_name=audio_name,
            audible_mp3_href=audible_mp3_href,
            audible_wav_href=audible_wav_href,
            clip_mp3_href=clip_mp3_href,
            clip_wav_href=clip_wav_href,
            report_href=report_href,
        )
        cards.append(
            f"""<article class="card">
  <button class="image-link" type="button" data-spectrogram-modal-trigger data-spectrogram-src="{escape(spectrogram_href)}" data-spectrogram-alt="Spectrogram {escape(time_label)}" data-spectrogram-title="{escape(audio_name)}">
    <img class="spectrogram" src="{escape(spectrogram_href)}" alt="Spectrogram {escape(time_label)}" loading="lazy">
    <span class="image-link-label">Expand spectrogram</span>
  </button>
  <div class="card-body">
    <p class="card-time">{escape(time_label)}<span class="card-rank">rank {escape(str(entry['rank']))}</span></p>
    <p class="card-filename">{escape(audio_name)}</p>
    <p class="card-stats">{det_count} detections · {escape(probability_text)} · {segments} activity segment{'s' if str(entry['activity_segment_count']) != '1' else ''} · {escape(clip_range)}</p>
    {audio_panel}
    {card_actions}
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


def render_html_document(title: str, body: str, *, back_link: str | None = None) -> str:
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


def render_hour_sections_html(entries_by_hour: dict[str, list[dict[str, object]]], summary_dir: Path) -> str:
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
              <section class="cards">{render_cards_html(hour_entries, summary_dir)}</section>
            </details>
            """.strip()
        )
    return "\n".join(sections)


def render_hour_card(hour_label: str, hour_filename: str, hour_entry_count: int) -> str:
    return f"""
            <section class="hour-card">
              <h2>Hour {escape(hour_label)}</h2>
              <p>{escape(str(hour_entry_count))} review clips</p>
              <a href="{escape(hour_filename)}">Open hour page</a>
            </section>
            """.strip()
