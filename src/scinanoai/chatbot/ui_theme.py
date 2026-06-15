"""Branding and visual theme for the SciNanoAI Gradio frontend.

Visuals and language-agnostic chrome live here; all translatable copy lives in
:mod:`ui_i18n`, so ``ui.py`` stays a thin layout + wiring layer.
"""

from __future__ import annotations

import gradio as gr

# --- Brand (language-agnostic) -----------------------------------------------

TITLE = "🔬 SciNanoAI"

# Inline + block math, so scientific answers render formulas correctly.
LATEX_DELIMITERS: list[dict[str, str | bool]] = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

# Force the dark palette on load (guarded so it redirects at most once). The
# in-header language link toggles via an inline onclick that relays to the
# hidden ``#lang-trigger`` button, so no global listener is needed here.
FORCE_DARK_JS = """
() => {
  const url = new URL(window.location.href);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.replace(url.href);
  }
}
"""


def status_html(online: bool, label: str) -> str:
    """Render the connection-status badge shown in the header.

    ``label`` is the already-localised status text (the caller picks it from the
    active language catalog).
    """
    state = "online" if online else "offline"
    return f"<span class='status-badge {state}'><span class='dot'></span>{label}</span>"


def build_theme() -> gr.Theme:
    """A dark 'lab' theme: slate surfaces, cyan accent."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.cyan,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
    ).set(
        body_background_fill_dark="#0b1220",
        background_fill_primary_dark="#0f172a",
        background_fill_secondary_dark="#111c33",
        block_background_fill_dark="#0f172a",
        border_color_primary_dark="#1e2a44",
        block_border_color_dark="#1e2a44",
        block_radius="14px",
        button_large_radius="12px",
        button_primary_background_fill_dark="#22d3ee",
        button_primary_background_fill_hover_dark="#06b6d4",
        button_primary_text_color_dark="#04212b",
    )


CUSTOM_CSS = """
/* Widen the centered content column. Gradio 6 caps `.app.fillable` width via
   responsive breakpoints (1280px on wide screens); override that cap. */
.fillable:not(.fill_width) { max-width: 1770px !important; }

/* Header banner: one HTML block, flex row with brand on the left and the
   action links (language toggle + logout) clustered on the right. */
.app-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; padding: 14px 18px; margin-bottom: 8px;
  border-bottom: 2px solid #22d3ee;
}
.app-header .brand { display: flex; flex-direction: column; gap: 2px; }
.app-header .brand .title { font-size: 1.5rem; font-weight: 700; line-height: 1.1; }
.app-header .brand .subtitle { font-size: .9rem; opacity: .7; }
.app-header .actions { display: flex; align-items: center; gap: 16px; }
.app-header .logout-link, .app-header .lang-link {
  font-size: .85rem; text-decoration: none; color: #22d3ee;
  white-space: nowrap; cursor: pointer;
}
.app-header .logout-link:hover, .app-header .lang-link:hover { text-decoration: underline; }

/* The language toggle's real control is a Gradio button kept out of view; the
   visible .lang-link clicks it via JS (see FORCE_DARK_JS). */
#lang-trigger { display: none !important; }

/* Status badge */
.status-badge { display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; }
.status-badge .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.status-badge.online .dot { background: #22c55e; box-shadow: 0 0 6px #22c55e; }
.status-badge.offline .dot { background: #ef4444; box-shadow: 0 0 6px #ef4444; }

/* Hide Gradio's built-in branding footer */
footer { display: none !important; }
"""
