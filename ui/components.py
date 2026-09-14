from __future__ import annotations

from collections.abc import Iterable
from html import escape

import streamlit as st


def render_brand() -> None:
    st.html(
        """
        <div class="admitra-brand">
            <div class="admitra-brand-mark">A</div>
            <div class="admitra-brand-name">Admitra</div>
            <div class="admitra-brand-subtitle">
                clinical intelligence
            </div>
        </div>
        """
    )


def inline_stats(
    items: Iterable[tuple[str, str]],
) -> None:

    values = "".join(
        (
            '<div class="inline-stat">'
            f'<div class="inline-stat-value">{escape(value)}</div>'
            f'<div class="inline-stat-label">{escape(label)}</div>'
            '</div>'
        )
        for value, label in items
    )

    st.html(
        f'<div class="inline-stats">{values}</div>'
    )


def page_header(
    kicker: str,
    title: str,
    description: str,
) -> None:

    st.html(
        (
            '<div class="page-hero">'
            f'<div class="eyebrow">{escape(kicker)}</div>'
            f'<h1 class="page-title">{escape(title)}</h1>'
            f'<div class="page-description">{escape(description)}</div>'
            '</div>'
        )
    )


def section_header(
    title: str,
    description: str,
    tag: str = "",
) -> None:

    st.html(
        (
            '<div class="section-header">'
            f'<div class="section-title">{escape(title)}</div>'
            f'<div class="section-description">{escape(description)}</div>'
            f'<div class="section-tag">{escape(tag)}</div>'
            '</div>'
        )
    )


def risk_signal(
    probability: float,
    risk_level: str,
    threshold_percent: float,
    review_recommended: bool,
) -> None:

    status = (
        "Review recommended"
        if review_recommended
        else "No review flag"
    )

    st.html(
        (
            '<div class="risk-panel">'
            '<div class="panel-label">'
            'Calibrated 30-day readmission risk'
            '</div>'
            f'<div class="risk-number">{probability:.1f}%</div>'
            '<div class="risk-detail">'
            f'{escape(risk_level.title())} risk · '
            f'{escape(status)} · '
            f'{threshold_percent:.0f}% operational threshold'
            '</div>'
            '</div>'
        )
    )


def render_drivers(
    drivers: list[dict[str, object]],
) -> None:

    for driver in drivers:

        direction = str(
            driver["direction"]
        )

        if direction == "increases risk":
            direction_class = "driver-up"

        elif direction == "decreases risk":
            direction_class = "driver-down"

        else:
            direction_class = ""

        st.html(
            (
                '<div class="driver-card">'
                f'<span class="driver-feature">'
                f'{escape(str(driver["feature"]))}'
                f'</span>'
                f'<span class="{direction_class}">'
                f' · {escape(direction)}'
                f'</span>'
                f'<div class="driver-value">'
                f'Observed value: {escape(str(driver["value"]))}'
                f'</div>'
                '</div>'
            )
        )


def render_disclaimer() -> None:

    st.caption(
        (
            "Admitra is a portfolio demonstration using synthetic "
            "Synthea healthcare data. It is not intended for clinical use."
        )
    )