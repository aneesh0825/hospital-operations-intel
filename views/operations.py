from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from config.constants import VALIDATION_METRICS
from services.registry import condition_summary, short_patient_id
from ui.components import inline_stats, page_header, section_header


def _open_workup(patient_id: object) -> None:
    st.session_state["selected_patient_id"] = patient_id
    st.session_state["active_page"] = "Patient Workup"


def _risk_distribution(registry: pd.DataFrame) -> alt.Chart:
    counts = registry["risk_level"].value_counts().reindex(["LOW", "MODERATE", "HIGH"], fill_value=0).reset_index()
    counts.columns = ["Risk tier", "Encounters"]
    return alt.Chart(counts).mark_bar(size=28, cornerRadiusEnd=3).encode(
        y=alt.Y("Risk tier:N", sort=["LOW", "MODERATE", "HIGH"], title=None, axis=alt.Axis(labelColor="#9ba3af", labelFontSize=12)),
        x=alt.X("Encounters:Q", title=None, axis=alt.Axis(grid=False, labels=False, ticks=False)),
        color=alt.Color("Risk tier:N", scale=alt.Scale(domain=["LOW", "MODERATE", "HIGH"], range=["#69d59a", "#f0c96b", "#ff7f8e"]), legend=None),
        tooltip=["Risk tier", alt.Tooltip("Encounters:Q", format=",")],
    ).properties(height=190).configure_view(strokeOpacity=0)


def render(registry: pd.DataFrame) -> None:
    total = len(registry)
    flagged = registry["intervention_recommended"].astype(bool)
    queue = registry.loc[flagged]
    flagged_rate = registry.loc[flagged, "actual_readmitted_30_days"].mean() * 100 if flagged.any() else 0.0
    non_flagged_rate = registry.loc[~flagged, "actual_readmitted_30_days"].mean() * 100 if (~flagged).any() else 0.0

    page_header(
        "OPERATIONS",
        "Readmission Operations",
        "See population risk, review pressure, and the patients behind it.",
    )
    inline_stats([
        (f"{total:,}", "encounters"),
        (f"{len(queue):,}", "review queue"),
        (f"{registry['actual_readmitted_30_days'].mean() * 100:.1f}%", "observed readmission"),
        (VALIDATION_METRICS["ROC-AUC"], "ROC-AUC"),
    ])

    chart_col, outcome_col = st.columns((1.15, 0.85), gap="large")
    with chart_col:
        with st.container(border=True):
            section_header("Population risk distribution", "Scored encounters by calibrated risk tier.", "LIVE REGISTRY")
            st.altair_chart(_risk_distribution(registry), width="stretch")
    with outcome_col:
        with st.container(border=True):
            section_header("Outcome separation", "Observed 30-day readmission by operational flag.", "OBSERVED")
            st.metric("Flagged encounters", f"{flagged_rate:.1f}%")
            st.metric("Non-flagged encounters", f"{non_flagged_rate:.1f}%")
            st.caption("Observed outcomes in synthetic registry data; not causal treatment effects.")

    st.write("")
    with st.container(border=True):
        section_header("Review queue", f"{queue['patient_id'].nunique():,} unique patients in the current review queue. Filter, select, and continue into Patient Workup.", "PRIORITY")
        c1, c2, c3 = st.columns((0.28, 0.28, 0.44))
        with c1:
            risk_filter = st.selectbox("Risk tier", ["All Risk Tiers", "HIGH", "MODERATE", "LOW"], key="queue_risk_filter")
        with c2:
            review_filter = st.selectbox("Review status", ["All Review Statuses", "Recommended", "Not Flagged"], key="queue_review_filter")
        with c3:
            search = st.text_input("Patient search", placeholder="Search patient ID…", key="queue_patient_search")

        priority = registry.sort_values("readmission_probability", ascending=False).drop_duplicates("patient_id").copy()
        if risk_filter != "All Risk Tiers":
            priority = priority.loc[priority["risk_level"] == risk_filter]
        if review_filter == "Recommended":
            priority = priority.loc[priority["intervention_recommended"].astype(bool)]
        elif review_filter == "Not Flagged":
            priority = priority.loc[~priority["intervention_recommended"].astype(bool)]
        priority["Patient"] = priority["patient_id"].map(short_patient_id)
        if search.strip():
            priority = priority.loc[priority["Patient"].str.contains(search.strip(), case=False, na=False)]
        priority["Risk (%)"] = priority["readmission_probability_percent"].round(1)
        priority["Review status"] = priority["intervention_recommended"].map({True: "RECOMMENDED", False: "NOT FLAGGED"})
        priority["Conditions"] = priority.apply(condition_summary, axis=1)
        table = priority[["Patient", "Risk (%)", "risk_level", "Review status", "admissions_last_90_days", "prior_readmissions_last_365_days", "Conditions"]].head(20).rename(columns={"risk_level": "Risk tier", "admissions_last_90_days": "Admissions / 90 days", "prior_readmissions_last_365_days": "Prior readmissions / year"})
        st.dataframe(table, width="stretch", hide_index=True, column_config={"Risk (%)": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.1f%%")})
        if not priority.empty:
            selected_label = st.selectbox("Continue with a patient", priority["Patient"].head(20).tolist(), key="queue_open_patient")
            selected_id = priority.loc[priority["Patient"] == selected_label, "patient_id"].iloc[0]
            st.button(
                "Open patient workup",
                type="primary",
                on_click=_open_workup,
                args=(selected_id,),
            )

    with st.expander("Model validation"):
        columns = st.columns(len(VALIDATION_METRICS))
        for column, (label, value) in zip(columns, VALIDATION_METRICS.items(), strict=True):
            column.metric(label, value)
        st.caption("Held-out patient-level validation on synthetic Synthea data. Not intended as clinical validation.")
