"""Aggregate-only research evidence viewer. No upload or clinical inference."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import streamlit as st
from strokebridge.evidence import load_bundle
from strokebridge.brain_heart_contract import sample_row, validate, mock_tests

st.set_page_config(page_title="StrokeBridge | 脑心桥", page_icon="🫀", layout="wide")
st.title("StrokeBridge · 脑心桥")
st.caption("Research evidence explorer | 脑心同查研究证据浏览器 · 2026-10-25")
st.warning("Research use only. Not a diagnostic or stroke-risk calculator. 当前不提供个体诊断或未来AF/卒中风险。")
b = load_bundle()
tabs = st.tabs(["Evidence / 证据", "A32–A34", "Data contract / 数据合同", "About / 边界"])
with tabs[0]:
    st.subheader("A bridge to validation—not a validated risk score")
    cols = st.columns(3)
    cols[0].metric("A32 supervised fits", b["A32_completion"]["fits"])
    cols[1].metric("A33/A34 new fits", 0)
    cols[2].metric("Outer patients per seed", 200)
    st.info("Two seeds evaluate the same patients; do not double the sample size. 两种子为同一批患者。")
    st.markdown("**AF-ID**: historical concurrent AF identification research.\n\n"
                "**P-MeasureAudit**: patient-level measurement and feasibility audits; practical gates failed.\n\n"
                "**Brain–Heart Risk**: planned only; paired brain/heart and dated outcomes are not available.")
    st.error("A34 stopped BEFORE training: endpoint-only correction was not admissible under frozen constraints.")
    st.json(b["evidence_boundaries"], expanded=True)
with tabs[1]:
    phase = st.selectbox("Phase", ["A32", "A33", "A34"])
    if phase == "A34":
        st.caption("Optimistic necessary-condition bounds, NOT measured model performance.")
        st.dataframe(b["A34_bounds"], hide_index=True, use_container_width=True)
    else:
        rows = b[phase + "_results"]
        seed = st.selectbox("Seed (same patients)", sorted({r["seed"] for r in rows}))
        policy = st.selectbox("Frozen decoder policy", sorted({r["policy"] for r in rows}))
        lead = st.selectbox("Lead", ["v1", "ii"])
        stage = st.radio("Stage", ["raw", "selected", "reported"], horizontal=True)
        out = []
        for r in rows:
            if (r["seed"], r["policy"], r["lead"]) != (seed, policy, lead):
                continue
            s = r["stages"][stage]
            out.append({"arm": r["arm"], "reference_waves": s["reference"],
                        "matched": s["matched"], "predictions": s["predicted"],
                        "P_recall": s["recall"], "precision_lower_bound": s["precision_lower"],
                        "negative_recall": s["negative_recall"], "biphasic_recall": s["biphasic_recall"],
                        "reported_patient_joint_coverage": r["strict_patient_joint_reported"]})
        st.dataframe(out, hide_index=True, use_container_width=True)
        st.caption("Partial annotations: precision_lower is a reference-matching lower bound, not a clinical no-P truth. "
                   "Patient joint coverage spans BOTH II and V1, not V1 alone. Undefined precision is blank, not zero.")
    with st.expander("Audited completion receipt"):
        st.json(b[phase + "_completion"])
with tabs[2]:
    st.subheader("Synthetic structure only / 仅合成结构示例")
    st.markdown("No patient upload is enabled. A schema PASS does not authorize source admission, fitting or prediction.")
    sample = sample_row()
    st.json(sample, expanded=False)
    st.json(validate(sample))
    if st.button("Run synthetic-only contract tests / 运行合成合同单测"):
        r = mock_tests()
        st.success(f"{r['count']} synthetic checks passed; fits=0, predictions=0")
with tabs[3]:
    st.markdown("### What this demonstration does not do\n\n"
                "- No individual ECG, imaging or identifiers are collected.\n"
                "- No model weights, raw waveforms or hospital data are hosted.\n"
                "- No AF diagnosis, stroke prediction, treatment recommendation or clinical alert.\n"
                "- SPH remains quarantined; RDB is not scored as an independent V1/no-P reference.\n"
                "- Review PASS means execution/provenance checks, not scientific effectiveness.\n\n"
                "See the release README and original source inventory for reproduction boundaries.")
    st.link_button("Streamlit official deployment documentation", "https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app")

