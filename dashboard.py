import streamlit as st
import pandas as pd
import tempfile
from quick_counter import QuickCountAnalyzer

# --- Konfigurasi halaman ---
st.set_page_config(page_title="Dashboard Quick Count", layout="wide")
st.title("Dashboard Quick Count")

# --- Upload file ---
uploaded_file = st.file_uploader("Upload Data Quick Count (CSV atau Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Simpan file sementara
    suffix = ".csv" if uploaded_file.name.endswith(".csv") else ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    # Inisialisasi analyzer
    analyzer = QuickCountAnalyzer(tmp_path)

    # --- Ringkasan hasil ---
    st.header("Ringkasan Hasil Quick Count")
    turnout = analyzer.calculate_voter_turnout()
    cols = st.columns(4)
    cols[0].metric("Suara Sah", f"{turnout['valid_votes']:,}")
    cols[1].metric("Tidak Sah", f"{turnout['invalid_votes']:,}")
    cols[2].metric("Total Masuk", f"{turnout['total_votes_cast']:,}")
    cols[3].metric("Partisipasi", f"{turnout['voter_turnout_percentage']:.1f}%")

    # --- Perolehan suara ---
    st.header("Perolehan Suara Kandidat")
    totals = analyzer.calculate_totals()
    percentages = analyzer.calculate_percentages()
    col1, col2 = st.columns([2, 1])
    with col1:
        st.bar_chart(pd.DataFrame.from_dict(totals, orient="index", columns=["Suara"]))
    with col2:
        st.dataframe(pd.DataFrame({
            "Kandidat": list(percentages.keys()),
            "Persentase": [f"{v:.2f}%" for v in percentages.values()]
        }))

    # --- Distribusi per kecamatan ---
    st.header("Distribusi Suara per Kecamatan")
    try:
        district_results = analyzer.calculate_by_district()
        st.dataframe(district_results)
        st.bar_chart(district_results.set_index("District")[analyzer.candidates])
    except Exception as e:
        st.warning(f"Data kecamatan tidak tersedia: {e}")

    # --- Confidence Interval & Margin of Error ---
    st.header("Confidence Interval & Margin of Error")
    ci = analyzer.calculate_confidence_intervals()
    moe = analyzer.calculate_margin_of_error()
    for c in analyzer.candidates:
        st.write(f"**{c}**: {ci[c][0]:.2f}% - {ci[c][1]:.2f}% (MoE ±{moe[c]:.2f}%)")

    # --- Korelasi antar kandidat ---
    st.header("Korelasi Antar Kandidat")
    corr = analyzer.calculate_correlation_matrix()
    st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

    # --- Simulasi Quick Count ---
    st.header("Simulasi Quick Count (Sampling)")
    sample_size = st.slider("Jumlah sampel TPS", 10, len(analyzer.data), 100, step=10)
    if st.button("Jalankan Simulasi"):
        analyzer.run_sampling_analysis(sample_size)

else:
    st.info("Silakan upload file CSV/Excel untuk memulai analisis.")
