from datetime import datetime
import os
import streamlit as st
import json
from analysis import load_transcript, compute_sentiment, compute_filler_ratio
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Transcript Analyzer", layout="wide")
st.title("🎙️ Transcript Analyzer")

UPLOAD_DIR="upload"

def render_dashboard(result_json: dict):
    st.subheader("🧾 Line-by-Line Analysis")

    view_mode = st.radio("Choose:", ["Original Text", "Highlighted filter words"], horizontal=True, label_visibility="collapsed")

    def sentiment_emoji(label):
        return {
            "POSITIVE": "😊",
            "NEGATIVE": "😠",
            "NEUTRAL": "😐",
            "UNKNOWN": "❓"
        }.get(label.upper(), "❓")

    def filler_icon(ratio):
        if ratio <= 0.00:
            return "🟢"
        elif ratio <= 0.20:
            return "🟡"
        else:
            return "🔴"

    # Header row
    header_cols = st.columns([0.5, 0.7, 3, 1, 1.2])
    headers = ["#", "Speaker", "Text", "Filler Ratio", "Sentiment"]
    for col, head in zip(header_cols, headers):
        col.markdown(f"**{head}**")

    # Data rows
    for i, turn in enumerate(result_json["dialogue"], start=1):
        cols = st.columns([0.5, 0.7, 3, 1, 1.2])
        cols[0].markdown(f"{i}")
        cols[1].markdown(f"Speaker {turn['speaker']}")
        cols[2].markdown(turn["text"] if view_mode == "Original Text" else turn["text_md"])
        cols[3].markdown(f"{filler_icon(turn['filler_ratio'])} {turn['filler_ratio']:.2f}")
        cols[4].markdown(f"{sentiment_emoji(turn['sentiment_label'])} {turn['sentiment_label']}")

    # ---------------- Chart Section ----------------
    st.markdown("---")
    st.subheader("📈 Speaker Turn Metrics")

    col1, col_spacer, col2 = st.columns([2, 3, 2])

    with col1:
        chart_metric = st.radio("📊 Select Metric", ["Both", "Word Count", "Filler Ratio"], index=0, horizontal=True)

    with col2:
        chart_speaker = st.radio("👤 Select Speaker View", ["Both", "A", "B"], index=0, horizontal=True)

    chart_data = []
    for i, entry in enumerate(result_json["dialogue"], start=1):
        if chart_speaker != "Both" and entry["speaker"] != chart_speaker:
            continue
        word_count = len([w for w in entry["text"].split() if w.isalpha()])
        chart_data.append({
            "Line": i,
            "Speaker": entry["speaker"],
            "Word Count": word_count,
            "Filler Ratio": entry["filler_ratio"]
        })

    df_chart = pd.DataFrame(chart_data)

    if not df_chart.empty:
        speaker_colors = {"A": "tab:blue", "B": "lightskyblue"}

        if chart_metric == "Word Count":
            plt.figure(figsize=(10, 4))
            for speaker in ["A", "B"]:
                if chart_speaker != "Both" and speaker != chart_speaker:
                    continue
                subset = df_chart[df_chart["Speaker"] == speaker]
                plt.bar(subset["Line"], subset["Word Count"], label=f"Speaker {speaker}", color=speaker_colors[speaker])
            plt.title("Word Count per Turn")
            plt.xlabel("Line Number")
            plt.ylabel("Word Count")
            plt.legend()
            st.pyplot(plt)

        elif chart_metric == "Filler Ratio":
            plt.figure(figsize=(10, 4))
            for speaker in ["A", "B"]:
                if chart_speaker != "Both" and speaker != chart_speaker:
                    continue
                subset = df_chart[df_chart["Speaker"] == speaker]
                plt.bar(subset["Line"], subset["Filler Ratio"], label=f"Speaker {speaker}", color=speaker_colors[speaker])
            plt.title("Filler Ratio per Turn")
            plt.xlabel("Line Number")
            plt.ylabel("Filler Ratio")
            plt.legend()
            st.pyplot(plt)

        elif chart_metric == "Both":
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax2 = ax1.twinx()

            for speaker in ["A", "B"]:
                if chart_speaker != "Both" and speaker != chart_speaker:
                    continue
                subset = df_chart[df_chart["Speaker"] == speaker]
                ax1.bar(subset["Line"], subset["Word Count"], label=f"Speaker {speaker}", alpha=0.6, color=speaker_colors[speaker])

            ax2.plot(df_chart["Line"], df_chart["Filler Ratio"], color='tab:orange', marker='o', label="Filler Ratio")

            ax1.set_xlabel("Line Number")
            ax1.set_ylabel("Word Count", color='tab:blue')
            ax2.set_ylabel("Filler Ratio", color='tab:orange')
            fig.suptitle("Word Count and Filler Ratio per Turn")

            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            st.pyplot(fig)
    else:
        st.warning("No data available for selected speaker/metric.")


    st.markdown("---")

    # ---------------- Sentiment Trend Chart ----------------
    st.subheader("📉 Sentiment Score Trend")

    trend_view = st.radio("👥 Select Speaker for Trend", ["Both", "A", "B"], index=0, horizontal=True)

    df_trend = pd.DataFrame([
        {
            "Line": i + 1,
            "Speaker": d["speaker"],
            "Sentiment Score": d["sentiment_score"]
        }
        for i, d in enumerate(result_json["dialogue"])
    ])

    if not df_trend.empty:
        plt.figure(figsize=(10, 4))

        speaker_colors = {"A": "tab:blue", "B": "lightskyblue"}

        if trend_view == "Both":
            for speaker in ["A", "B"]:
                subset = df_trend[df_trend["Speaker"] == speaker]
                plt.plot(subset["Line"], subset["Sentiment Score"], marker='o',
                        label=f"Speaker {speaker}", color=speaker_colors[speaker])
        else:
            subset = df_trend[df_trend["Speaker"] == trend_view]
            plt.plot(subset["Line"], subset["Sentiment Score"], marker='o',
                    label=f"Speaker {trend_view}", color=speaker_colors[trend_view])

        # 表情符号
        plt.text(df_trend["Line"].min() - 0.5, 1.05, "😊", fontsize=14)
        plt.text(df_trend["Line"].min() - 0.5, -1.1, "😠", fontsize=14)

        # 视觉修饰
        plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.title("Sentiment Score per Turn")
        plt.xlabel("Line Number")
        plt.ylim(-1.2, 1.2)
        plt.yticks([])  # 隐藏纵轴数字
        plt.legend()
        st.pyplot(plt)
    else:
        st.warning("No sentiment trend data to display.")



# ---------------------- Sidebar: Settings -----------------------
st.sidebar.header("⚙️ Settings")

# Custom Filler Words
st.sidebar.markdown("### 📝 Custom Filler Words")
custom_fillers_input = st.sidebar.text_input(
    "Comma-separated filler words",
    value="um, like, you know"
)
filler_set = {w.strip().lower() for w in custom_fillers_input.split(",") if w.strip()}

st.sidebar.markdown("### ✅ Active Filler Set")
st.sidebar.write(", ".join(sorted(filler_set)) if filler_set else "_None_")

st.sidebar.markdown("---")

# File Upload
st.sidebar.markdown("### 📤 Upload Transcript File")
uploaded_file = st.sidebar.file_uploader("Choose a .txt file", type=["txt"])


# ---------------------- File Handling -----------------------
filepath = "transcript.txt"  # Default fallback

def is_directory_writable(path: str) -> bool:
    try:
        test_file = os.path.join(path, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception as e:
        return False

if uploaded_file is not None:
    try:
        if uploaded_file.size > 1_048_576:
            st.error("❌ File too large. Maximum size allowed is 1MB.")
            st.stop()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"transcript_{timestamp}.txt"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        filepath = os.path.join(UPLOAD_DIR, saved_filename)

        if not is_directory_writable(UPLOAD_DIR):
            st.error(f"❌ Upload folder `{UPLOAD_DIR}` is not writable. Please check permissions.")
            st.stop()

        transcript_lines = uploaded_file.getvalue().decode("utf-8").splitlines()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript_lines))
            
        st.sidebar.success("✅ File uploaded successfully.")
    except Exception as e:
        st.sidebar.error(f"❌ Failed to process uploaded file: {e}")
        st.stop()
else:
    st.sidebar.info("ℹ️ No file uploaded. Analyze default `transcript.txt`.")

# ---------------------- Load Transcript -----------------------
with st.spinner("🔍 Loading transcript..."):
    transcript = load_transcript(filepath)
    if not transcript:
        st.error("❌ Failed to load or parse the transcript. Please check format.")
        st.stop()

# ---------------------- Analysis -----------------------
with st.spinner("🧠 Running sentiment and filler analysis..."):
    enriched_data = []
    total_sentiment = 0.0
    total_filler = 0.0

    for entry in transcript:
        text = entry["text"]
        sentiment_label, sentiment_score = compute_sentiment(text)
        filler_ratio, text_md = compute_filler_ratio(text, fillers=filler_set)

        enriched_entry = {
            **entry,
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "filler_ratio": filler_ratio,
            "text_md": text_md
        }

        enriched_data.append(enriched_entry)
        total_sentiment += sentiment_score
        total_filler += filler_ratio

    avg_sentiment = round(total_sentiment / len(enriched_data), 4)
    avg_filler = round(total_filler / len(enriched_data), 4)

    result = {
        "dialogue": enriched_data,
        "averages": {
            "sentiment_score": avg_sentiment,
            "filler_ratio": avg_filler
        }
    }

    json_result = json.dumps(result, indent=2, ensure_ascii=False)

st.success("✅ Analysis complete!")

# ---------------------- Debug button -----------------------
with st.expander("🐞 Show JSON", expanded=False):
    st.code(json_result, language="json")


render_dashboard(result)
