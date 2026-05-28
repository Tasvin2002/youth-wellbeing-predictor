"""
=============================================================================
CAI20303 MACHINE LEARNING - FINAL PROJECT
app.py  —  Streamlit deployment (no TensorFlow, uses sklearn MLPRegressor)
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Youth Digital Wellbeing Predictor", page_icon="🧠", layout="centered")

@st.cache_resource
def load_artifacts():
    model     = joblib.load("model/ann_improved_model.pkl")
    scaler    = joblib.load("model/scaler.pkl")
    feat_cols = joblib.load("model/feature_columns.pkl")
    ord_encs  = joblib.load("model/ordinal_encoders.pkl")
    nom_feats = joblib.load("model/nominal_features.pkl")
    metrics   = joblib.load("model/metrics.pkl")
    return model, scaler, feat_cols, ord_encs, nom_feats, metrics

model, scaler, feat_cols, ord_encs, nom_feats, metrics = load_artifacts()

def classify(score):
    if score >= 75: return "Good Wellbeing",     "🟢", "#27AE60"
    elif score >= 55: return "Moderate Wellbeing","🟡", "#F39C12"
    elif score >= 35: return "At Risk",           "🟠", "#E67E22"
    else:             return "High Risk",          "🔴", "#E74C3C"

def recommendations(inputs):
    r = []
    if inputs["social_media_hours"] > 4:
        r.append("📵 Reduce social media to under 2 hours/day — you are currently over 4 hours.")
    if inputs["sleep_hours"] < 7:
        r.append("😴 Aim for 7–9 hours of sleep per night for optimal cognitive function.")
    if inputs["stress_level"] > 6:
        r.append("🧘 High stress detected — consider mindfulness, exercise, or speaking with a counsellor.")
    if inputs["study_hours_per_week"] < 10:
        r.append("📚 Increase study hours — consistent academic effort improves wellbeing.")
    if inputs["short_video_hours"] > 2:
        r.append("📱 Reduce TikTok/Reels consumption — short videos are a key driver of digital overstimulation.")
    if inputs["late_night_usage"] == "Often":
        r.append("🌙 Stop using social media late at night — it disrupts sleep and increases anxiety.")
    if inputs["anxiety_score"] > 6:
        r.append("💬 Elevated anxiety — consider reaching out to a campus counsellor or mental health professional.")
    if inputs["depression_score"] > 6:
        r.append("❤️ Concerning depression score — please seek support from a trusted person or professional.")
    if inputs["digital_addiction_score"] > 15:
        r.append("🔒 Signs of digital addiction — set daily screen-time limits on your device.")
    if inputs["education_content_hours"] < 0.5:
        r.append("🎓 Spend at least 30 minutes daily on educational content to balance screen time.")
    if not r:
        r.append("✅ Your digital habits appear healthy. Keep maintaining this balance!")
    return r

# ─── UI ──────────────────────────────────────────────────────────────────────
st.title("🧠 Youth Digital Wellbeing Predictor")
st.markdown(
    "**CAI20303 Machine Learning — Final Project**  \n"
    "Predicting digital wellbeing among youth in **Selangor, Malaysia**"
)
st.divider()

with st.expander("📊 Model Performance Summary", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Linear Regression R²", f"{metrics['lr']['R2']:.4f}")
    c2.metric("ANN Initial R²",        f"{metrics['ann']['R2']:.4f}")
    c3.metric("ANN Improved R²",       f"{metrics['imp']['R2']:.4f}",
              delta=f"+{metrics['imp']['R2']-metrics['lr']['R2']:.4f} vs Baseline")
    st.caption(f"Baseline MAE: {metrics['lr']['MAE']:.4f}  |  ANN Improved MAE: {metrics['imp']['MAE']:.4f}  |  RMSE: {metrics['imp']['RMSE']:.4f}")

st.subheader("📝 Enter Your Digital Behaviour Profile")

with st.form("form"):
    st.markdown("#### 👤 Personal Information")
    c1, c2, c3 = st.columns(3)
    age         = c1.slider("Age", 15, 25, 20)
    gender      = c2.selectbox("Gender", ["Male","Female","Other"])
    urban_rural = c3.selectbox("Location", ["Urban","Rural"])

    c4, c5 = st.columns(2)
    family_income_level = c4.selectbox("Family Income Level", ["Low","Middle","High"])
    device_access       = c5.selectbox("Device Access", ["Smartphone","Both","Shared Device","Laptop"])

    st.markdown("#### 🎓 Education")
    c6, c7, c8 = st.columns(3)
    education_level     = c6.selectbox("Education Level", ["School","Diploma","Graduate","Postgraduate","PhD","Dropout"])
    field_of_study      = c7.selectbox("Field of Study", ["Science","Arts","Engineering","Business","Health","None"])
    academic_motivation = c8.slider("Academic Motivation (1–10)", 1.0, 10.0, 5.0, 0.1)

    st.markdown("#### 📱 Social Media Behaviour")
    c9, c10 = st.columns(2)
    internet_access_hours          = c9.slider("Internet Access Hours/Day", 0.0, 12.0, 5.0, 0.1)
    social_media_hours             = c10.slider("Social Media Hours/Day", 0.0, 9.0, 3.0, 0.1)

    c11, c12, c13 = st.columns(3)
    sessions_per_day               = c11.slider("Sessions Per Day", 1, 20, 5)
    average_session_length_minutes = c12.slider("Avg Session Length (min)", 1.0, 120.0, 30.0, 1.0)
    late_night_usage               = c13.selectbox("Late Night Usage", ["Never","Sometimes","Often"])

    st.markdown("#### 🎬 Content Consumption (hours/day)")
    c14, c15, c16, c17 = st.columns(4)
    education_content_hours       = c14.slider("Educational", 0.0, 5.0, 0.5, 0.1)
    short_video_hours             = c15.slider("Short Videos", 0.0, 5.0, 1.0, 0.1)
    entertainment_content_hours   = c16.slider("Entertainment", 0.0, 5.0, 1.0, 0.1)
    news_content_hours            = c17.slider("News", 0.0, 3.0, 0.3, 0.1)

    st.markdown("#### 💬 Engagement")
    c18, c19, c20 = st.columns(3)
    likes_given_per_day           = c18.slider("Likes/Day", 0.0, 200.0, 30.0, 1.0)
    comments_written_per_day      = c19.slider("Comments/Day", 0.0, 30.0, 3.0, 0.1)
    posts_created_per_week        = c20.slider("Posts/Week", 0.0, 20.0, 2.0, 0.5)

    st.markdown("#### 🧠 Cognitive & Academic")
    c21, c22, c23 = st.columns(3)
    online_learning_hours         = c21.slider("Online Learning Hours/Day", 0.0, 10.0, 1.0, 0.1)
    study_hours_per_week          = c22.slider("Study Hours/Week", 0.0, 40.0, 15.0, 0.5)
    class_attendance_rate         = c23.slider("Class Attendance (%)", 0.0, 100.0, 85.0, 1.0)

    c24, c25, c26 = st.columns(3)
    brain_rot_index               = c24.slider("Brain Rot Index (0–50)", 0.0, 50.0, 15.0, 0.5)
    attention_span_minutes        = c25.slider("Attention Span (min)", 10.0, 60.0, 45.0, 1.0)
    productivity_score            = c26.slider("Productivity Score (0–10)", 0.0, 10.0, 6.0, 0.1)

    st.markdown("#### 😟 Psychological")
    c27, c28, c29, c30 = st.columns(4)
    sleep_hours                   = c27.slider("Sleep Hours/Night", 4.0, 9.0, 7.0, 0.5)
    stress_level                  = c28.slider("Stress Level (1–10)", 1.0, 10.0, 5.0, 0.1)
    anxiety_score                 = c29.slider("Anxiety (1–10)", 1.0, 10.0, 4.0, 0.1)
    depression_score              = c30.slider("Depression (1–10)", 1.0, 10.0, 3.0, 0.1)

    st.markdown("#### 💸 Digital Spending & Risks")
    c31, c32 = st.columns(2)
    ads_viewed_per_day            = c31.slider("Ads Viewed/Day", 0.0, 200.0, 50.0, 1.0)
    ads_clicked_per_week          = c32.slider("Ads Clicked/Week", 0.0, 30.0, 5.0, 0.5)

    c33, c34 = st.columns(2)
    impulse_purchase_score        = c33.slider("Impulse Purchase Score (0–15)", 0.0, 15.0, 5.0, 0.1)
    digital_spending_per_month    = c34.slider("Digital Spending/Month (USD)", 0.0, 300.0, 50.0, 1.0)

    c35, c36 = st.columns(2)
    digital_addiction_score       = c35.slider("Digital Addiction Score (0–30)", 0.0, 30.0, 10.0, 0.1)
    cyberbullying_exposure        = c36.selectbox("Cyberbullying Exposure", ["No","Yes"])
    adult_content_exposure        = st.selectbox("Adult Content Exposure", ["No","Yes"])

    submit = st.form_submit_button("🔍 Predict My Wellbeing Score", use_container_width=True)

if submit:
    raw = {
        "age": age, "internet_access_hours": internet_access_hours,
        "academic_motivation": academic_motivation, "online_learning_hours": online_learning_hours,
        "social_media_hours": social_media_hours, "sessions_per_day": sessions_per_day,
        "average_session_length_minutes": average_session_length_minutes,
        "education_content_hours": education_content_hours,
        "short_video_hours": short_video_hours,
        "entertainment_content_hours": entertainment_content_hours,
        "news_content_hours": news_content_hours,
        "likes_given_per_day": likes_given_per_day,
        "comments_written_per_day": comments_written_per_day,
        "posts_created_per_week": posts_created_per_week,
        "brain_rot_index": brain_rot_index,
        "attention_span_minutes": attention_span_minutes,
        "study_hours_per_week": study_hours_per_week,
        "class_attendance_rate": class_attendance_rate,
        "productivity_score": productivity_score,
        "sleep_hours": sleep_hours, "stress_level": stress_level,
        "anxiety_score": anxiety_score, "depression_score": depression_score,
        "ads_viewed_per_day": ads_viewed_per_day,
        "ads_clicked_per_week": ads_clicked_per_week,
        "impulse_purchase_score": impulse_purchase_score,
        "digital_spending_per_month": digital_spending_per_month,
        "digital_addiction_score": digital_addiction_score,
        "family_income_level": family_income_level,
        "education_level": education_level,
        "late_night_usage": late_night_usage,
        "gender": gender, "urban_rural": urban_rural,
        "device_access": device_access, "field_of_study": field_of_study,
        "cyberbullying_exposure": cyberbullying_exposure,
        "adult_content_exposure": adult_content_exposure,
    }

    df_in = pd.DataFrame([raw])
    for col, enc in ord_encs.items():
        df_in[col] = enc.transform(df_in[[col]])
    df_in = pd.get_dummies(df_in, columns=nom_feats, drop_first=False)
    for col in feat_cols:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[feat_cols]

    score = float(model.predict(scaler.transform(df_in))[0])
    score = np.clip(score, 0, 100)
    label, emoji, color = classify(score)

    st.divider()
    st.subheader("📈 Prediction Result")
    ca, cb = st.columns([1, 2])
    with ca:
        st.metric("Wellbeing Index Score", f"{score:.1f} / 100")
    with cb:
        st.markdown(
            f"<div style='background:{color}22;border-left:6px solid {color};"
            f"padding:12px 20px;border-radius:6px'>"
            f"<span style='font-size:1.4rem'>{emoji}</span> "
            f"<strong style='font-size:1.2rem;color:{color}'>{label}</strong></div>",
            unsafe_allow_html=True
        )
    st.progress(int(score), text=f"Score: {score:.1f} / 100")
    st.caption("🟢 Good (75–100)  🟡 Moderate (55–74)  🟠 At Risk (35–54)  🔴 High Risk (0–34)")

    st.divider()
    st.subheader("💡 Personalised Recommendations")
    for rec in recommendations(raw):
        st.markdown(f"- {rec}")

    st.divider()
    st.caption("⚠️ For educational purposes only. Not clinical advice. If experiencing mental health difficulties, please consult a professional.")

st.divider()
st.caption("CAI20303 Machine Learning — Final Project | MSU | Dataset: Kaggle Global Student Digital Behavior Dataset")
