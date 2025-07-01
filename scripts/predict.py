import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import smtplib
import ssl
from email.message import EmailMessage
import streamlit as st


# Load environment variables
EMAIL_SENDER = st.secrets["general"]["email_sender"]
EMAIL_PASSWORD = st.secrets["general"]["email_password"]


# Load model and scaler
model = joblib.load("model/rul_model_xgb.pkl")
scaler = joblib.load("model/scaler.pkl")

# Sensor features (15 sensors only)
sensor_features = [
    'sensor_1', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_5',
    'sensor_6', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_10',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_16', 'sensor_17', 'sensor_18', 'sensor_19', 'sensor_20', 'sensor_21'
]


# Email alert function
def send_email_alert(receiver_email, subject, body):
    msg = EmailMessage()
    msg['From'] = EMAIL_SENDER
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.set_content(body)

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        st.success("📧 Email alert sent successfully!")
    except Exception as e:
        st.error(f"Email failed to send: {e}")

# Streamlit UI setup
st.set_page_config(page_title="AeroGuardian - Engine Failure Predictor", layout="wide")

# Sidebar settings
lang = st.sidebar.selectbox("🌐 Language", ["English", "ಕನ್ನಡ"])
sound_enabled = st.sidebar.toggle("🔊 Sound Alerts", value=True)
receiver_email = st.sidebar.text_input("📧 Enter email to receive alerts", value="")

# Translation strings
T = {
    "title": {"English": "AeroGuardian - Engine Failure Predictor", "ಕನ್ನಡ": "ಏರೋಗಾರ್ಡಿಯನ್ - ಎಂಜಿನ್ ವೈಫಲ್ಯ ಭವಿಷ್ಯವಾಣಿ"},
    "subtitle": {"English": "🔍 Predict Remaining Useful Life (RUL) from sensor data.", "ಕನ್ನಡ": "🔍 ಸಂವೇದಕ ಡೇಟಾದಿಂದ ಉಳಿದ ಉಪಯುಕ್ತ ಜೀವನವನ್ನು ಊಹಿಸಿ"},
    "input": {"English": "📥 Input Sensor Readings", "ಕನ್ನಡ": "📥 ಸಂವೇದಕ ಮೌಲ್ಯಗಳನ್ನು ನಮೂದಿಸಿ"},
    "predict": {"English": "🛠 Predict RUL", "ಕನ್ನಡ": "🛠 ಉಳಿದ ಜೀವನವನ್ನು ಊಹಿಸಿ"},
    "normal": {"English": "🟢 Engine condition normal.", "ಕನ್ನಡ": "🟢 ಎಂಜಿನ್ ಸ್ಥಿತಿ ಸಾಮಾನ್ಯವಾಗಿದೆ."},
    "caution": {"English": "🟡 Caution: Monitor engine.", "ಕನ್ನಡ": "🟡 ಎಚ್ಚರಿಕೆ: ಎಂಜಿನ್ ನಿಗಾವಹಿಸಿ."},
    "critical": {"English": "🔴 Critical: Engine failure likely soon!", "ಕನ್ನಡ": "🔴 ಗಂಭೀರ: ಎಂಜಿನ್ ಶೀಘ್ರದಲ್ಲೇ ವಿಫಲವಾಗಬಹುದು!"},
    "upload": {"English": "📤 Upload CSV for Batch Engine Prediction", "ಕನ್ನಡ": "📤 CSV ಅಪ್ಲೋಡ್ ಮಾಡಿ"},
    "csv_upload": {"English": "Upload a CSV file with sensor readings", "ಕನ್ನಡ": "ಸಂವೇದಕ ಡೇಟಾ ಹೊಂದಿರುವ CSV ಅಪ್ಲೋಡ್ ಮಾಡಿ"},
    "batch_result": {"English": "✅ Batch RUL Predictions:", "ಕನ್ನಡ": "✅ ಗುಂಪು ಊಹೆಗಳು"},
    "visualize": {"English": "📈 Visualize Sensor Trends", "ಕನ್ನಡ": "📈 ಸಂವೇದಕ ಡೇಟಾ ಗ್ರಾಫ್"},
    "select_sensors": {"English": "Select sensors to visualize:", "ಕನ್ನಡ": "ಗ್ರಾಫ್ ಮಾಡಲು ಸಂವೇದಕ ಆಯ್ಕೆಮಾಡಿ"},
    "rul_dist": {"English": "📊 Distribution of Predicted RUL", "ಕನ್ನಡ": "📊 RUL ವಿತರಣಾ ಚಾರ್ಟ್"},
    "live_mode": {"English": "⏱ Live Engine Monitoring Mode", "ಕನ್ನಡ": "⏱ ಲೈವ್ ಎಂಜಿನ್ ಮೋಡ್"},
    "start_live": {"English": "▶ Start Live Monitoring", "ಕನ್ನಡ": "▶ ಲೈವ್ ಪ್ರಾರಂಭಿಸಿ"},
    "critical_alert": {"English": "🚨 CRITICAL ALERT: Engine RUL dangerously low!", "ಕನ್ನಡ": "🚨 ಎಚ್ಚರಿಕೆ: RUL ಅಪಾಯಕರ ಮಟ್ಟದಲ್ಲಿ ಕಡಿಮೆ"},
    "warning_alert": {"English": "⚠ Caution: Maintenance required soon.", "ಕನ್ನಡ": "⚠ ಎಚ್ಚರಿಕೆ: ಶೀಘ್ರದಲ್ಲಿ ನಿರ್ವಹಣೆ ಅಗತ್ಯವಿದೆ"},
    "ok_alert": {"English": "✅ Engine is performing normally.", "ಕನ್ನಡ": "✅ ಎಂಜಿನ್ ಸರಿಯಾಗಿ ಕೆಲಸ ಮಾಡುತ್ತಿದೆ"}
}

# CSS styling
st.markdown("""
<style>
.stApp { background-color: #0d1b2a; color: #e0e1dd; font-family: 'Segoe UI', sans-serif; }
.stSidebar { background-color: #1b263b; }
.stButton>button {
    background-color: #ff6b6b; color: white; border-radius: 8px;
    font-size: 16px; padding: 10px 24px;
}
.stButton>button:hover { background-color: #fa5252; }
</style>
""", unsafe_allow_html=True)

# Logo and title
col1, col2 = st.columns([1, 8])
with col1:
    st.image("assets/logo.png.jpg", width=100)
with col2:
    st.markdown(f"<h1 style='padding-top: 20px'>{T['title'][lang]}</h1>", unsafe_allow_html=True)
st.subheader(T["subtitle"][lang])

# Sidebar sensor sliders
st.sidebar.header(T["input"][lang])
input_data = []
if st.sidebar.button("🔄 Set All to Normal"):
    input_data = [800.0] * len(sensor_features)
else:
    for sensor in sensor_features:
        input_data.append(st.sidebar.slider(sensor, 0.0, 2000.0, 800.0, step=1.0))

# Predict single RUL
if st.button(T["predict"][lang]):
    input_dict = {sensor: [value] for sensor, value in zip(sensor_features, input_data)}
    input_df = pd.DataFrame(input_dict)
    scaled_input = scaler.transform(input_df)
    predicted_rul = model.predict(scaled_input)[0]

    if predicted_rul > 100:
        st.markdown(f"<h3 style='color:lime'>📈 RUL: {predicted_rul:.2f} cycles</h3>", unsafe_allow_html=True)
        st.success(T["normal"][lang])
    elif predicted_rul > 50:
        st.markdown(f"<h3 style='color:yellow'>📈 RUL: {predicted_rul:.2f} cycles</h3>", unsafe_allow_html=True)
        st.warning(T["caution"][lang])
    else:
        st.markdown(f"<h3 style='color:red'>📈 RUL: {predicted_rul:.2f} cycles</h3>", unsafe_allow_html=True)
        st.error(T["critical"][lang])
        if receiver_email:
            send_email_alert(
                receiver_email,
                subject="🚨 AeroGuardian Critical Alert!",
                body=f"Warning! RUL is dangerously low: {predicted_rul:.2f} cycles."
            )

    if sound_enabled:
        sound = "clang_and_wobble" if predicted_rul > 100 else "beep_short" if predicted_rul > 50 else "beep_alert"
        st.markdown(f'<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/{sound}.ogg" type="audio/ogg"></audio>', unsafe_allow_html=True)

# Batch CSV
st.markdown("---")
st.subheader(T["upload"][lang])
csv_file = st.file_uploader(T["csv_upload"][lang], type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file)
    if set(sensor_features).issubset(df.columns):
        scaled_batch = scaler.transform(df[sensor_features])
        df["Predicted_RUL"] = model.predict(scaled_batch)
        st.markdown(T["batch_result"][lang])
        st.dataframe(df)

        # Visualizations
        st.markdown("---")
        st.subheader(T["visualize"][lang])
        selected_sensors = st.multiselect(T["select_sensors"][lang], sensor_features, default=sensor_features)
        if selected_sensors:
            st.line_chart(df[selected_sensors])

        st.subheader(T["rul_dist"][lang])
        fig, ax = plt.subplots()
        sns.histplot(df["Predicted_RUL"], bins=10, kde=True, color="skyblue", ax=ax)
        st.pyplot(fig)

        # Live mode
        st.markdown("---")
        st.subheader(T["live_mode"][lang])
        if st.button(T["start_live"][lang]):
            chart_area = st.empty()
            table_area = st.empty()
            alert_area = st.empty()
            live_data = pd.DataFrame(columns=sensor_features + ['Predicted_RUL'])
            for _, row in df.iterrows():
                input_row = row[sensor_features].to_frame().T
                scaled = scaler.transform(input_row)
                pred_rul = model.predict(scaled)[0]
                row_data = row[sensor_features].to_dict()
                row_data["Predicted_RUL"] = pred_rul
                live_data = pd.concat([live_data, pd.DataFrame([row_data])], ignore_index=True)
                chart_area.line_chart(live_data[selected_sensors])
                table_area.write(live_data.tail(1))
                if pred_rul <= 50:
                    alert_area.error(T["critical_alert"][lang])
                elif pred_rul <= 100:
                    alert_area.warning(T["warning_alert"][lang])
                else:
                    alert_area.success(T["ok_alert"][lang])
                time.sleep(1)
    else:
        st.error("❌ Uploaded file missing required sensors.")
