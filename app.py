import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib

from deepface import DeepFace
from PIL import Image
import tempfile

# Load your Text model
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))

# Emoji dictionary
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Functions for text emotion
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

# Function for facial emotion
def detect_emotion_image(img_path):
    result = DeepFace.analyze(img_path=img_path, actions=['emotion'], enforce_detection=False)
    return result

# Main app
def main():
    st.title("Emotion Detection App 😍😠😐")
    st.subheader("Choose between Text or Facial Emotion Recognition")

    app_mode = st.selectbox("Select Mode", ["Text Emotion Detection", "Facial Emotion Detection"])

    if app_mode == "Text Emotion Detection":
        with st.form(key='text_form'):
            raw_text = st.text_area("Type your text here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict.get(prediction, "")
                st.write("{}: {}".format(prediction, emoji_icon))
                st.write("Confidence: {:.2f}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)

    elif app_mode == "Facial Emotion Detection":
        uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                img.save(temp_file.name)
                temp_img_path = temp_file.name

            # Analyze
            with st.spinner('Analyzing emotion...'):
                result = detect_emotion_image(temp_img_path)
                emotion = result[0]['dominant_emotion']  # Access the first element of the list
                emotions = result[0]['emotion']  # Access the emotions dictionary

                st.success(f"Detected Emotion: {emotion} {emotions_emoji_dict.get(emotion, '')}")
                st.write("Detailed Probabilities:")

                emo_df = pd.DataFrame(list(emotions.items()), columns=["Emotion", "Probability"])
                fig2 = alt.Chart(emo_df).mark_bar().encode(x='Emotion', y='Probability', color='Emotion')
                st.altair_chart(fig2, use_container_width=True)

if __name__ == '__main__':
    main()