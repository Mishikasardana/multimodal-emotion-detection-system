# Multimodal Emotion Detection System

This project is a comprehensive Multimodal Emotion Detection System that leverages image and text data to identify and classify human emotions accurately. By integrating computer vision and natural language processing (NLP) techniques, this system can recognize emotions from facial expressions and textual inputs, making it a powerful tool for human-computer interaction, mental health analysis, and context-aware applications.

##  Project Overview

The system combines:
- Facial Emotion Recognition using a Deep CNN model
- Text-Based Emotion Detection using a machine learning pipeline

Each modality processes input data separately and contributes to a final emotion prediction, allowing for robust, multimodal emotion analysis.

##  Tools & Technologies Used

- Python
- OpenCV, Matplotlib, Seaborn
- TensorFlow / Keras
- Scikit-learn
- Natural Language Toolkit (NLTK)
- Jupyter Notebook
- YAML (for config management)

##  File Structure

| File/Folder | Description |
|-------------|-------------|
| facial-emotion-recognition_final.ipynb | Facial emotion detection using CNN |
| text_emotion_detection_(1)[1].ipynb | Text emotion detection model notebook |
| text_emotion_detection_(1)[1].py | Python version of text emotion detection |
| model.yaml | Configuration file |
| performance_dist.png | Model performance distribution plot |
| confusion_matrix_dcnn.png | Confusion matrix for CNN model |
| app.py | Main application script (for integration/UI) |
| *.pkl files | Serialized models |
| group_no_5_1023172&_10231726.pdf | Project report |

##  Modalities

### 1. Facial Emotion Detection
- Dataset used: [FER-2013]
- Model: Deep Convolutional Neural Network (CNN)
- Output: Emotion classes like happy, sad, angry, surprised, neutral, disgust, fear

### 2. Text Emotion Detection
- Dataset used: Emotion-tagged sentences
- Models: Logistic Regression, SVM, Random Forest (tested)
- Output: Emotion classes like happy, sad, angry, surprised, neutral, disgust, fear


##  Contributors

- [Mishika Sardana](https://github.com/Mishikasardana)
- [Ragini2424](https://github.com/Ragini2424)

## Future Work

- Combine text and image predictions using a decision fusion strategy
- Add support for audio-based emotion detection
- Deploy as a web app or mobile app for real-time use


