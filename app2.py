import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import tempfile
import whisper
import ffmpeg
from textblob import TextBlob
import re
import os
from transformers import pipeline
import altair as alt
import pandas as pd
import mediapipe as mp
import spacy
import wave
import spacy_streamlit
import joblib
import librosa
import soundfile as sf
from skimage.transform import resize
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
# trained models
# emotion_model = load_model('deep_face.h5')
emotion_audio_model = joblib.load('Emotion_Audio_Model.pkl')

# Labels the models can predict
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#tone_labels = ['calm', 'happy', 'disgust', 'neutral']
tone_labels = ['Disgusted/Fearful/Sad',  'Neutral/Happy']

# Initialize Hugging Face emotion detection pipeline
text_emotion_detector = pipeline('sentiment-analysis', model='j-hartmann/emotion-english-distilroberta-base')

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()
binary_labels = {0: 'Disgusted/Fearful/Sad', 1: 'Neutral/Happy'}

# Load spaCy model for NER
nlp = spacy.load('en_core_web_sm')

def extract_entities(transcription):
    text = transcription["text"]
    doc = nlp(text)
    entities = {'PERSON': [], 'ORG': [], 'JOB': []}

    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            entities['PERSON'].append(ent.text)
        elif ent.label_ == 'ORG':
            entities['ORG'].append(ent.text)
        # You might need a custom model or additional logic for job titles

    return entities
def perform_topic_modeling(transcription, n_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([transcription['text']])
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
    
    return topics
# Custom CSS
st.markdown("""
    <style>
    .title {
        color: #4CAF50;
        font-family: 'Arial', sans-serif;
        font-size: 2em;
    }
    .subheader {
        color: #FFC107;
        font-family: 'Arial', sans-serif;
        font-size: 1.5em;
    }
    .text {
        color: #FFFFFF;
        font-family: 'Helvetica', sans-serif;
        font-size: 1.2em;
    }
    .highlight {
        color: #FF5722;
        font-weight: bold;
    .sum{
        color: #FF5726;
        font-weight: bold;        
    }
    </style>
    """, unsafe_allow_html=True)

# face box detection
def detect_faces(frame):
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            faces.append(bbox)
    return faces
# audio analysis function
def pause_detection(audio_path):
    TOP_DB_LEVEL = 30

    x, sr = librosa.load(audio_path)
    y = librosa.amplitude_to_db(abs(x))
    refDBVal = np.max(y)
    n_fft = 2048
    S = librosa.stft(x, n_fft=n_fft, hop_length=n_fft // 2)
    D = librosa.amplitude_to_db(np.abs(S), ref=np.max)

    topDB = TOP_DB_LEVEL

    nonMuteSections = librosa.effects.split(x, top_db=topDB)
    total_time = nonMuteSections[-1][-1] / sr

    initial_pause = nonMuteSections[0][0] / sr
    initial_pause_percent = initial_pause * 100 / total_time

    mute = nonMuteSections[0][0]
    for i in range(1, len(nonMuteSections)):
        mute += (nonMuteSections[i][0] - nonMuteSections[i - 1][1])
    mute = mute / sr

    mute_percent = (mute * 100) / total_time

    return initial_pause_percent, mute_percent

# predict expression
def predict_expression(frame, faces):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    expressions = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        resized_frame = cv2.resize(face_roi, (48, 48))
        img_array = img_to_array(resized_frame) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)
        predictions = emotion_model.predict(img_array)
        confidence_scores = {class_labels[i]: predictions[0][i] for i in range(len(class_labels))}
        predicted_expression = class_labels[np.argmax(predictions)]
        expressions.append((predicted_expression, confidence_scores, (x, y, w, h)))  # Return label, confidence scores, and bounding box
    return expressions

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_path='audio.wav'):
    ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)
    return audio_path

# Function to get audio duration
def get_audio_duration(audio_path):
    with wave.open(audio_path, 'r') as audio_file:
        frames = audio_file.getnframes()
        rate = audio_file.getframerate()
        duration = frames / float(rate)
    return duration

# Function to transcribe audio using Whisper
def transcribe_audio_with_confidence(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, verbose=True)
    return result

# Function to analyze confidence in transcription
def analyze_confidence(transcription):
    segments = transcription['segments']
    confidences = [(segment['text'], segment['avg_logprob']) for segment in segments]
    return confidences

def summarize_transcript(transcription):
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Assuming transcription is a dictionary with the text under 'text' key
    text = transcription.get('text', '')

    # Summarize the text
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    
    return summary[0]['summary_text']
# Function to analyze speech text
def speech_analysis(text, duration):
    words = text.split()
    num_words = len(words)
    num_sentences = len(re.findall(r'[.!?]+', text))
    speech_rate = num_words / (duration / 60)  # words per minute
    sentiment = TextBlob(text).sentiment

    # Define filler words with their respective weights
    filler_words = {
        'um': 1.0,
        'uh': 1.0,
        'like': 0.8,
        'you know': 0.8,
        'so': 0.5,
        'well': 0.5,
    }

    num_fillers = sum(text.lower().count(filler) * weight for filler, weight in filler_words.items())
    filler_ratio = num_fillers / num_words

    disfluencies = num_fillers + text.count(',')

    analysis = {
        'num_words': num_words,
        'num_sentences': num_sentences,
        'speech_rate': speech_rate,
        'sentiment_polarity': sentiment.polarity,
        'sentiment_subjectivity': sentiment.subjectivity,
        'disfluencies': disfluencies,
        'filler_ratio': filler_ratio  # Add this to the analysis
    }

    return analysis

# Function to split text into chunks
def split_text_into_chunks(text, max_length=512):
    words = text.split()
    for i in range(0, len(words), max_length):
        yield ' '.join(words[i:i + max_length])

# Function to perform NER using spaCy
def perform_ner(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities
def score_audio_confidence(analysis, hf_confidence_level, audio_emotion):
    # Scoring speech rate (average speaking rate is between 110 and 160 wpm)
    if analysis['speech_rate'] < 110:
        speech_rate_score = 0.5
    elif 110 <= analysis['speech_rate'] <= 160:
        speech_rate_score = 1.0
    else:
        speech_rate_score = 0.7

    # Scoring sentiment polarity (scale from -1 to 1)
    sentiment_score = (analysis['sentiment_polarity'] + 1) / 2

    # Scoring disfluencies (fewer disfluencies is better)
    if analysis['disfluencies'] <= 5: # need improvement
        disfluencies_score = 1.0
    else:
        disfluencies_score = 0.5

    # Scoring filler word ratio (fewer filler words is better)
    if analysis['filler_ratio'] <= 0.02:
        filler_word_score = 1.0
    elif 0.02 < analysis['filler_ratio'] <= 0.05:
        filler_word_score = 0.7
    else:
        filler_word_score = 0.5

    # Confidence level scoring
    confidence_level_score = 1.0 if hf_confidence_level == 'High' else 0.5

    # Aggregate score
    total_score = (speech_rate_score + sentiment_score + filler_word_score + disfluencies_score +
                   confidence_level_score) / 5.0
    total_score = total_score * 100
    return {
        'total_score': total_score,
        'speech_rate_score': speech_rate_score,
        'sentiment_score': sentiment_score,
        'disfluencies_score': disfluencies_score,
        'filler_word_score': filler_word_score,
        'confidence_level_score': confidence_level_score
    }
def score_video_confidence(face_expressions, audio_emotion):
    # Facial emotion scoring
    if face_expressions:
        dominant_face_expression = max(face_expressions, key=lambda x: max(x[1].values()))[0]
        if dominant_face_expression in ['Happy', 'Surprise']:
            face_expression_score = 1.0
        elif dominant_face_expression == 'Neutral':
            face_expression_score = 0.7
        else:
            face_expression_score = 0.5
    else:
        face_expression_score = 0.5

    # Audio emotion scoring
    if audio_emotion in ['Neutral/Happy']:
        audio_emotion_score = 1.0
    else:
        audio_emotion_score = 0.5

    # Aggregate score
    total_score = (face_expression_score + audio_emotion_score) / 2.0
    total_score = total_score * 100
    return {
        'total_score': total_score,
        'face_expression_score': face_expression_score,
        'audio_emotion_score': audio_emotion_score
    }
# Function to determine confidence level
def enhanced_audio_analysis(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    
    return {
        'chroma_stft': np.mean(chroma_stft),
        'rmse': np.mean(rmse),
        'spec_cent': np.mean(spec_cent),
        'spec_bw': np.mean(spec_bw),
        'rolloff': np.mean(rolloff),
        'zcr': np.mean(zcr),
        'mfcc': [np.mean(coeff) for coeff in mfcc]
    }
def determine_confidence_level(video_path):
    audio_path = extract_audio_from_video(video_path)
    duration = get_audio_duration(audio_path)
    transcription = transcribe_audio_with_confidence(audio_path)
    confidences = analyze_confidence(transcription)
    combined_text = ' '.join([text for text, _ in confidences])
    analysis = speech_analysis(combined_text, duration)
    
    classifier = pipeline('sentiment-analysis', model='bhadresh-savani/distilbert-base-uncased-emotion')
    hf_results = []
    for chunk in split_text_into_chunks(combined_text, max_length=512):
        hf_results.extend(classifier(chunk[:512]))

    hf_confidence_score = max(hf_results, key=lambda x: x['score'])
    hf_confidence_level = 'High' if hf_confidence_score['label'] in ['joy', 'confidence'] else 'Low'
    hiring_keywords = find_hiring_keywords(transcription)
    entities = extract_entities(transcription)
    topics = perform_topic_modeling(transcription)
    # Perform NER analysis
    ner_results = perform_ner(combined_text)

    # Facial emotion analysis
    video_capture = cv2.VideoCapture(video_path)
    face_expressions = []
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        faces = detect_faces(frame)
        if faces:
            expressions = predict_expression(frame, faces)
            face_expressions.extend(expressions)
    video_capture.release()

    # Extract audio features
    audio_data, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfccs_resized = resize(mfccs, (128, 128), mode='constant')
    mfccs_expanded = np.expand_dims(mfccs_resized, axis=0)
    mfccs_expanded = np.expand_dims(mfccs_expanded, axis=-1)
    audio_predictions = emotion_audio_model.predict(mfccs_expanded)
    audio_emotion = tone_labels[np.argmax(audio_predictions)]
    audio_confidence_scores = {tone_labels[i]: audio_predictions[0][i] for i in range(len(tone_labels))}

    initial_pause_percent, mute_percent = pause_detection(audio_path)
    enhanced_features = enhanced_audio_analysis(audio_path)
    audio_scores = score_audio_confidence(analysis, hf_confidence_level, audio_emotion)
    video_scores = score_video_confidence(face_expressions, audio_emotion)
    summary = summarize_transcript(transcription)
    return {
        'transcription': transcription,
        'text_analysis': analysis,
        'duration': duration,
        'sentiment_analysis': analysis,
        'hf_confidence_level': hf_confidence_level,
        'ner_results': ner_results,
        'face_expressions': face_expressions,
        'audio_emotion': audio_emotion,
        'initial_pause_percent': initial_pause_percent,
        'mute_percent': mute_percent,
        'enhanced_audio_features': enhanced_features,
        'audio_confidence_scores': audio_confidence_scores,
        'audio_scores': audio_scores,
        'video_scores': video_scores,
        'hiring_keywords': hiring_keywords,
        'entities': entities,
        'topics': topics,
        'summary': summary
    }
def qa(transcript):
    qa_pipeline= pipeline("question-answering")
    questions = {
    "current_workplace": "Where is John Doe currently working?",
    "previous_workplace": "Where did John Doe previously work?",
    "education": "Where did John Doe study?",
    "university": "Which university did John Doe attend?",
    "role": "What is John Doe's current role?",
    "experience": "How many years of experience does John Doe have?",
    "technoligies": "What technologies does John Doe know?",
    }
    for key, question in questions.items():
        result = qa_pipeline(question=question, context=transcript['text'])
        print(f"{key.capitalize().replace('_', ' ')}: {result['answer']}")

def score_analysis(analysis, confidence_level, face_expressions, audio_emotion):
    # Scoring speech rate (average speaking rate is between 110 and 160 wpm)
    if analysis['speech_rate'] < 110:
        speech_rate_score = 0.5
    elif 110 <= analysis['speech_rate'] <= 160:
        speech_rate_score = 1.0
    else:
        speech_rate_score = 0.7

    # Scoring sentiment polarity (scale from -1 to 1)
    sentiment_score = (analysis['sentiment_polarity'] + 1) / 2

    # Scoring disfluencies (fewer disfluencies is better)
    if analysis['disfluencies'] <= 5: # need improvement
        disfluencies_score = 1.0
    else:
        disfluencies_score = 0.5

    # Scoring filler word ratio (fewer filler words is better)
    if analysis['filler_ratio'] <= 0.02:
        filler_word_score = 1.0
    elif 0.02 < analysis['filler_ratio'] <= 0.05:
        filler_word_score = 0.7
    else:
        filler_word_score = 0.5

    # Confidence level scoring
    confidence_level_score = 1.0 if confidence_level == 'High' else 0.5

    # Facial emotion scoring
    if face_expressions:
        dominant_face_expression = max(face_expressions, key=lambda x: max(x[1].values()))[0]
        if dominant_face_expression in ['Happy', 'Surprise']:
            face_expression_score = 1.0
        elif dominant_face_expression == 'Neutral':
            face_expression_score = 0.7
        else:
            face_expression_score = 0.5
    else:
        face_expression_score = 0.5

    # Audio emotion scoring
    if audio_emotion in ['Neutral/Happy']:
        audio_emotion_score = 1.0
    else:
        audio_emotion_score = 0.5

    # Aggregate score
    total_score = (speech_rate_score + sentiment_score + filler_word_score + disfluencies_score +
                   confidence_level_score + face_expression_score + audio_emotion_score) / 7.0
    total_score = total_score*100
    return {
        'total_score': total_score,
        'speech_rate_score': speech_rate_score,
        'sentiment_score': sentiment_score,
        'disfluencies_score': disfluencies_score,
        'filler_word_score': filler_word_score,
        'confidence_level_score': confidence_level_score,
        'face_expression_score': face_expression_score,
        'audio_emotion_score': audio_emotion_score
    }
def create_pie_chart(scores, category):
        data = pd.DataFrame({
            'Category': [category, 'Other'],
            'Score': [scores[category], 1 - scores[category]]
        })
        pie_chart = alt.Chart(data).mark_arc().encode(
            theta = alt.Theta(field='Score:Q'),
            color = alt.Color(field='Category:N', scale=alt.Scale(range=['#1f77b4', '#d3d3d3'])), 
            tooltip=['Category', 'Score']
        ).properties(
            title=f'{category} Score'
        )
        return pie_chart
def detect_gender(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    sf.write('temp.wav', y_resampled, 16000)
    gender_model = joblib.load('gender_model.pkl')
    features = librosa.feature.mfcc(y=y_resampled, sr=16000, n_mfcc=13).T
    mean_features = np.mean(features, axis=0).reshape(1, -1)
    gender_prediction = gender_model.predict(mean_features)
    return 'Male' if gender_prediction == 1 else 'Female'

def find_hiring_keywords(transcription):
    keywords = ['experience', 'company', 'role', 'skill', 'qualification', 'job', 'position']
    matches = {keyword: transcription.get('text','').lower().count(keyword) for keyword in keywords}
    return matches
def main():
    st.markdown("<h1 class='title'>Video Confidence Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='subheader'>Upload a video for analysis</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        st.markdown("<h2 class='subheader'>Uploaded Video</h2>", unsafe_allow_html=True)
        st.video(tmp_file_path)

        analysis_result = determine_confidence_level(tmp_file_path)
        scores = score_analysis(analysis_result['sentiment_analysis'], analysis_result['hf_confidence_level'],
                                analysis_result['face_expressions'], analysis_result['audio_emotion'])
        #audio_scores = analysis_result['audio_scores']
        #video_scores = analysis_result['video_scores']
        #st.markdown("<h3 class='highlight'>Audio Confidence Score: {:.2f}%</h3>".format(audio_scores['total_score']), unsafe_allow_html=True)
        #st.markdown("<h3 class='highlight'>Video Confidence Score: {:.2f}%</h3>".format(video_scores['total_score']), unsafe_allow_html=True)
        st.markdown("<h3 class='highlight'>Overall Confidence Score: {:.2f}%</h3>".format(scores['total_score']), unsafe_allow_html=True)
        #st.markdown("<h2 class='subheader'>Transcription Analysis</h2>", unsafe_allow_html=True)
        #st.markdown("<h3>Hiring Keywords Found:</h3>")
        #st.write(analysis_result['hiring_keywords'])
        
        #st.markdown("<h3>Entities Extracted:</h3>")
        #st.write(analysis_result['entities'])
        
        #st.markdown("<h3>Topics Identified:</h3>")
        #for topic, words in analysis_result['topics'].items():
            #st.write(f"{topic}: {', '.join(words)}")
        # Add the current score to a pandas DataFrame
        if 'score_data' not in st.session_state:
            st.session_state['score_data'] = pd.DataFrame(columns=['Timestamp', 'Score'])

        new_entry = pd.DataFrame({'Timestamp': [datetime.datetime.now()], 'Score': [scores['total_score']]})
        st.session_state['score_data'] = pd.concat([st.session_state['score_data'], new_entry], ignore_index=True)
        top_candidate_index = st.session_state['score_data']['Score'].idxmax()
        max_score = st.session_state['score_data']['Score'].max()
        selected_candidate_index = st.session_state['score_data'][st.session_state['score_data']['Score'] == max_score].index[0]
        st.markdown(f"<h3 class='sum'>Summary: {analysis_result['summary']}</h3>", unsafe_allow_html=True)
        
        st.markdown(f"<h3 class='highlight'>Selected Candidate: {selected_candidate_index} with a overall Score of {max_score:.2f}%. This candidate had an overall higher score than previous candidates.</h3>", unsafe_allow_html=True)
        # Display the DataFrame 
        st.markdown("<h2 class='subheader'>Confidence Scores Over Time</h2>", unsafe_allow_html=True)
        st.dataframe(st.session_state['score_data'])
        top_score = st.session_state['score_data']['Score'].max()
        qa(analysis_result['transcription'])

        # Plot the histogram using Altair
        histogram = alt.Chart(st.session_state['score_data'].reset_index()).mark_bar().encode(
            x=alt.X('index:N', title='Index'),
            y=alt.Y('Score:Q', title='Confidence Score'), 
            color=alt.condition(
                alt.datum.index == top_candidate_index + 1,
                alt.value('orange'),  # Highlight the top candidate
                alt.value('steelblue')
            ),
        ).properties(
            title='Confidence Score Distribution',
            width=800,
            height=400
        )
        

        st.altair_chart(histogram, use_container_width=True)
        st.sidebar.header("Detailed Breakdown")
        categories = ['speech_rate_score', 'sentiment_score', 'disfluencies_score', 'filler_word_score', 'face_expression_score', 'audio_emotion_score','confidence_level_score',]
        
        # Prepare data for the pie chart
        pie_data = []
        for category in categories:
            pie_data.append({'Category': category.replace('_', ' ').title(), 'Score': scores[category]})

        pie_data_df = pd.DataFrame(pie_data)
        #pie_data_df.loc[len(pie_data_df)] = ['Other', 1 - pie_data_df['Score'].sum()]  # Add 'Other' category

        pie_chart = alt.Chart(pie_data_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field='Score', type='quantitative'),
            color=alt.Color(field='Category', type='nominal', scale=alt.Scale(range=['#1f77b4', '#d3d3d3', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])),
            tooltip=['Category', 'Score']
        ).properties(
            title='Overall Confidence Score Breakdown'
        )

        st.sidebar.altair_chart(pie_chart, use_container_width=True)




        

        # Sidebar for detailed breakdown
        st.sidebar.header("Detailed Breakdown")
        with st.sidebar.expander("Video Confidence Score"):
            video_scores = analysis_result['video_scores']
            st.write(f"**Video Confidence Score:** {video_scores['total_score']}")
        with st.sidebar.expander("Audio Confidence Score"):
            audio_scores = analysis_result['audio_scores']
            st.write(f"**Audio Confidence Score:** {round(audio_scores['total_score'])}")
        with st.sidebar.expander("Textual Analysis"):
            st.write("**Textual Analysis:**")
            analysis = analysis_result['text_analysis'] 
            st.write(f"**Number of Words:** {analysis['num_words']}")
            st.write(f"**Number of Sentences:** {analysis['num_sentences']}")
            st.write(f"**Speech Rate:** {analysis['speech_rate']:.2f} words per minute - This measures how quickly the person spoke.")
            st.write(f"**Sentiment Polarity:** {analysis['sentiment_polarity']:.2f} - This shows if the text is positive or negative.")
            st.write(f"**Sentiment Subjectivity:** {analysis['sentiment_subjectivity']:.2f} - This indicates how subjective or objective the text is.")
            st.write(f"**Disfluencies:** {analysis['disfluencies']} - This counts filler words and pauses, which can impact clarity.")
            st.write(f"**Filler Word Ratio:** {analysis['filler_ratio']:.2%} - This shows how often filler words are used, which might affect the speech's fluency.")

        with st.sidebar.expander("Facial Expression Analysis"):
            st.write("**Facial Expression Analysis:**")
            if analysis_result['face_expressions']:
                expressions = [exp[0] for exp in analysis_result['face_expressions']]
                most_common_expression = max(set(expressions), key=expressions.count)
                st.write(f"The most common facial expression detected in the video was: {most_common_expression}.")
                st.write("This gives an overall sense of the dominant emotion expressed throughout the video.")
            else:
                st.write("No facial expressions were detected or analyzed.")

        with st.sidebar.expander("Audio Emotion Analysis"):
            st.write("**Audio Emotion Analysis:**")
            st.write(f"**Predicted Audio Emotion:** {analysis_result['audio_emotion']}")
            st.write("This reflects the general emotion conveyed in the audio portion of the video.")
            st.write("**Confidence Scores:**")
            for label, score in analysis_result['audio_confidence_scores'].items():
                st.write(f"{label}: {'High' if score > 0.5 else 'Low'}")

        with st.sidebar.expander("Voice Analysis"):
            enhanced_features = analysis_result['enhanced_audio_features']
            st.write("**Pause Detection Analysis:**")
            st.write(f"**Initial Pause Percent:** {analysis_result['initial_pause_percent']:.2f}% - Indicates how much of the beginning of the audio was silent.")
            st.write(f"**Mute Percent:** {analysis_result['mute_percent']:.2f}% - Shows the percentage of time the audio was completely silent.")
            st.write(f"**RMSE:** {enhanced_features['rmse']:.2f} - This measures the loudness of the audio, indicating how much energy the sound has. A lower RMSE indicates a quieter speech, while a higher RMSE suggests stronger, more confident speech. Based on the score it is {'**Lower**' if enhanced_features['rmse'] < 0.1 else '**Higher**'}")
        #st.sidebar.write(f"**Speech Rate Score:** {scores['speech_rate_score']:.2f}")
        #st.sidebar.write(f"**Sentiment Score:** {scores['sentiment_score']:.2f}")
        #st.sidebar.write(f"**Disfluencies Score:** {scores['disfluencies_score']:.2f}")
        #st.sidebar.write(f"**Filler Word Score:** {scores['filler_word_score']:.2f}")
        #st.sidebar.write(f"**Confidence Level Score:** {scores['confidence_level_score']:.2f}")
        #st.sidebar.write(f"**Face Expression Score:** {scores['face_expression_score']:.2f}")
        #st.sidebar.write(f"**Audio Emotion Score:** {scores['audio_emotion_score']:.2f}")

if __name__ == "__main__":
    main()


