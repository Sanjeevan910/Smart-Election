import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av # Needed for streamlit_webrtc frame handling

# --- Replace win32com.client with a cross-platform TTS library ---
# Option 1: gTTS (Google Text-to-Speech) - requires internet connection
from gtts import gTTS
import os
def speak_web(text):
    tts = gTTS(text=text, lang='en')
    tts.save("temp_speech.mp3")
    st.audio("temp_speech.mp3", format="audio/mp3", autoplay=True)
    os.remove("temp_speech.mp3") # Clean up the file

# Option 2: pyttsx3 (offline, but might need more setup for server)
# import pyttsx3
# engine = pyttsx3.init()
# def speak_web(text):
#     engine.say(text)
#     engine.runAndWait()


# --- Your Face Recognition/Detection Models ---
# Assuming you have functions or models loaded here
# Example placeholders:
# face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Or your deep learning model
# voter_recognizer = YourVoterRecognitionModel() # Your trained model
# registered_voters = {"face_embedding_1": "John Doe", ...} # Your database of voter embeddings/features

# --- Video Transformer Class for Streamlit-WebRTC ---
class FaceVoterTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # Load your face recognition model here if it's part of the transformer
        # self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        # self.recognizer.read("trainer.yml") # Example: load trained model

        # You might need to load your pre-registered voter data (e.g., embeddings)
        # self.registered_voter_embeddings = load_voter_embeddings_from_db()

        self.last_detected_name = None # To prevent repetitive announcements

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24") # Convert WebRTC frame to OpenCV format

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)

        current_detected_name = None

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # --- Your Face Recognition Logic Here ---
            # Example: id, confidence = self.recognizer.predict(roi_gray)
            # if confidence < some_threshold:
            #     name = get_name_from_id(id) # Function to get name from ID
            #     cv2.putText(img, name, (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            #     current_detected_name = name
            # else:
            #     cv2.putText(img, "Unknown", (x+5,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Placeholder for your actual recognition logic
            # For demonstration, let's just say "Face Detected"
            name = "Voter Detected"
            cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            current_detected_name = name # Or the actual recognized name

        # This part of the speech should ideally happen outside the transformer
        # or be triggered by a state change passed back to Streamlit app context
        # For a simple demo, we can put it here, but it might run too often.
        if current_detected_name and current_detected_name != self.last_detected_name:
            st.session_state.detected_voter_name = current_detected_name # Store in session state
            self.last_detected_name = current_detected_name


        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit App Layout ---
st.title("Smart Election Voting System (Web Demo)")
st.write("Ensure your webcam is enabled. The system will detect faces in real-time.")

# Initialize session state for voter name if not present
if 'detected_voter_name' not in st.session_state:
    st.session_state.detected_voter_name = None

# Start the WebRTC stream
ctx = webrtc_streamer(
    key="voter-system",
    video_processor_factory=FaceVoterTransformer,
    rtc_configuration={  # This is needed for WebRTC to work over the internet
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}, # Only video, no audio input
    async_processing=True, # Process frames asynchronously
)

# Display detected voter name and trigger speech outside the transformer
if st.session_state.detected_voter_name:
    st.success(f"Detected: {st.session_state.detected_voter_name}")
    # You might want to control when speech happens more carefully
    # e.g., only on first detection or when a "Vote" button is pressed
    # speak_web(f"Welcome, {st.session_state.detected_voter_name}")

st.write("---")
st.subheader("Voting Actions (Conceptual)")

if st.session_state.detected_voter_name:
    if st.button("Cast Vote for Candidate A"):
        # Log vote, update database, etc.
        st.info(f"{st.session_state.detected_voter_name} voted for Candidate A!")
        speak_web(f"{st.session_state.detected_voter_name} has voted for Candidate A. Thank you.")
        st.session_state.detected_voter_name = None # Clear after voting

    if st.button("Cast Vote for Candidate B"):
        # Log vote, update database, etc.
        st.info(f"{st.session_state.detected_voter_name} voted for Candidate B!")
        speak_web(f"{st.session_state.detected_voter_name} has voted for Candidate B. Thank you.")
        st.session_state.detected_voter_name = None # Clear after voting
else:
    st.warning("Please stand in front of the camera for face detection to proceed with voting.")

st.write("---")
st.info("This is a demonstration. Real-world voting systems require robust security, voter verification, and data integrity measures.")
