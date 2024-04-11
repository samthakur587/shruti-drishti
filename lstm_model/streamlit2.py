import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import mediapipe as mp

class_mapping = ['1. loud', '11. rich', '3. happy', '16. cheap', '17. flat','18. curved', '19. male', '2. quiet', '3. happy', '4. sad','48. Hello', '49. How are you', '55. Thank you', '7. Deaf', '8. Blind','9.Nice']

# Load the model architecture from JSON file
with open("model_seq.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Create the model from the JSON
loaded_model = tf.keras.models.model_from_json(loaded_model_json)

# Load the weights into the model
loaded_model.load_weights("model_seq.weights.h5")
model = loaded_model


# Function to extract pose landmarks from a frame
def extract_pose_landmarks(frame, pose):
    # Process the frame
    results = pose.process(frame)

    # Extract landmarks if available
    if results.pose_landmarks:
        landmarks = np.array([[landmark.x, landmark.y] for landmark in results.pose_landmarks.landmark])
        return landmarks[:25]
    else:
        return None

def predict(landmarks):
    # Make predictions using your model
    landmark = np.expand_dims(landmarks,axis=0)
    landmarks_re = landmark.reshape(landmark.shape[0], 30, 50)
    prediction = model.predict(landmarks_re)
    prob = np.argmax(prediction)
    name = class_mapping[prob]
    return name

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # URL of the IP camera stream
    url = "http://100.120.165.250:8080/video"
    
    # Open the IP camera stream
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        st.error("Unable to open webcam.")
        return

    sequence = []
    stframe = st.empty()
    prediction = ""
    count =0
    while True:
        count+=1
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to RGB (Streamlit uses RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_flipped = cv2.flip(frame_rgb, 1)
        landmarks = extract_pose_landmarks(frame_flipped,pose)
        if landmarks is not None:
            sequence.append(landmarks)

            # Display the frame
            if len(sequence) == 30:
                inputs = np.array(sequence)
                prediction = predict(inputs)
                # Display the prediction
                sequence = []
                # Write prediction onto the frame
        cv2.putText(frame_flipped, "Prediction: " + prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # Display the prediction
        stframe.image(frame_flipped, channels="RGB", use_column_width=True)
        # Release the webcam
        if st.button('end'):
            break
    cap.release()

# Define the thank function
def thank():
    st.write("Thank you for using the Sign-to-Text Demo! 🤟📝")

if __name__ == "__main__":
    st.markdown(
        "<h1 style='text-align: center; color: #0066ff;'>Sign-to-Text Demo 🤟📝</h1>", 
        unsafe_allow_html=True
    )
    if st.button('start'):
        main()
    