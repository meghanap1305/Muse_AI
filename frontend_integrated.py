import streamlit as st
import cv2
import os
import json
import numpy as np
import base64  # Required for the audio fix

# --- PAGE SETUP ---
st.set_page_config(page_title="MUSE AI", layout="centered")

st.markdown("<h1 style='text-align: center;'>MUSE AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Strict Geometry Scanner (Geometry + Audio)</p>", unsafe_allow_html=True)

# --- CONTROLS ---
col1, col2 = st.columns(2)

with col1:
    camera_type = st.radio("Choose Camera:", ["Laptop Webcam", "Phone (IP Webcam)"])
    # Added Audio Toggle
    enable_audio = st.checkbox("Autoplay Audio", value=True)

source = 0 
with col2:
    if camera_type == "Phone (IP Webcam)":
        ip_url = st.text_input("Enter IP URL:", "http://192.168.1.5:8080/video")
        source = ip_url
    else:
        st.write("Using default laptop camera.")
        source = 0

start = st.button("Start Scanner", type="primary")

# --- SETTINGS ---
MIN_MATCH_COUNT = 15
MIN_INLIERS = 14

# --- HELPER FUNCTION: CRASH-PROOF AUDIO ---
def get_audio_html(file_path):
    """
    Reads an audio file and converts it to a base64 HTML tag.
    This bypasses Streamlit's widget ID system to prevent DuplicateElementId errors in loops.
    """
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        # Hidden audio player that autoplays
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        return md
    except Exception as e:
        return ""

# --- FUNCTIONS ---
def load_dataset_features():
    if not os.path.exists("images.json"):
        st.error("images.json not found!")
        st.stop()
        
    with open("images.json", "r") as f:
        art_info = json.load(f)
        
    orb = cv2.ORB_create(nfeatures=2500) 
    dataset_features = []
    
    print("\n--- LOADING DATASET ---")
    
    for item in art_info:
        label = item["label"]
        img_path = item["file"]
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path, 0)
            if img is not None:
                kp, des = orb.detectAndCompute(img, None)
                if des is not None:
                    dataset_features.append({
                        "kp": kp,
                        "des": des,
                        "info": item,
                        "shape": img.shape
                    })
            else:
                print(f"⚠️ ERROR READING IMAGE: {img_path}")
        else:
            print(f"❌ FILE MISSING: {img_path}")
            
    print("-----------------------\n")
    return dataset_features

# --- MAIN LOGIC ---
if start:
    dataset = load_dataset_features()
    
    if len(dataset) == 0:
        st.error("No images loaded! Check your images.json and file paths.")
        st.stop()
        
    st.success(f"Active Brain: {len(dataset)} objects loaded.")

    # Matcher Setup
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    orb = cv2.ORB_create(nfeatures=2500)

    # 1. Start Camera
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce lag for IP cams

    # 2. Check if connection was successful (CRITICAL FIX)
    if not cap.isOpened():
        st.error(f"Could not connect to camera: {source}")
        st.info("Tip: If using 'IP Webcam', make sure the URL ends in '/video' and your phone and laptop are on the same WiFi.")
        st.stop()
    
    info_placeholder = st.empty()
    audio_placeholder = st.empty() # Placeholder for the invisible audio player
    last_played_audio = None
    
    st.warning("Press 'q' in the POP-UP window to stop scanning.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Video stream lost or ended.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

        best_match_name = "Unknown"
        max_inliers = 0
        best_object_data = None
        
        # Check against database
        if des_frame is not None and len(dataset) > 0:
            
            for data in dataset:
                des_db = data["des"]
                kp_db = data["kp"]
                
                if des_db is not None:
                    matches = bf.match(des_db, des_frame)
                    
                    if len(matches) > MIN_MATCH_COUNT:
                        src_pts = np.float32([kp_db[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                        # RANSAC Check (Geometric verification)
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        
                        if mask is not None:
                            inliers_count = np.sum(mask)
                            
                            if inliers_count > max_inliers:
                                max_inliers = int(inliers_count)
                                best_match_name = data["info"]["label"]
                                best_object_data = data["info"]

        # --- DECISION LOGIC ---
        display_frame = frame.copy()
        
        if max_inliers >= MIN_INLIERS:
            # --- MATCH FOUND CASE ---
            status_text = f"FOUND: {best_match_name}"
            sub_text = f"Confidence: {max_inliers} verified points"
            color = (0, 255, 0) # Green
            
            if best_object_data:
                info_placeholder.markdown(f"""
                ### ✅ Object Detected: {best_object_data['label']}
                **Confidence:** {max_inliers} verified points (RANSAC)  
                **History:** {best_object_data['backstory']}
                """)
                
                # --- AUDIO PLAYBACK LOGIC ---
                # Only play if enabled AND it's a new object (don't loop continuously)
                if enable_audio and best_object_data['label'] != last_played_audio:
                    audio_path = best_object_data.get('audio')
                    if audio_path and os.path.exists(audio_path):
                        # Use the HTML helper function to play audio safely
                        audio_html = get_audio_html(audio_path)
                        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)
                        last_played_audio = best_object_data['label']
                
        else:
            # --- NO MATCH CASE ---
            status_text = "NO MATCH FOUND"
            sub_text = f"Best match: {max_inliers} (Need {MIN_INLIERS})"
            color = (0, 0, 255) # Red
            
            # CLEAR the info box
            info_placeholder.info("Scanning for objects...")
            
            # NOTE: We do NOT reset last_played_audio here.
            # This prevents the audio from restarting immediately if the object 
            # flickers out of view for a millisecond.

        # Draw text on video
        cv2.putText(display_frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(display_frame, sub_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow("MUSE AI - Smart Scanner", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()