MUSE AI - Smart Museum Scanner

MUSE AI is an intelligent computer vision application designed to act as a digital museum guide. It uses advanced feature matching algorithms to recognize artifacts (coins, statues, notes) in real-time through a webcam and provides instant audio-visual feedback with historical backstories.

 Key Features

High-Accuracy Recognition: Uses SIFT (Scale-Invariant Feature Transform) to recognize objects even if they are rotated, zoomed, or tilted. This method is used to match the features of the image.when u scan an image ,the code tries to figure 2000 features from the image like shape ,patterns the curves etc.. for matching.

Glare Removal: Implements CLAHE (Contrast Limited Adaptive Histogram Equalization) to see details on shiny objects like coins and metal flasks.I had to implement this because when i scan an image under lamp or light ,there will be white glare lines which the code reads as a feature and increases its confidence.

Audio Narrations: audio descriptions (history/backstory) is played when an object is found.

Dual Camera Support: Works with a Laptop Webcam or a Phone Camera (via IP Webcam).

Tech Stack

Language: Python 3.x
Streamlit
Computer Vision: OpenCV (cv2)
NumPy 

Prepare Data:
Place your images in the dataset/ folder.
Ensure your images.json paths match the actual file locations.

