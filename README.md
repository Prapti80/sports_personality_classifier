# Sports Personality Classifier

A simple Machine Learning web application that identifies selected sports personalities from an uploaded image using OpenCV and LBPH.

## Live Demo
🔗 **Deployed Link:**  
https://sports-personality-classifier-2.onrender.com
⏳ **Note:**  
This app is hosted on Render (Free Tier).  
Please wait **30–60 seconds** on first load as the server may take time to start.

## About the Project
This project uses a traditional computer vision approach (LBPH) for face recognition instead of deep learning.
Users can:
- Upload an image
- Detect the face
- Predict the sports personality
- View the result instantly

## Supported Personalities
The model can classify **only these 5 sports personalities**:
- Virat Kohli  
- Lionel Messi  
- Roger Federer  
- Serena Williams  
- Maria Sharapova  

<img width="1914" height="1005" alt="image" src="https://github.com/user-attachments/assets/e087d8a2-8199-40b7-a948-5a311556fa60" />


## Tech Stack

- **Python**
- **Flask**
- **OpenCV (LBPH)**
- **HTML & CSS**
- **Render (Deployment)**

## Model Used
- **Algorithm:** Local Binary Pattern Histogram (LBPH)
- Fast and effective for small face datasets
- Suitable for academic and mini projects

## Features
- Face detection
- LBPH face recognition
- Clean single-page UI
- Image preview
- Cloud deployment
