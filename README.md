# Driver Drowsiness Detection Key Features
- Real-time monitoring of driver's eyes using a camera
- Fatigue and drowsiness detection algorithm
- Timely alerts to prevent the driver from falling asleep
- Non-intrusive and easy to set up
- Progressive Web App (PWA) support - Install like a native app!
- Offline functionality with caching
- Fast loading with optimized performance on System


## Project Overview
The Driver Drowsiness Detection System is a non-intrusive solution designed to monitor and detect signs of fatigue in drivers. By analyzing eye states through a camera feed, the system can identify early symptoms of drowsiness and issue timely warnings, helping to prevent accidents caused by driver fatigue. This project aims to enhance road safety, especially for those driving long distances who may not recognize their own drowsiness in time.


## Key Features
- Real-time monitoring of driver’s eyes using a camera
- Fatigue and drowsiness detection algorithm
- Timely alerts to prevent the driver from falling asleep
- Non-intrusive and easy to set up

## Technologies Used
- Python
- OpenCV
- Dlib or Mediapipe (for facial landmark detection)
- NumPy, Pandas

  
 ## Optional: create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


## Setup Instructions
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the model:**

    The dlib facial landmark model is required for blink detection. Download it from the link below, unzip it, and place the `shape_predictor_68_face_landmarks.dat` file inside the `/models` directory.
    * **Download Link**: [shape_predictor_68_face_landmarks.dat](https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

4. **Run the application:**
   ```bash
   python main.py
   ```
   (Replace `main.py` with the actual entry point if different.)

## Usage
- Ensure your webcam is connected.
- Run the application as shown above.
- The system will start monitoring your eyes and alert you if signs of drowsiness are detected.

## Folder Structure
```
driver-drowsiness-detection-system/
│
├── main.py                  # Entry point for the system
├── blinkDetect.py            # Drowsiness detection script
├── lanedetection.py          # Lane detection module
├── ui.py                     # User interface handling
├── requirements.txt          # Python dependencies
├── CONTRIBUTING.md           # Contribution guidelines
├── DEPLOYMENT.md             # Deployment instructions
├── PWA_README.md              # PWA installation & usage
├── PWA_IMPLEMENTATION.md      # Technical details of PWA
│
├── models/                   # Pre-trained ML models
│   └── haarcascade_frontalface_default.xml
│
├── streamlit_app/            # Web app (Streamlit + PWA)
│   ├── streamlit_app.py
│   ├── streamlit_app_pwa.py
│   ├── manifest.json
│   ├── sw.js
│   └── icons/                # App icons for PWA
│
├── assets/                   # Images, audio, and test files
│   ├── alarm.wav
│   ├── alarm.mp3
│   ├── eye.PNG
│   ├── eye_aspect_ratio.PNG
│   ├── face.PNG
│   └── test_image.jpg

```

## Contribution Guidelines
We welcome contributions to improve this project! To get started:
1. **Star** ⭐ this repository to show your support.
2. **Fork** 🍴 the repository to your own GitHub account.
3. Create a new branch for your feature or bugfix.
4. Make your changes and commit them with clear messages.
5. Push your branch to your forked repo.
6. Open a Pull Request describing your changes.

---


## 🌐 Streamlit Web App (New Feature!)

This app version uses Streamlit + Mediapipe to detect drowsiness via webcam directly in the browser.

### Usage:
```bash
# Standard web app
streamlit run streamlit_app/streamlit_app.py

# Or run the enhanced PWA version
streamlit run streamlit_app/streamlit_app_pwa.py
```

## 📱 Progressive Web App (PWA) Support

**NEW!** This project now supports Progressive Web App functionality! Install it like a native app on your device.

### PWA Features:
- 📱 **Installable**: Add to home screen on mobile/desktop
- 🚀 **Fast Loading**: Cached resources for better performance  
- 🔄 **Offline Support**: Basic functionality without internet
- 📲 **App-like Experience**: Full-screen, native app behavior

### Quick PWA Installation:
1. **Mobile**: Open in browser → Menu → "Add to Home Screen"
2. **Desktop**: Look for install icon in address bar → Click "Install"

📖 **[Complete PWA Guide](PWA_README.md)** - Detailed installation and usage instructions

Dependencies:
Install using:
```bash
pip install -r requirements.txt
```

Features:
- Real-time Eye Aspect Ratio (EAR) monitoring
- Drowsiness warning overlay
- Web-based UI with live video
- **PWA installable web app**
- **Offline caching support**

If you have any questions or need guidance, feel free to open an issue or ask in the discussions!

---

## 👥 Contributors

Thanks to these wonderful people for contributing 💖

<!-- contrib.rocks: auto-updating avatars grid -->
[![Contributors](https://contrib.rocks/image?repo=Gagandeep-2003/driver-drowsiness-detection-system)](https://github.com/Gagandeep-2003/driver-drowsiness-detection-system/graphs/contributors)

<!-- Vaunt.dev: auto-updating contributors graph (SVG) -->
<p align="center">
  <a href="https://github.com/Gagandeep-2003/driver-drowsiness-detection-system/graphs/contributors">
    <img 
      src="https://api.vaunt.dev/v1/github/entities/Gagandeep-2003/repositories/driver-drowsiness-detection-system/contributors?format=svg&limit=54" 
      width="900" 
      height="400" 
      alt="Contributors Graph by Vaunt.dev" 
    />
  </a>
</p>



Thank you for your interest in improving the Driver Drowsiness Detection System! Your contributions are greatly appreciated.

<p align="center">
  <a href="#top" style="font-size: 18px; padding: 8px 16px; display: inline-block; border: 1px solid #ccc; border-radius: 6px; text-decoration: none;">
    ⬆️ Back to Top
  </a>
</p>
