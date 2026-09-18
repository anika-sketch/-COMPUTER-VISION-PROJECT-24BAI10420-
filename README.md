###### VIT BHOPAL COMPUTER VISION PROJECT STATEMENT

###### Student Name : Anika Saxena

###### Registration number: 24BAI10420

###### Course Code : CSE3010

###### 

###### Project Title :Real-Time Driver Drowsiness \& Distraction Detection System



A CPU-only, webcam-based computer vision system that monitors a driver's eyes, mouth, and head pose in real time, fuses these signals into a single fatigue score, and raises a visual + audio alert before drowsiness or distraction leads to an accident.



##### Overview



Drowsy and distracted driving is one of the most preventable causes of road accidents, yet most vehicles — especially budget and commercial ones — have no driver-state monitoring at all. This project implements a low-cost alternative that runs on any ordinary RGB webcam: it tracks the driver's eye closure (EAR), yawning (MAR), and head orientation (optical flow), fuses them into a single weighted drowsiness score, and alerts the driver the moment sustained drowsiness or distraction is detected — entirely offline, with no special hardware or cloud dependency.



The system applies concepts directly from the course syllabus: low-level image preprocessing (Module 1), feature extraction (Module 3), and motion/pattern analysis (Module 4), combined into one working real-time pipeline.



##### Features



* **Real-time face, eye \& mouth detection** with lighting-robust preprocessing (grayscale conversion + CLAHE histogram equalization + denoising)
* **Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)** computation for blink and yawn detection — supports both exact 6/8-point landmark formulas and a Haar-cascade bounding-box fallback
* **Head-pose tracking via Lucas-Kanade optical flow (KLT)** to catch distraction (driver looking away)
* **Multi-signal fusion** — a weighted drowsiness score combining EAR + MAR + head-tilt, instead of relying on eye closure alone
* **Consecutive-frame confirmation** to avoid false alerts from normal blinking
* **Real-time on-screen indicator** (green / yellow(orange) / red) plus a non-blocking audio alert
* **Session logging** and an end-of-session fatigue-trend report (CSV + PNG plot)
* **Configurable sensitivity** — EAR/MAR thresholds and alert-frame count are adjustable.
* **Validated configuration** — invalid thresholds or mis-summed fusion weights are rejected before a session starts
* **Graceful error handling** — camera-not-found, no-face-detected, and missing-audio-backend all degrade to a warning instead of crashing



##### Technologies / Tools Used



|Category|Tool|
|-|-|
|Language|Python 3.9+|
|Computer Vision|OpenCV |
|Numerical computation|NumPy|
|Data logging|pandas, CSV|
|Plotting (session report)|Matplotlib|
|Testing|pytest-compatible assert-based unit/validation tests|
|Version control|Git / GitHub|

##### 

##### Project Structure



driver-drowsiness-detection/

├── LICENSE

├── README.md

├── requirements.txt

├── config.py

├── main.py

├── alert\_manager.py

├── capture.py

├── face\_detector.py

├── feature\_extractor.py

├── head\_pose.py

├── preprocess.py

├── scorer.py

├── session\_logger.py

├── screenshots/

│   ├── alert\_state.png

│   └── normal\_state.png

└── tests/

&#x20;   ├── test\_config.py

&#x20;   ├── test\_feature\_extractor.py

&#x20;   └── test\_scorer.py



##### Installation \& Setup

##### 

1. **Clone the repository**

2. **Create a virtual environment (recommended)**
   
3. **Install dependencies**

&#x20;  opencv-python>=4.8
   numpy>=1.24
   pandas>=2.0
   matplotlib>=3.7
   pytest>=7.4
   simpleaudio>=1.0.4   



4. **Run the application**



This opens your default webcam and starts monitoring. Press **`q`** to stop the session and generate the end-of-session fatigue report.



5. **Adjust sensitivity (optional)**
Edit `config.json` to change EAR/MAR thresholds, the fusion weights, or the number of consecutive frames required before an alert triggers:

&#x20;  

##### Instructions for Testing

Run the automated test suite with: pytest tests


(The tests are also plain, dependency-free `assert`-based functions, so they can be run without pytest via `python tests/test\\\_scorer.py`, etc.)



This covers:

* **Unit tests** (`test\\\_feature\\\_extractor.py`) — EAR/MAR formulas validated against hand-computed landmark coordinates, the bounding-box fallback ratios, and malformed-input error handling. **8/8 passing.**
* **Validation tests** (`test\\\_scorer.py`) — drowsiness-state transition logic (Normal → Drowsy/Distracted → Normal) against synthetic score sequences, including a single-frame blink that must **not** trigger an alert. **8/8 passing.**
* **Configuration tests** (`test\\\_config.py`) — invalid thresholds and mis-summed fusion weights are rejected. **3/3 passing**



To manually test the live pipeline:

1. Run `python main.py` and confirm the indicator shows **green** under normal conditions.
2. Slowly close your eyes for 1–2 seconds — the indicator should turn **red** and the alert should sound.
3. Turn your head away from the camera and hold for a couple of seconds to confirm distraction detection (**orange** indicator).
4. Press `q` to stop and check that it contains a new session log (CSV) and a fatigue-trend plot (PNG).



##### License



This project was built for academic purposes as part of the CSE3010 Computer Vision course.

