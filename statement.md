###### VIT BHOPAL COMPUTER VISION PROJECT STATEMENT 

###### Student Name : Anika Saxena 

###### Registration number: 24BAI10420

###### Course Code : CSE3010



##### Problem Statement

Drowsy and distracted driving is a leading contributor to road accidents worldwide, and current mitigation largely depends on the driver's own self-awareness — which is exactly what fatigue degrades first. Advanced Driver Monitoring Systems exist in premium vehicles, but they rely on specialized infrared hardware, making them unavailable to the vast majority of vehicles on the road, including two-wheelers, taxis, and commercial fleet vehicles, where driver fatigue is most common due to long working hours.



This project addresses that gap by building a **low-cost, hardware-independent system** that runs on any ordinary RGB webcam or phone camera, and that can reliably detect the visual signs of drowsiness (prolonged eye closure, yawning) and distraction (sustained head-turn away from the road) in real time — alerting the driver early enough to prevent an incident, without needing cloud connectivity, specialized sensors, or expensive hardware.





##### Scope of the Project



The system covers:



* Real-time monitoring of a single driver's face via a standard webcam
* Computation of eye, mouth, and head-pose metrics frame by frame
* Fusion of these metrics into a single drowsiness/distraction score
* Real-time on-screen and audio alerting
* Session-level logging with an exported fatigue-trend report

**Out of scope** (see Future Enhancements in the project report):

* Multi-occupant monitoring
* Integration with vehicle telematics or CAN bus data
* Infrared / night-vision imaging for low-light or nighttime driving
* Cloud-based analytics or fleet dashboards





#### Target Users

* **Individual drivers** who want a personal, low-cost fatigue-monitoring aid for daily commutes or long drives.
* **Fleet and commercial transport operators** who want an affordable way to flag risky driving sessions after the fact, using the exported session report, without installing specialized hardware across their vehicle fleet.





#### High-Level Features

1. **Face \& Feature Detection** — captures webcam frames, preprocesses them for varying lighting conditions, detects the driver's face, and extracts eye and mouth regions of interest.
2. **Drowsiness / Distraction Analysis** — computes the Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head-tilt angle (via optical flow tracking), then fuses these into a single weighted drowsiness score.
3. **Alert \& Logging** — classifies the driver's state (Normal / Drowsy / Distracted) based on the fused score sustained over consecutive frames, triggers a real-time visual and audio alert when needed, and logs every event to a session file that is summarized into an end-of-session fatigue-trend report.



This design  fuses **two independent signals** (eye closure and head-pose deviation) rather than relying on eye closure alone, reducing false positives from normal blinking while also catching distraction events that eye-only systems miss.

