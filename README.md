# Describe with YOLOv8

A computer vision project that uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) to detect objects and describe their **spatial position** and **approximate distance** (close/medium/far) using speech synthesis. The system works with both webcam and static images.

---

##  Features

- Real-time object detection using YOLOv8
- Spatial localization using a 3x3 grid (e.g., "top-left", "center-middle", "bottom-right")
- Depth approximation based on bounding box size (close / medium / far)
- Converts detections into **spoken natural language**
- Webcam support and static image support

---

##  Files

| File              | Description |
|-------------------|-------------|
| `detect_webcam.py`| Capture from webcam, describe object positions with speech |
| `detect_image.py` | Analyze a static image and generate a natural language description |
| `dependency.py`   | Checks YOLOv8 installation and loads model |
| `images/`         | Folder for test images like `walk.jpg` |
| `requirements.txt`| List of dependencies |

---

##  Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/describe-with-yolov8.git
cd describe-with-yolov8
```

### 2. Set up environment
You can use venv or conda, your choice.

```bash
pip install -r requirements.txt
```

### 3. Required Python Packages
If you're setting up manually:

```bash
pip install ultralytics opencv-python pyttsx3
```

 Make sure to download the YOLOv8n model if not already present (yolov8n.pt is automatically pulled by ultralytics when you call YOLO("yolov8n.pt")).

---

##  How to Use

###  Webcam Mode

```bash
python detect_webcam.py
```

Detects objects from a webcam frame

Outputs spoken description like:  
"person close center-middle, chair far top-left"

### Static Image Mode
Edit the image_folder and image_name in detect_image.py, then run:

```bash
python detect_image.py
```

Detects and describes objects in a given image

---

## Example Output

**Image Description Example:**

In the image, I see a person close in the center-middle, a chair is far in the top-left.

**Speech Output:**  
The system will read the same sentence aloud using your system's text-to-speech engine.

### 🔧 Future Improvements
- Add quadrant overlays on output images
- Export descriptions as .txt or .json
- Support live video stream analysis
- Make model selectable via CLI

---

## License
This project is for educational and demo purposes.

---

## Author
Akshat G  
Feel free to connect on [LinkedIn](https://www.linkedin.com/in/akshat-gururaj)


