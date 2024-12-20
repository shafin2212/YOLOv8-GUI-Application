Here is the `README.md` code you can use for your project:

```markdown
 YOLOv8 GUI Application

An intuitive graphical user interface (GUI) built with PyQt5 to facilitate real-time object detection using the YOLOv8 deep learning framework. This application enables users to load custom models, perform detections on various inputs (images, videos, webcam), and export results in JSON and PDF formats for further analysis.

---

 Features
- Custom YOLOv8 Model Support: Easily load `.pt` files for object detection.
- Multiple Input Sources: Perform detection on:
  - Images
  - Video files
  - Live webcam feed
- Save Detection Results: Export detections as:
  - JSON: Detailed object detection data.
  - PDF: Visual analysis with graphs and statistics.
- Screen Recording: Record and save live detections.
- Screenshot Functionality: Capture frames during object detection.

---

 Getting Started

Follow these steps to get the application running on your system:

 1. Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- `pip` (Python package manager)

 2. Clone the Repository
```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

 3. Install Dependencies
Run the following command to install all required Python libraries:
```bash
pip install -r requirements.txt
```

 4. Run the Application
Execute the following command to launch the GUI:
```bash
python your_script_name.py
```

---

 How to Use

1. Launch the GUI: Run the script (`python your_script_name.py`).
2. Load YOLOv8 Model:
   - Click Select Model and choose a `.pt` file trained using YOLOv8.
3. Select Input Source:
   - Load an image, video, or use the webcam.
4. Start Detection:
   - Press Start Detection to begin processing.
5. View and Save Results:
   - Use options to save detections in JSON or PDF formats.
6. Additional Features:
   - Use Record to save live video streams.
   - Take screenshots of active detections.

---

 Folder Structure

```
project-folder/
│
├── your_script_name.py           Main GUI application
├── README.md                     Documentation
├── requirements.txt              Required dependencies
├── models/                       YOLOv8 `.pt` model files
├── utils/                        Utility functions for detection/report generation
├── images/                       Sample input images
├── results/                      Outputs (JSON, PDF, etc.)
└── screenshots/                  Screenshots of the GUI in action
```

---

 Screenshots

 Main GUI Interface
![Main GUI](![image](https://github.com/user-attachments/assets/dfba851f-953d-4bbc-8771-d9e88368ac04)
)

 Object Detection Results
![Detection Results](![image](https://github.com/user-attachments/assets/55ee3037-cdf2-4c80-999f-6beaf725031f)
)

---

 Dependencies

The project requires the following Python libraries:
- `ultralytics`: YOLOv8 framework.
- `PyQt5`: For creating the GUI.
- `matplotlib`: For plotting graphs.
- `opencv-python-headless`: For image and video processing.
- `reportlab`: For generating PDF reports.
- `numpy`: For numerical computations.

Install these dependencies using:
```bash
pip install -r requirements.txt
```

---

 Saving and Exporting

- JSON Export:
  - Detection results (class, confidence, timestamp) are saved in a structured JSON file.
- PDF Export:
  - Summary and analysis reports are saved with object counts and graphs.
- Recording and Screenshots:
  - Save live video streams and frames for offline use.

---

 Contributing

We welcome contributions to improve the functionality or usability of this project.  
To contribute:
1. Fork this repository.
2. Create a new feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Make your changes and test thoroughly.
4. Commit and push your changes:
   ```bash
   git commit -m "Add new feature"
   git push origin feature-name
   ```
5. Open a pull request with a detailed description.

---

 
