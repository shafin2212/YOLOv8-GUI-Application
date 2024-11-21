import sys
import cv2
import os
import json
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget,
    QLabel, QHBoxLayout, QFileDialog, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import QTimer, Qt, QDateTime
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Function to perform inference
def run_yolo_inference(frame, model):
    results = model.predict(frame)
    return results


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BRACU ALTER")
        self.setGeometry(100, 100, 1200, 800)

        # Set window colors using stylesheets
        self.setStyleSheet("background-color: #1C1C1C; color: white;")

        # Heading label
        heading_label = QLabel("BRACU ALTER", self)
        heading_label.setStyleSheet("color: #3498DB; font-size: 24pt; font-weight: bold;")
        heading_label.setAlignment(Qt.AlignCenter)
        heading_label.setGeometry(10, 10, self.width(), 40)

        # Main layout
        main_layout = QHBoxLayout()

        # Option section layout (left side)
        option_layout = QVBoxLayout()
        button_style = "background-color: black; color: #33E9FF; font-size: 10pt; font-family: Arial; border: 2px white solid;border-radius: 25px;"
        
        # Button to select YOLOv8 model
        self.select_model_button = QPushButton("Select Model")
        self.select_model_button.setStyleSheet(button_style)
        self.select_model_button.clicked.connect(self.select_model)
        option_layout.addWidget(self.select_model_button)

        # Button to start YOLOv8 detection
        self.detect_button = QPushButton("Start Detection")
        self.detect_button.setStyleSheet(button_style)
        self.detect_button.clicked.connect(self.start_detection)
        option_layout.addWidget(self.detect_button)

        # Button to stop YOLOv8 detection
        self.stop_button = QPushButton("Close Detection")
        self.stop_button.setStyleSheet(button_style)
        self.stop_button.clicked.connect(self.stop_detection)
        option_layout.addWidget(self.stop_button)

        # Camera feed layout (top middle)
        camera_layout = QVBoxLayout()
        self.camera_feed_label = QLabel(self)
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.camera_feed_label.setFixedSize(800, 600)  # Adjust the size as needed
        camera_layout.addWidget(self.camera_feed_label)

        # Timer label
        self.timer_label = QLabel("00:00:00", self)
        self.timer_label.setStyleSheet("color: #4275f5; font-size: 15pt; font-weight: bold;")
        self.timer_label.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(self.timer_label)

        main_layout.addLayout(option_layout)
        main_layout.addLayout(camera_layout)

        # Detected objects section layout (right side)
        detected_objects_layout = QVBoxLayout()
        self.detected_objects_table = QTableWidget()
        self.detected_objects_table.setRowCount(0)
        self.detected_objects_table.setColumnCount(3)
        self.detected_objects_table.setHorizontalHeaderLabels(['Class', 'Confidence', 'Time'])
        detected_objects_layout.addWidget(self.detected_objects_table)

        main_layout.addLayout(detected_objects_layout)

        # Processing details layout
        processing_details_layout = QVBoxLayout()
        processing_details_label = QLabel("Processing Details:")
        processing_details_label.setStyleSheet("background-color: #3498DB; color: white; font-size: 10pt;")
        processing_details_layout.addWidget(processing_details_label)

        self.processing_details_output = QTextEdit()
        self.processing_details_output.setStyleSheet("background-color: #1C1C1C; color: white; font-size: 10pt;")
        processing_details_layout.addWidget(self.processing_details_output)
        main_layout.addLayout(processing_details_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Timer for updating camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_camera_feed)
        self.cap = None

        # YOLO model
        self.model = None

        # Detected objects set
        self.detected_objects_set = set()
        self.analysis_data = {}

    def select_model(self):
        pt_file, _ = QFileDialog.getOpenFileName(self, "Select .pt file", "", "PyTorch Model Files (*.pt)")
        if pt_file:
            self.load_model(pt_file)

    def load_model(self, pt_file):
        try:
            from ultralytics import YOLO  # Import YOLO from ultralytics
            self.model = YOLO(pt_file)  # Load YOLOv8 model
            self.processing_details_output.append(f"Model loaded successfully from {pt_file}.")
        except Exception as e:
            self.processing_details_output.append(f"Error loading model: {str(e)}")

    def start_detection(self):
        if self.model is None:
            self.processing_details_output.append("Error: No model loaded. Please select a model first.")
            return

        self.processing_details_output.append("Starting YOLOv8 detection...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.processing_details_output.append("Error: Could not open camera.")
            return

        self.timer.start(30)
        self.analysis_data.clear()

    def stop_detection(self):
        self.processing_details_output.append("Closing detection...")
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.save_detected_objects_to_json()
        self.save_analysis_as_pdf()

    def update_camera_feed(self):
        if self.cap is None or not self.cap.isOpened():
            self.processing_details_output.append("Error: Camera is not initialized or could not be opened.")
            return

        ret, frame = self.cap.read()
        if ret:
            results = run_yolo_inference(frame, self.model)
            annotated_frame = results[0].plot()
            frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            qt_image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_feed_label.setPixmap(pixmap)
            self.update_detected_objects_table(results[0])
        else:
            self.processing_details_output.append("Error: Could not read frame from camera.")

    def update_detected_objects_table(self, result):
        for detection in result.boxes:
            detected_class = result.names[int(detection.cls)]
            confidence = float(detection.conf)
            if detected_class not in self.detected_objects_set:
                self.detected_objects_set.add(detected_class)
                row_position = self.detected_objects_table.rowCount()
                self.detected_objects_table.insertRow(row_position)
                self.detected_objects_table.setItem(row_position, 0, QTableWidgetItem(detected_class))
                self.detected_objects_table.setItem(row_position, 1, QTableWidgetItem(f"{confidence:.2f}"))
                self.detected_objects_table.setItem(row_position, 2, QTableWidgetItem(QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")))

                if detected_class in self.analysis_data:
                    self.analysis_data[detected_class] += 1
                else:
                    self.analysis_data[detected_class] = 1

    def save_detected_objects_to_json(self):
        detected_objects_list = []
        for row in range(self.detected_objects_table.rowCount()):
            detected_object = {
                "class": self.detected_objects_table.item(row, 0).text(),
                "confidence": self.detected_objects_table.item(row, 1).text(),
                "time": self.detected_objects_table.item(row, 2).text()
            }
            detected_objects_list.append(detected_object)
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Detected Objects", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, "w") as file:
                json.dump(detected_objects_list, file, indent=4)
            self.processing_details_output.append(f"Detected objects saved to {file_path}")

    def save_analysis_as_pdf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Analysis", "", "PDF Files (*.pdf)")
        if file_path:
            with PdfPages(file_path) as pdf:
                plt.figure()
                plt.bar(self.analysis_data.keys(), self.analysis_data.values())
                plt.xlabel('Class')
                plt.ylabel('Count')
                plt.title('Object Detection Analysis')
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            self.processing_details_output.append(f"Analysis saved as PDF: {file_path}")

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
