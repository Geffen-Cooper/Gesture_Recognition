import cv2
import numpy as np
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QApplication
from PyQt5 import QtWidgets, QtGui, uic
import ahk
import sys


# Load the UI file
UI_FILE = "main.ui"
Ui_Widget, base_class = uic.loadUiType(UI_FILE)

class MyWidget(base_class, Ui_Widget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Connect add_new_gesture button to add_new_gesture_row slot
        self.add_new_gesture.clicked.connect(self.add_new_gesture_row)

        # Connect save_gesture button to save_gestures_table slot
        self.save_gesture.clicked.connect(self.save_gestures_table)

        # Set up the gestures table
        self.gestures_table.setColumnCount(3)
        self.gestures_table.setHorizontalHeaderLabels(['Name', 'ID', 'Action'])

        # Set up the video label
        self.video.setScaledContents(True)

        # Create the video capture object
        self.capture = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.gestures = {}

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Convert the frame to RGB format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize the frame to fit the video label
            height, width, channel = frame.shape
            ratio = height / width
            self.video.setFixedHeight(int(self.video.width() * ratio))
            frame_resized = cv2.resize(frame, (self.video.width(), int(self.video.width() * ratio)))

            # Convert the frame to QImage format
            qimg = QtGui.QImage(frame_resized.data, frame_resized.shape[1], frame_resized.shape[0], QtGui.QImage.Format_RGB888)

            # Set the video label's pixmap to the QImage
            self.video.setPixmap(QtGui.QPixmap.fromImage(qimg))

            # Detect gesture and trigger action
            gesture = self.detect_gesture(frame)
            if gesture:
                self.trigger_action(gesture)

    def detect_gesture(self, frame):
        # Dummy model for detecting gesture
        return np.random.choice(["Gesture A", "Gesture B", "Gesture C"])

    def trigger_action(self, gesture):
        # Update the label with the detected gesture
        self.prediction_label.setText(gesture)

        # Trigger the appropriate action if a gesture is detected
        if gesture in self.gestures:
            action = self.gestures[gesture]["action"]
            # ahk.run_script(action)
            print(f"Running Script: {action}")

    def add_new_gesture_row(self):
        # Add a new row to the gestures table
        rowPosition = self.gestures_table.rowCount()
        self.gestures_table.insertRow(rowPosition)

    def save_gestures_table(self):
        # Update the gestures dictionary with the table contents
        for row in range(self.gestures_table.rowCount()):
            gesture_id = self.gestures_table.item(row, 1).text()
            name = self.gestures_table.item(row, 0).text()
            action = self.gestures_table.item(row, 2).text()
            self.gestures[gesture_id] = {"name": name, "action": action}

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyWidget()
    window.show()
    sys.exit(app.exec_())


