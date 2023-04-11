import cv2
import numpy as np
from PyQt5 import uic, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QApplication, QFileDialog
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton
from PyQt5 import QtWidgets, QtGui, uic
import ahk
import sys


# Load the UI file
from model.HandPoseModel import HandPoseModel

# UI_FILE = "main.ui"
# Ui_Widget, base_class = uic.loadUiType(UI_FILE)
#
# class MyWidget(base_class, Ui_Widget):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)
#
#         # Connect add_new_gesture button to add_new_gesture_row slot
#         self.add_new_gesture.clicked.connect(self.add_new_gesture_row)
#
#         # Connect save_gesture button to save_gestures_table slot
#         self.save_gesture.clicked.connect(self.save_gestures_table)
#
#         # Set up the gestures table
#         self.gestures_table.setColumnCount(3)
#         self.gestures_table.setHorizontalHeaderLabels(['Name', 'ID', 'Action'])
#
#         # Set up the video label
#         self.video.setScaledContents(True)
#
#         # Create the video capture object
#         self.capture = cv2.VideoCapture(0)
#         self.timer = QtCore.QTimer(self)
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(1)
#
#         # cv2 histogram equalization
#         self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#
#         self.gestures = {}
#         self.handpose_model = HandPoseModel(filter_handedness='Left', draw_pose=True)
#
#     def equalize_hist(self, frame):
#         # converting to LAB color space
#         lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
#         lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
#         frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#         return frame
#
#     def update_frame(self):
#         ret, frame = self.capture.read()
#         if ret:
#             # Convert the frame to RGB format
#             frame = self.equalize_hist(frame)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = cv2.flip(frame, 1)
#
#
#             # Detect gesture and trigger action
#             gesture = self.detect_gesture(frame)
#             if gesture:
#                 self.trigger_action(gesture)
#
#             # Resize the frame to fit the video label
#             height, width, channel = frame.shape
#             ratio = height / width
#             self.video.setFixedHeight(int(self.video.width() * ratio))
#             frame_resized = frame
#             # frame_resized = cv2.resize(frame, (self.video.width(), int(self.video.width() * ratio)))
#
#             # Convert the frame to QImage format
#             # qimg = QtGui.QImage(frame_resized.video, frame_resized.shape[1], frame_resized.shape[0], QtGui.QImage.Format_RGB888)
#             qimg = QtGui.QImage(frame_resized, frame_resized.shape[1], frame_resized.shape[0], QtGui.QImage.Format_RGB888)
#             # Set the video label's pixmap to the QImage
#             self.video.setPixmap(QtGui.QPixmap.fromImage(qimg))
#
#
#     def detect_gesture(self, frame):
#         landmark_vector = self.handpose_model(frame)
#
#         # Dummy model for detecting gesture
#         return np.random.choice(["Gesture A", "Gesture B", "Gesture C"])
#
#     def trigger_action(self, gesture):
#         # Update the label with the detected gesture
#         self.prediction_label.setText(gesture)
#
#         # Trigger the appropriate action if a gesture is detected
#         if gesture in self.gestures:
#             action = self.gestures[gesture]["action"]
#             # ahk.run_script(action)
#             print(f"Running Script: {action}")
#
#     def add_new_gesture_row(self):
#         # # Add a new row to the gestures table
#         rowPosition = self.gestures_table.rowCount()
#         self.gestures_table.insertRow(rowPosition)
#
#     def save_gestures_table(self):
#         # Update the gestures dictionary with the table contents
#         for row in range(self.gestures_table.rowCount()):
#             gesture_id = self.gestures_table.item(row, 1).text()
#             name = self.gestures_table.item(row, 0).text()
#             action = self.gestures_table.item(row, 2).text()
#             self.gestures[gesture_id] = {"name": name, "action": action}

class GestureControlMain(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Gesture Control")

        uic.loadUi("home.ui", self)

        # Create a button and connect it to a slot
        button = QPushButton("Open New Window", self)
        button.setGeometry(100, 50, 100, 30)
        button.clicked.connect(self.open_new_window)

    def open_new_window(self):
        self.new_window = NewGestureWindow()
        self.new_window.show()

class NewGestureWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Create Gesture")

        # Load the UI file created in PyQt5 Designer
        uic.loadUi("create_gesture.ui", self)

        # Connect add_new_gesture button to add_new_gesture_row slot
        self.select_macro.clicked.connect(self.handle_select_macro)
        self.select_dataset_folder.clicked.connect(self.handle_select_dataset_folder)

        # ahk.run_script(open('script.ahk', 'r').read(os.path.getsize('script.ahk')))
    def handle_select_macro(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select AHK script", "", "AHK Scripts (*.ahk)", options=options)
        if fileName:
            print(f"Selected: {fileName}")
            self.action_path_box.setText(fileName)

    def handle_select_dataset_folder(self):
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select dataset folder for gesture", options=options)
        if folder_path:
            print(f"Selected: {folder_path}")
            self.dataset_path_box.setText(folder_path)

if __name__ == "__main__":
    # app = QtWidgets.QApplication(sys.argv)
    # window = MyWidget()
    # window.show()
    # sys.exit(app.exec_())

    app = QApplication(sys.argv)
    window = GestureControlMain()
    window.show()
    sys.exit(app.exec_())


