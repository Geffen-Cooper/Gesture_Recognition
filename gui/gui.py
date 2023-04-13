import glob
import json
import logging
import re
import sys
from pathlib import Path

import ahk
import cv2
import numpy as np
import pandas as pd
import torch
from PyQt5 import QtCore
from PyQt5 import QtGui, uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
# Load the UI file
from torchvision import transforms

from model.HandPoseModel import HandPoseModel
from model.datasets import DataframeToNumpy, NormalizeAndFilter, ToTensor
from model.models import CentroidModel


class GestureControlMain(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Gesture Control")

        # Popup window properties
        self.new_gesture_window: NewGestureWindow = None
        self.new_gesture_id: int = None

        uic.loadUi("home.ui", self)


        # Setup gestures
        self.gestures = []  # [{id, name, action, dataset}]]
        self.gesture_ongoing = False
        self.gesture_hist: pd.DataFrame = None
        self.gesture_frame_hist = []
        self.frame_wo_hand = 0
        self.GESTURE_MIN_LEN = 30
        self.GESTURE_MAX_LEN = 80
        self.MAX_FRAMES_WO_HAND = 20

        # Load gestures
        self.gesture_load_path = "assets/gestures.json"
        if Path(self.gesture_load_path).is_file():
            with open(self.gesture_load_path, 'r') as f:
                self.gestures = json.load(f)
                self.init_gesture_table()
                print(f"Loaded gestures from {self.gesture_load_path}")

        # Set up the video
        self.video.setScaledContents(True)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Create the video capture object
        self.CAMERA_IDX = 0
        self.capture = cv2.VideoCapture(self.CAMERA_IDX)
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.update_frame)
        self.fps = 15
        self.timer.start(1000 // self.fps)

        # Setup models
        self.centroid_model: CentroidModel = None
        self.setup_model("assets/ProtomodelTrain.pth", "assets/protomodel_params.json")
        self.lm_type = "w"
        self.use_clahe = False
        self.video_mode = True
        # self.resolution_method = "f"
        self.median_filter = False
        self.transforms = transforms.Compose([DataframeToNumpy(), NormalizeAndFilter(median_filter=self.median_filter), ToTensor()])

        self.handpose_model = HandPoseModel(filter_handedness='Right', draw_pose=True, video_mode=self.video_mode, lm_type=self.lm_type)


        # Connect button handlers
        self.add_new_gesture.clicked.connect(self.open_new_gesture_window)
        self.edit_gesture.clicked.connect(self.open_new_gesture_window_for_edit)

    def open_new_gesture_window_for_edit(self):
        selected_row = self.gestures_table.currentRow()
        id_item = self.gestures_table.item(selected_row, 0)
        self.open_new_gesture_window(None, gesture_id=int(id_item.text()))

    def open_new_gesture_window(self, _, gesture_id=None):
        if self.new_gesture_window is None or not self.new_gesture_window.isVisible():
            if gesture_id is None:
                if len(self.gestures) > 0:
                    gesture_id = max(self.gestures, key=lambda x: x['id'])['id'] + 1
                else:
                    gesture_id = 0

            self.new_gesture_id = gesture_id
            self.new_gesture_window = NewGestureWindow(fps=self.fps, id=self.new_gesture_id)
            self.new_gesture_window.save_sample_signal.connect(self.handle_save_sample)
            self.new_gesture_window.confirm_info_signal.connect(self.handle_confirm_info)
            self.new_gesture_window.finished.connect(self.handle_dialog_finished)
            self.new_gesture_window.show()

    def equalize_hist(self, frame):
        # converting to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            # Convert the frame to RGB format
            if self.use_clahe:
                frame = self.equalize_hist(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)

            # Detect gesture and trigger action
            gesture, proba = self.process_landmark_vector(frame)

            if gesture is not None:
                self.prediction_label.setText(str(gesture) +"\t"+ str(proba.tolist()))

                self.trigger_action(gesture)

            # Resize the frame to fit the video label
            height, width, channel = frame.shape
            ratio = height / width
            self.video.setFixedHeight(int(self.video.width() * ratio))
            frame_resized = frame

            # Convert the frame to QImage format
            qimg = QtGui.QImage(frame_resized, frame_resized.shape[1], frame_resized.shape[0], QtGui.QImage.Format_RGB888)
            self.video.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def process_landmark_vector(self, frame):
        landmark_vector = self.handpose_model(frame)

        if landmark_vector is not None:
            self.frame_wo_hand = 0
            if not self.gesture_ongoing:
                # When gesture starts, clear variables
                self.gesture_ongoing = True
                self.frame_wo_hand = 0
                self.init_gesture_hist()
                self.gesture_frame_hist = []
            # When hand is detected, add it to the dataframe
            self.append_to_gesture_hist(landmark_vector[0].tolist())
            self.gesture_frame_hist.append(frame)

        else:
            if self.gesture_ongoing:
                self.frame_wo_hand += 1
                self.append_to_gesture_hist([0 for i in range(63)])

        # Detect if gesture has ended
        if self.gesture_ongoing and (self.frame_wo_hand >= self.MAX_FRAMES_WO_HAND or self.gesture_len() >= self.GESTURE_MAX_LEN):
            self.gesture_ongoing = False

            if not self.gesture_ongoing:
                # If gesture ended with current frame
                if self.gesture_len() < self.GESTURE_MIN_LEN:
                    print("Gesture ignored, too short")
                else:
                    if self.new_gesture_window is not None and self.new_gesture_window.isVisible() == True:
                        frames = np.stack(self.gesture_frame_hist, axis=0)
                        self.new_gesture_window.update_video(frames)
                        return (None, None)
                    else:
                        detected_gesture, proba = self.classify_gesture()
                        return detected_gesture, proba
        return (None, None)

    def classify_gesture(self):
        print(f"Classifying gesture of length: {self.gesture_len()}")
        pred_gesture, pred_proba = self.centroid_model(torch.unsqueeze(self.transforms(self.gesture_hist), dim=0))
        return (pred_gesture.item(), pred_proba) if pred_gesture is not None else (None, None)

    def handle_save_sample(self, dataset_path):
        ds_path = Path(dataset_path)

        # Find highest numbered file
        csv_files = glob.glob(f"{ds_path}/*.csv")
        if len(csv_files) == 0:
            n = 0
        else:
            pattern = re.compile(r"(\d+)\.csv")
            highest_numbered_file = max(csv_files, key=lambda x: int(pattern.search(x).group(1)))
            n = int(pattern.search(highest_numbered_file).group(1)) + 1

        save_path = ds_path / Path(f"{n}.csv")
        self.gesture_hist.to_csv(save_path, index=False)
        print(f"Saved to {save_path}")
        self.new_gesture_window.alertIsSaved(save_path.stem, len(csv_files))
        self.new_gesture_window.alertUpdateCount(len(csv_files))

    def handle_confirm_info(self, sample_info):
        self.add_edit_gesture(sample_info['name'], sample_info['id'], sample_info['action_path'], sample_info['dataset_path'])
        num_files = len(glob.glob(f"{Path(sample_info['dataset_path'])}/*.csv"))
        self.new_gesture_window.alertUpdateCount(num_files)

    def handle_dialog_finished(self, result):
        # Delete and recreate centroids for new / modified gesture
        self.recreate_centroid(self.new_gesture_id)
        self.new_gesture_id = None

    def recreate_centroid(self, id):
        gesture, _ = self.get_gesture_by_id(id)
        assert gesture is not None

        # Delete centroid if exists
        _ = self.centroid_model.delete_centroid_if_exists(id)

        # Create centroid anew
        dfs = [pd.read_csv(csv_path) for csv_path in glob.glob(f"{Path(gesture['dataset'])}/*.csv")]
        for df in dfs:
            data = torch.from_numpy(df.values).unsqueeze(dim=0)
            self.centroid_model.learn_centroid(data, class_num=id)

    def add_edit_gesture(self, name, id, action, dataset):
        gesture, gesture_idx = self.get_gesture_by_id(id)
        if gesture is None:
            self.gestures.append({'name': name, 'id': id, 'action': action, 'dataset': dataset})
            row_position = self.gestures_table.rowCount()
            self.gestures_table.insertRow(row_position)
            self.gestures_table.setItem(row_position, 0, QTableWidgetItem(str(id)))
            self.gestures_table.setItem(row_position, 1, QTableWidgetItem(name))
            self.gestures_table.setItem(row_position, 2, QTableWidgetItem(action))
            self.gestures_table.setItem(row_position, 3, QTableWidgetItem(dataset))

        else:
            gesture['name'] = name
            gesture['action'] = action
            gesture['dataset'] = dataset

            self.gestures_table.setItem(gesture_idx, 0, QTableWidgetItem(str(id)))
            self.gestures_table.setItem(gesture_idx, 1, QTableWidgetItem(name))
            self.gestures_table.setItem(gesture_idx, 2, QTableWidgetItem(action))
            self.gestures_table.setItem(gesture_idx, 3, QTableWidgetItem(dataset))

        try:
            with open(self.gesture_load_path, 'w') as f:
                json.dump(self.gestures, f)
                print(f"Saved gestures to {self.gesture_load_path}")
        except Exception as e:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            logger.error(e, exc_info=True)

    def get_gesture_by_id(self, id):
        assert id == int(id)
        matches = [(i, x) for i, x in enumerate(self.gestures) if x['id'] == id]
        return (matches[0][1] if len(matches) > 0 else None), matches[0][0] if len(matches) > 0 else None

    def learn_gesture(self, dfs, gesture_num):
        data = torch.stack([self.transforms(df) for df in dfs])
        self.centroid_model.learn_centroid(data, gesture_num)

    def setup_model(self, checkpoint_path, params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        self.centroid_model = CentroidModel(**params['model_params'], checkpoint_file=checkpoint_path)

        for gesture in self.gestures:
            self.recreate_centroid(gesture['id'])

    def trigger_action(self, gesture):
        # Trigger the appropriate action if a gesture is detected
        if 0 <= gesture < len(self.gestures):
            action = self.gestures[gesture]["action"]
            # ahk.run_script(action)
            print(f"Running Script: {action}")

    def gesture_len(self):
        return len(self.gesture_hist.index)

    def append_to_gesture_hist(self, row):
        self.gesture_hist.loc[len(self.gesture_hist.index)] = row

    def init_gesture_hist(self):
        lms_x = ["lmx" + str(i) for i in range(21)]
        lms_y = ["lmy" + str(i) for i in range(21)]
        lms_z = ["lmz" + str(i) for i in range(21)]
        col_names = lms_x + lms_y + lms_z
        self.gesture_hist = pd.DataFrame(columns=col_names)

    def init_gesture_table(self):
        for gesture in sorted(self.gestures, key=lambda x: x['id']):
            row_position = self.gestures_table.rowCount()
            self.gestures_table.insertRow(row_position)
            self.gestures_table.setItem(row_position, 0, QTableWidgetItem(str(gesture['id'])))
            self.gestures_table.setItem(row_position, 1, QTableWidgetItem(gesture['name']))
            self.gestures_table.setItem(row_position, 2, QTableWidgetItem(gesture['action']))
            self.gestures_table.setItem(row_position, 3, QTableWidgetItem(gesture['dataset']))


class NewGestureWindow(QDialog):
    # Signals must be class variables not instance variables
    save_sample_signal = pyqtSignal(str)
    confirm_info_signal = pyqtSignal(dict)

    def __init__(self, fps, id):
        super().__init__()
        self.setWindowTitle("Create Gesture")

        # Load the UI file created in PyQt5 Designer
        uic.loadUi("create_gesture.ui", self)

        # Video stuff
        self.video.setScaledContents(True)
        self.video_data = None
        self.cur_frame_idx = 0
        self.fps = fps
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // self.fps)

        # Connect add_new_gesture button to add_new_gesture_row slot
        self.select_macro.clicked.connect(self.handle_select_macro)
        self.select_dataset_folder.clicked.connect(self.handle_select_dataset_folder)
        self.save_sample_btn.setEnabled(False)
        self.save_sample_btn.clicked.connect(self.handle_save_sample)
        self.id_box.setText(str(id))

        self.confirm_btn.clicked.connect(self.handle_confirm)
        self.edit_btn.clicked.connect(self.handle_edit)

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

    def handle_confirm(self):
        name = self.name_box.text()
        id = self.id_box.text()
        action_path = self.action_path_box.text()
        dataset_path = self.dataset_path_box.text()

        name_is_valid = len(name) > 0
        id_is_valid = True
        try:
            _ = int(id)
        except:
            id_is_valid = False

        action_path_is_valid = len(action_path) > 0
        try:
            Path(action_path).is_file()
        except:
            action_path_is_valid = False

        ds_path_is_valid = len(dataset_path) > 0
        try:
            Path(dataset_path).mkdir(parents=True, exist_ok=True)
        except:
            ds_path_is_valid = False

        errors_lst = []
        if not name_is_valid:
            errors_lst.append("Name must not be blank!")
        if not id_is_valid:
            errors_lst.append("ID is invalid!")
        if not action_path_is_valid:
            errors_lst.append("Action path is invalid!")
        if not ds_path_is_valid:
            errors_lst.append("Dataset path is invalid!")

        if name_is_valid and id_is_valid and action_path_is_valid and ds_path_is_valid:
            self.save_sample_btn.setEnabled(True)
            self.name_box.setEnabled(False)
            self.action_path_box.setEnabled(False)
            self.dataset_path_box.setEnabled(False)
            self.select_macro.setEnabled(False)
            self.select_dataset_folder.setEnabled(False)
            self.confirm_info_signal.emit(
                {'name': name, 'id': int(id), 'action_path': action_path, 'dataset_path': dataset_path})
        else:
            self.status_browser.setText("\n".join(errors_lst))

    def handle_edit(self):
        self.save_sample_btn.setEnabled(False)
        self.name_box.setEnabled(True)
        self.action_path_box.setEnabled(True)
        self.dataset_path_box.setEnabled(True)
        self.select_macro.setEnabled(True)
        self.select_dataset_folder.setEnabled(True)

    def handle_save_sample(self):
        dataset_path = self.dataset_path_box.text()

        self.save_sample_signal.emit(dataset_path)

    def update_video(self, video_data):
        if self.video_data is not None:
            self.video_data.flags.writeable = True
        self.video_data = video_data
        self.video_data.flags.writeable = False

    def update_frame(self):
        if self.video_data is not None:
            self.cur_frame_idx = (self.cur_frame_idx + 1) % len(self.video_data)
            frame = self.video_data[self.cur_frame_idx, :, :, :]
            height, width, channel = frame.shape
            ratio = height / width
            self.video.setFixedHeight(int(self.video.width() * ratio))

            qimg = QtGui.QImage(frame, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.video.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def alertIsSaved(self, file_name, count):
        self.status_browser.setText(f"Saved sample to {file_name}, total: {count} samples")

    def alertUpdateCount(self, count):
        self.sample_count_disp.display(count)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureControlMain()
    window.show()
    sys.exit(app.exec_())
