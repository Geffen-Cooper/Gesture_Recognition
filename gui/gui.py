import glob
import json
import logging
import re
import sys
from pathlib import Path
from typing import List

import ahk
import cv2
import numpy as np
import pandas as pd
import torch
from PyQt5 import QtCore, Qt
from PyQt5 import QtGui, uic
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QGraphicsScene
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
# Load the UI file
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torchvision import transforms

from model.HandPoseModel import HandPoseModel
from model.datasets import DataframeToNumpy, NormalizeAndFilter, ToTensor
from model.models import CentroidModel, SklearnModel, FewShotModel

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import seaborn as sns


class GestureControlMain(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Gesture Control")

        # Popup window properties
        self.new_gesture_window: NewGestureWindow = None
        self.new_gesture_idx: int = None

        uic.loadUi("home.ui", self)


        # Setup gestures
        self.gestures: List[dict] = []  # [{name, action, dataset}]]
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
        self.fps = 20
        self.timer.start(1000 // self.fps)

        # Setup conf mat
        self.conf_mat_scene = QGraphicsScene(self)
        self.conf_mat.setScene(self.conf_mat_scene)

        # Setup models
        self.lm_type = None
        self.use_clahe = None
        self.video_mode = None
        self.resolution_method = None
        self.median_filter = None

        self.few_shot_model: FewShotModel = None
        self.setup_fewshot_model("assets/AttentionRegObjective.pth", "assets/AttentionRegObjective.json")
        self.transforms = transforms.Compose([DataframeToNumpy(), NormalizeAndFilter(median_filter=self.median_filter), ToTensor()])

        self.handpose_model = HandPoseModel(filter_handedness='Right', draw_pose=True, video_mode=self.video_mode, lm_type=self.lm_type)

        # Connect button handlers
        self.add_new_gesture.clicked.connect(self.open_new_gesture_window)
        self.edit_gesture.clicked.connect(self.open_new_gesture_window_for_edit)

    def open_new_gesture_window_for_edit(self):
        selected_row = self.gestures_table.currentRow()
        if selected_row >= 0:
            self.open_new_gesture_window(None, gesture_idx=selected_row)
        else:
            self.status_box.setText('No row selected to edit.')

    def open_new_gesture_window(self, _, gesture_idx=None):
        if self.new_gesture_window is None or not self.new_gesture_window.isVisible():
            if gesture_idx is not None:
                name = self.gestures[gesture_idx]['name']
                action = self.gestures[gesture_idx]['action']
                dataset = self.gestures[gesture_idx]['dataset']
                self.new_gesture_idx = gesture_idx
                self.new_gesture_window = NewGestureWindow(fps=self.fps, idx=self.new_gesture_idx, name=name, action=action, dataset=dataset)
            else:
                self.new_gesture_idx = len(self.gestures)
                self.new_gesture_window = NewGestureWindow(fps=self.fps, idx=self.new_gesture_idx)

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
                self.status_box.setText(str(gesture) +"\t"+ str(proba.tolist()))
                print(f"Detected Gesture {gesture}, proba: {proba.tolist()}")

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
        pred_gesture, pred_proba = None, None
        if self.few_shot_model.num_classes() > 0:
            pred_gesture, pred_proba = self.few_shot_model(torch.unsqueeze(self.transforms(self.gesture_hist), dim=0))
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
        self.add_edit_gesture(sample_info['name'], sample_info['idx'], sample_info['action_path'], sample_info['dataset_path'])
        num_files = len(glob.glob(f"{Path(sample_info['dataset_path'])}/*.csv"))
        self.new_gesture_window.alertUpdateCount(num_files)

    def handle_dialog_finished(self, result):
        # Delete and recreate centroids for new / modified gesture
        if 0 <= self.new_gesture_idx < len(self.gestures):
            self.recreate_centroid(self.new_gesture_idx)
            train_acc, cm = self.few_shot_model.do_train()
            self.update_status(f"Finished training model. Training acc: {train_acc}")
            self.update_conf_mat(cm)
        self.new_gesture_idx = None

    def update_status(self, txt):
        self.status_box.setText(txt)

    def update_conf_mat(self, cm):
        # convert the confusion matrix to a pandas DataFrame and plot it using seaborn
        plt.figure(figsize=(5, 3.5))

        gesture_names = [x['name'] for x in self.gestures]
        cm_df = pd.DataFrame(cm, index=gesture_names, columns=gesture_names)
        ax = sns.heatmap(cm_df, annot=True, cmap='Blues', fmt='g')
        fig = ax.get_figure()
        fig.canvas.draw()

        self.conf_mat_scene.clear()
        self.conf_mat_scene.addWidget(FigureCanvas(fig))

        self.conf_mat.setSceneRect(0, 0, 500, 350)
        # self.conf_mat.fitInView(self.conf_mat_scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def recreate_centroid(self, idx):
        print(f"Recreate centroid: {idx}")
        gesture = self.get_gesture_by_idx(idx)
        assert gesture is not None

        # Delete centroid if exists
        deleted_idx = self.few_shot_model.delete_centroid_if_exists(idx)

        # Create centroid anew
        dfs = [pd.read_csv(csv_path) for csv_path in glob.glob(f"{Path(gesture['dataset'])}/*.csv")]
        for df in dfs:
            data = torch.from_numpy(df.values).unsqueeze(dim=0)
            self.few_shot_model.add_data_for_class(data, class_num=idx)

    def add_edit_gesture(self, name, idx, action, dataset):
        gesture = self.get_gesture_by_idx(idx)
        if gesture is None:
            self.gestures.append({'name': name, 'action': action, 'dataset': dataset})
            row_position = self.gestures_table.rowCount()
            assert row_position == len(self.gestures) - 1
            self.gestures_table.insertRow(row_position)
            self.gestures_table.setItem(row_position, 0, QTableWidgetItem(name))
            self.gestures_table.setItem(row_position, 1, QTableWidgetItem(action))
            self.gestures_table.setItem(row_position, 2, QTableWidgetItem(dataset))

        else:
            gesture['name'] = name
            gesture['action'] = action
            gesture['dataset'] = dataset

            self.gestures_table.setItem(idx, 0, QTableWidgetItem(name))
            self.gestures_table.setItem(idx, 1, QTableWidgetItem(action))
            self.gestures_table.setItem(idx, 2, QTableWidgetItem(dataset))

        try:
            with open(self.gesture_load_path, 'w') as f:
                json.dump(self.gestures, f)
                print(f"Saved gestures to {self.gesture_load_path}")
        except Exception as e:
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            logger.error(e, exc_info=True)

    def get_gesture_by_idx(self, idx):
        assert idx == int(idx)
        if 0 <= idx < len(self.gestures):
            return self.gestures[idx]
        else:
            return None

    def learn_gesture(self, dfs, gesture_num):
        data = torch.stack([self.transforms(df) for df in dfs])
        self.few_shot_model.learn_centroid(data, gesture_num)

    def setup_fewshot_model(self, checkpoint_path, params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)

        self.lm_type = params['lm_type']
        self.use_clahe = params['use_clahe']
        self.video_mode = params['video_mode']
        self.resolution_method = params['resolution_method']
        self.median_filter = params['median_filter']

        # self.few_shot_model = CentroidModel(**params['model_params'], checkpoint_file=checkpoint_path)
        # self.few_shot_model = SklearnModel(**params['model_params'], checkpoint_file=checkpoint_path, model=SVC(probability=True))
        self.few_shot_model = SklearnModel(**params['model_params'], checkpoint_file=checkpoint_path, model=DecisionTreeClassifier(min_samples_leaf=2))

        for idx, gesture in enumerate(self.gestures):
            self.recreate_centroid(idx)

        if len(self.gestures) > 0:
            train_acc, cm = self.few_shot_model.do_train()
            self.update_status(f"Finished training model. Training acc: {train_acc}")
            self.update_conf_mat(cm)

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
        for gesture in self.gestures:
            row_position = self.gestures_table.rowCount()
            self.gestures_table.insertRow(row_position)
            self.gestures_table.setItem(row_position, 0, QTableWidgetItem(gesture['name']))
            self.gestures_table.setItem(row_position, 1, QTableWidgetItem(gesture['action']))
            self.gestures_table.setItem(row_position, 2, QTableWidgetItem(gesture['dataset']))


class NewGestureWindow(QDialog):
    # Signals must be class variables not instance variables
    save_sample_signal = pyqtSignal(str)
    confirm_info_signal = pyqtSignal(dict)

    def __init__(self, fps, idx, name=None, action=None, dataset=None):
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
        self.idx_box.setText(str(idx))

        if name is not None:
            self.name_box = name
        if action is not None:
            self.action_path_box = action
        if dataset is not None:
            self.dataset_path_box = dataset

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
        idx = self.idx_box.text()
        action_path = self.action_path_box.text()
        dataset_path = self.dataset_path_box.text()

        name_is_valid = len(name) > 0
        idx_is_valid = True
        try:
            _ = int(idx)
        except:
            idx_is_valid = False

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
        if not idx_is_valid:
            errors_lst.append("Index is invalid!")
        if not action_path_is_valid:
            errors_lst.append("Action path is invalid!")
        if not ds_path_is_valid:
            errors_lst.append("Dataset path is invalid!")

        if name_is_valid and idx_is_valid and action_path_is_valid and ds_path_is_valid:
            self.save_sample_btn.setEnabled(True)
            self.name_box.setEnabled(False)
            self.action_path_box.setEnabled(False)
            self.dataset_path_box.setEnabled(False)
            self.select_macro.setEnabled(False)
            self.select_dataset_folder.setEnabled(False)
            self.confirm_info_signal.emit(
                {'name': name, 'idx': int(idx), 'action_path': action_path, 'dataset_path': dataset_path})
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
