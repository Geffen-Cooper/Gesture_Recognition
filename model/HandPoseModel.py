import time
import warnings

import cv2
import torch
from torch import nn
import mediapipe as mp


class HandPoseModel(nn.Module):
    def __init__(self, filter_handedness=None, draw_pose=True, model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5, video_mode=True, lm_type="n", *args, **kwargs):
        """

        Args:
            filter_handedness: If two hands are detected, only output result for specified hand. Options: [None, 'Left', 'Right']
            model_complexity:
            max_num_hands:
            min_detection_confidence:
            min_tracking_confidence:
            video_mode:
        """
        # Load mediapipe handpose model
        super().__init__()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=model_complexity,
                                         max_num_hands=max_num_hands,
                                         min_detection_confidence=min_detection_confidence,
                                         min_tracking_confidence=min_tracking_confidence,
                                         static_image_mode=not video_mode)

        assert filter_handedness in (None, 'Left', 'Right'), "Invalid value for filter_handedness"
        self.filter_handedness = filter_handedness
        self.draw_pose = draw_pose

        if lm_type == "n":
            warnings.warn(f"Using lm_type n!")
        assert lm_type in ("n", "w")
        self.lm_type = lm_type

        # measure FPS
        self.last_frame_time = 0
        self.curr_frame_time = 0
        self.frame_count = 0
        self.fps = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def forward(self, image):
        """
        Takes in a single frame (RGB)
        Args:
            image:

        Returns:

        """
        results = self.hands.process(image=image)
        result_vectors = []

        hands_detected = results.multi_hand_landmarks is not None

        if hands_detected:
            for hand_landmarks, world_hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks, results.multi_handedness):

                # Annotate frame
                if self.draw_pose:
                    # calculate the fps
                    self.frame_count += 1
                    self.curr_frame_time = time.time()
                    diff = self.curr_frame_time - self.last_frame_time
                    if diff > 1:
                        self.fps = round(self.frame_count / (diff), 2)
                        self.last_frame_time = self.curr_frame_time
                        self.frame_count = 0

                    cv2.putText(image, f"FPS: {self.fps}, {handedness.classification[0].label} hand", (7, 30), self.font, 1, (100, 255, 0), 1, cv2.LINE_AA)

                # Process pose if correct hand
                if (handedness is None) or (self.filter_handedness == handedness.classification[0].label):

                    # Only draw pose if correct hand or it becomes confusing
                    if self.draw_pose:
                        image.flags.writeable = True
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())
                        image.flags.writeable = False
                    # Convert landmarks into vector
                    lm_list_x = []
                    lm_list_y = []
                    lm_list_z = []

                    landmarks_to_use = hand_landmarks if self.lm_type == "n" else world_hand_landmarks
                    for lm in landmarks_to_use.landmark:
                        lm_list_x.append(lm.x)
                        lm_list_y.append(lm.y)
                        lm_list_z.append(lm.z)

                    frame_landmark_vector = torch.Tensor([lm_list_x + lm_list_y + lm_list_z])
                    result_vectors.append(frame_landmark_vector)

                else:
                    continue
            return None if len(result_vectors) == 0 else torch.concat(result_vectors, dim=0)
        return None
