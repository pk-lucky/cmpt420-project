import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Track:
    def __init__(self, detection, track_id, max_age=5):
        self.id = track_id
        self.detection = detection
        self.category = detection['category']
        self.label_name = detection.get('label_name', '')
        self.score = detection['score']
        
        self.age = 0  # How many frames the track has existed
        self.hits = 1  # How many times it was matched with a detection
        self.time_since_update = 0  # Frames since last match
        self.max_age = max_age  # Maximum frames to keep without matching
        
        self.kf = self._init_kalman_filter(detection)
        
        self.history = [self._get_box_center(detection['box'])]
        
    def _init_kalman_filter(self, detection):
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        box = detection['box']
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        kf.x = np.array([x, y, w, h, 0, 0, 0, 0])
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ])
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        kf.R = np.eye(4) * 5
        kf.Q = np.eye(8) * 0.1
        kf.Q[4:, 4:] *= 10
        kf.P = np.eye(8) * 100
        
        return kf
    
    def _get_box_center(self, box):
        return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    
    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._get_prediction_box()
    
    def update(self, detection):
        box = detection['box']
        x = (box[0] + box[2]) / 2
        y = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        self.kf.update(np.array([x, y, w, h]))
        
        self.detection = detection
        self.score = detection['score']
        self.hits += 1
        self.time_since_update = 0
        
        self.history.append(self._get_box_center(box))
    
    def _get_prediction_box(self):
        x, y, w, h = self.kf.x[:4]
        return np.array([x - w/2, y - h/2, x + w/2, y + h/2])
    
    def get_state(self):
        box = self._get_prediction_box()
        return {
            'box': box,
            'score': self.score,
            'category': self.category,
            'track_id': self.id,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'label_name': self.label_name
        }
    
    def is_valid(self):
        return self.time_since_update <= self.max_age

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_age=10, min_hits=3):
        self.tracks = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.frame_count = 0
    
    def update(self, detections):
        self.frame_count += 1
        
        if len(self.tracks) == 0:
            for detection in detections:
                self.tracks.append(Track(detection, self.next_id, self.max_age))
                self.next_id += 1
            return self.get_tracks()
        
        track_boxes = []
        for track in self.tracks:
            track_boxes.append(track.predict())
        
        if len(detections) == 0:
            valid_tracks = [t for t in self.tracks if t.is_valid()]
            self.tracks = valid_tracks
            return self.get_tracks()
            
        detection_boxes = []
        for detection in detections:
            detection_boxes.append(detection['box'])
        
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for t, track_box in enumerate(track_boxes):
            for d, det_box in enumerate(detection_boxes):
                iou_matrix[t, d] = self._calculate_iou(track_box, det_box)
        
        row_indices, col_indices = linear_sum_assignment(-iou_matrix)
        
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= self.iou_threshold:
                self.tracks[row].update(detections[col])
                matched_indices.append(col)
                if col in unmatched_detections:
                    unmatched_detections.remove(col)
        
        for i in unmatched_detections:
            self.tracks.append(Track(detections[i], self.next_id, self.max_age))
            self.next_id += 1
        
        valid_tracks = [t for t in self.tracks if t.is_valid()]
        self.tracks = valid_tracks
        
        return self.get_tracks()
    
    def get_tracks(self):
        return [track.get_state() for track in self.tracks if track.hits >= self.min_hits]
    
    def _calculate_iou(self, box1, box2):
        box1 = np.array(box1)
        box2 = np.array(box2)
        
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area

    def visualize_tracks(self, image_path, output_path=None, show_history=True, show_plot=False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        cmap = cm.get_cmap('tab10')
        
        for track in self.tracks:
            if track.hits >= self.min_hits:
                track_data = track.get_state()
                box = track_data['box'].astype(int)
                track_id = track_data['track_id']
                
                color_rgba = cmap(track_id % 10)
                color = tuple(int(255 * c) for c in color_rgba[:3])
                
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                
                label = f"ID:{track_id} {track_data['category'].split('.')[-1]}"
                cv2.putText(image, label, (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                if show_history and len(track.history) > 1:
                    for i in range(1, len(track.history)):
                        pt1 = tuple(map(int, track.history[i-1]))
                        pt2 = tuple(map(int, track.history[i]))
                        cv2.line(image, pt1, pt2, color, 2)
        
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if output_path:
            cv2.imwrite(output_path, image_bgr)
            print(f"Visualization saved to {output_path}")
        
        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.title(f"Frame {self.frame_count}: {len(self.tracks)} Tracks")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return image