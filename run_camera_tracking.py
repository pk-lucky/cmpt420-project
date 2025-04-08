import os
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
import cv2
import time
import matplotlib.cm as cm

from camera_detector import CameraDetector
from camera_tracker import SimpleTracker

def process_scene(nusc, scene_idx=0, output_dir='output', visualize=True, show_plots=False):
    os.makedirs(output_dir, exist_ok=True)
    scene = nusc.scene[scene_idx]
    print(f"Processing scene {scene_idx}: {scene['name']}")
    sample_token = scene['first_sample_token']
    
    detector = CameraDetector(confidence_threshold=0.5)
    tracker = SimpleTracker(iou_threshold=0.3, max_age=3, min_hits=2)
    
    frame_idx = 0
    total_detection_time = 0
    total_tracking_time = 0
    
    sample_tokens = []
    current_token = sample_token
    while current_token != '':
        sample_tokens.append(current_token)
        sample = nusc.get('sample', current_token)
        current_token = sample['next']
    
    with tqdm(total=len(sample_tokens), desc="Processing frames", unit="frame") as pbar:
        for frame_idx, sample_token in enumerate(sample_tokens):
            sample = nusc.get('sample', sample_token)
            
            cam_token = sample['data']['CAM_FRONT']
            cam_path = nusc.get_sample_data_path(cam_token)
            
            start_time = time.time()
            detections = detector.detect(cam_path)
            detection_time = time.time() - start_time
            total_detection_time += detection_time
            
            start_time = time.time()
            tracks = tracker.update(detections)
            tracking_time = time.time() - start_time
            total_tracking_time += tracking_time
            
            pbar.set_postfix({
                "Objects": len(detections),
                "Tracks": len(tracks),
                "Det Time": f"{detection_time:.3f}s",
                "Track Time": f"{tracking_time:.3f}s"
            })
            pbar.update(1)
            
            if visualize:
                image = cv2.imread(cam_path)
                det_image = image.copy()
                for detection in detections:
                    box = detection['box'].astype(int)
                    category = detection['category']
                    score = detection['score']
                    
                    if 'human' in category:
                        color = (0, 255, 0)  
                    elif 'vehicle.car' in category:
                        color = (0, 0, 255)  
                    else:
                        color = (255, 0, 0)  
                    
                    cv2.rectangle(det_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    label = f"{category.split('.')[-1]}: {score:.2f}"
                    cv2.putText(det_image, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                det_output_path = os.path.join(output_dir, f'detections_{frame_idx:03d}.jpg')
                cv2.imwrite(det_output_path, det_image)
                
                track_image = image.copy()
                cmap = cm.get_cmap('tab10')
                
                for track in tracker.tracks:
                    if track.hits >= tracker.min_hits:
                        track_data = track.get_state()
                        box = track_data['box'].astype(int)
                        track_id = track_data['track_id']
                        
                        color_rgba = cmap(track_id % 10)
                        color = (int(255 * color_rgba[2]), int(255 * color_rgba[1]), int(255 * color_rgba[0]))
                        
                        cv2.rectangle(track_image, (box[0], box[1]), (box[2], box[3]), color, 2)
                        
                        label = f"ID:{track_id} {track_data['category'].split('.')[-1]}"
                        cv2.putText(track_image, label, (box[0], box[1]-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        if len(track.history) > 1:
                            for i in range(1, len(track.history)):
                                pt1 = tuple(map(int, track.history[i-1]))
                                pt2 = tuple(map(int, track.history[i]))
                                cv2.line(track_image, pt1, pt2, color, 2)
                
                track_output_path = os.path.join(output_dir, f'tracks_{frame_idx:03d}.jpg')
                cv2.imwrite(track_output_path, track_image)
    
    print(f"\nProcessed {len(sample_tokens)} frames from scene {scene_idx}")
    print(f"Average detection time: {total_detection_time/len(sample_tokens):.3f}s")
    print(f"Average tracking time: {total_tracking_time/len(sample_tokens):.3f}s")

def create_tracking_video(output_dir, video_name='tracking.mp4', fps=10):
    track_images = [f for f in os.listdir(output_dir) if f.startswith('tracks_') and f.endswith('.jpg')]
    track_images.sort()
    
    if len(track_images) == 0:
        print("No track visualization images found.")
        return
    
    first_img = cv2.imread(os.path.join(output_dir, track_images[0]))
    height, width, _ = first_img.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, video_name)
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for img_name in tqdm(track_images, desc="Creating video"):
        img = cv2.imread(os.path.join(output_dir, img_name))
        video.write(img)
    
    video.release()
    print(f"Video saved to {video_path}")

def main():
    dataroot = 'v1.0-mini'
    nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
    scene_idx = 0
    output_dir = 'camera_tracking_results'
    process_scene(nusc, scene_idx, output_dir, visualize=True, show_plots=False)
    create_tracking_video(output_dir)
    print("\nProcessing complete. To view results:")
    print(f"1. Check individual frame images in: {output_dir}")
    print(f"2. Watch the tracking video: {os.path.join(output_dir, 'tracking.mp4')}")

if __name__ == "__main__":
    main()