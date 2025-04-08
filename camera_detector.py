import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
import matplotlib.pyplot as plt

class CameraDetector:
    def __init__(self, confidence_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True).to(self.device).eval()
        self.confidence_threshold = confidence_threshold
        self.relevant_coco_classes = {
            1: 'person',           # human.pedestrian.adult
            2: 'bicycle',          # vehicle.bicycle
            3: 'car',              # vehicle.car
            4: 'motorcycle',       # vehicle.motorcycle
            6: 'bus',              # vehicle.bus.bendy
            8: 'truck',            # vehicle.truck
        }
        
        self.coco_to_nuscenes = {
            1: 'human.pedestrian.adult',    
            2: 'vehicle.bicycle',           
            3: 'vehicle.car',               
            4: 'vehicle.motorcycle',        
            6: 'vehicle.bus.bendy',         
            8: 'vehicle.truck',             
        }
        
    def detect(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
                
        with torch.no_grad():
            predictions = self.model(image_tensor)[0]
        
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        detections = [
            {
                'box': boxes[i].astype(int),
                'score': scores[i],
                'category': self.coco_to_nuscenes[label_id],
                'label_id': label_id,
                'label_name': self.relevant_coco_classes.get(label_id, 'unknown')
            }
            for i, (score, label_id) in enumerate(zip(scores, labels))
            if score > self.confidence_threshold and label_id in self.coco_to_nuscenes
        ]
        
        return detections
    
    def visualize_detections(self, image_path, detections, output_path=None, show_plot=False):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        color_map = {
            'human': (0, 255, 0),       
            'vehicle.car': (255, 0, 0), 
            'vehicle.truck': (255, 165, 0),  
            'vehicle.bus': (255, 165, 0),
            'default': (0, 0, 255)      
        }
        
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = box
            
            color = next(
                (color_map[key] for key in color_map if key in detection['category']),
                color_map['default']
            )
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{detection['label_name']}: {detection['score']:.2f}"
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Visualization saved to {output_path}")
        
        if show_plot:
            plt.figure(figsize=(12, 8))
            plt.imshow(image)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        
        return image