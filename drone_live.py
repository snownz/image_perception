import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import cv2
import time
import random
import numpy as np
import base64
import threading
import queue
import json
from typing import List, Dict, Tuple, Optional
from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import copy
from dataclasses import dataclass, field
from src.models import ObjectDetectionModel

# Initialize Flask app and SocketIO
app = Flask(__name__)
app.config['SECRET_KEY'] = 'livestream-secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Shared queues for communication between threads
frame_queue = queue.Queue(maxsize=10)  # Limit queue size to prevent memory issues
detection_queue = queue.Queue(maxsize=10)

# Shared storage for latest data
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
detection_lock = threading.Lock()
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class BackboneConfig:
    in_channels: float = 3
    embed_dim: float = 384
    num_heads: float = 8
    depth: float = 4
    num_tokens: float = 4096
    model: str = "linear"

@dataclass
class ComputePrecision:
    grad_scaler: bool = True

@dataclass
class Detection:
    nc: int = 24
    ch: tuple = (384, 384, 384)
    hd: int = 256  # hidden dim
    nq: int = 300  # num queries
    ndp: int = 4  # num decoder points
    nh: int = 8  # num head
    ndl: int = 6  # num decoder layers
    d_ffn: int = 1024  # dim of feedforward
    dropout: float = 0.0
    act: nn.Module = nn.ReLU()
    eval_idx: int = -1
    # Training args
    learnt_init_query: bool = False

@dataclass
class Config:
    log_dir: str = "./models/"
    name: str = "detection_v5_small"
    backbone_name: str = "encoder_v5"
    compute_precision: ComputePrecision = field(default_factory=ComputePrecision)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    detection: Detection = field(default_factory=Detection)

def resize_image(image_tensor, size):
    return F.interpolate(image_tensor.unsqueeze(0), size=(size, size),
                         mode='bilinear', align_corners=False).squeeze(0)

def load_model():
    global model
    cfg = Config()
    model = ObjectDetectionModel(cfg, device)
    model.to(device)
    model.eval()
load_model()

class RTMPReader:

    def __init__(self, rtmp_url: str, reconnect_delay: float = 2.0, max_retries: int = -1):
        """
        Initialize an RTMP reader with reconnection capabilities.
        
        Args:
            rtmp_url: The RTMP stream URL
            reconnect_delay: Seconds to wait before reconnection attempts
            max_retries: Maximum number of reconnection retries (-1 for infinite)
        """
        self.rtmp_url = rtmp_url
        self.reconnect_delay = reconnect_delay
        self.max_retries = max_retries
        self.cap = None
        self.running = False
        self.connect()
    
    def connect(self) -> bool:
        """Establish connection to the RTMP stream."""
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.rtmp_url)
        return self.cap.isOpened()
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the RTMP stream with automatic reconnection.
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.reconnect():
                return False, None
        
        # Attempt to read frame
        ret, frame = self.cap.read()
        if not ret:
            if not self.reconnect():
                return False, None
            ret, frame = self.cap.read()
        
        return ret, frame
    
    def reconnect(self) -> bool:
        """
        Attempt to reconnect to the RTMP stream.
        
        Returns:
            True if reconnection was successful, False otherwise
        """
        retries = 0
        while self.max_retries < 0 or retries < self.max_retries:
            print(f"Connection lost. Attempting to reconnect ({retries + 1})...")
            time.sleep(self.reconnect_delay)
            
            if self.connect():
                print("Successfully reconnected to RTMP stream.")
                return True
            
            retries += 1
        
        print(f"Failed to reconnect after {retries} attempts.")
        return False
    
    def release(self):
        """Release the video capture resources."""
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class MockObjectDetector:

    """Mock object detector that generates random detections."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize with optional list of class names.
        
        Args:
            class_names: List of class names to use for mock detections
        """
        self.class_names = class_names or [
            "person", "car", "truck", "bicycle", "motorcycle", 
            "dog", "cat", "bird", "drone"
        ]
        self.running = False
    
    def detect(self, frame: np.ndarray, bbox_noise_factor: float = 3, conf_noise_factor: float = 0.00) -> List[Dict]:
        
        if frame is None:
            return []
        
        height, width = frame.shape[:2]

        # helper function to generate a random detection
        def generate_random_detection() -> Dict:

            w = random.randint(width // 10, width // 3)
            h = random.randint(height // 10, height // 3)
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            # Calculate center coordinates
            xc = x + w / 2
            yc = y + h / 2

            # Random class and confidence
            class_id = random.randint(0, len(self.class_names) - 1)
            confidence = random.uniform(0.4, 0.98)

            # Clip coordinates to be within frame bounds
            xc = min(max(xc, 0), width - 10)
            yc = min(max(yc, 0), height - 10)
            w = min(max(w, 0), width - 10)
            h = min(max(h, 0), height - 10)

            return {
                "bbox": [xc, yc, w, h],  # center format: [xc, yc, w, h]
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": confidence
            }
        
        with detection_lock:

            global latest_detections
            l_D = copy.deepcopy( latest_detections )
            
        if l_D:
            # Remove one detection (choose a random index)
            remove_index = random.randint(0, len(l_D) - 1)
            l_D.pop(remove_index)
            
            # Add a new detection
            new_det = generate_random_detection()
            l_D.append(new_det)
            
            # Apply noise to each detection's bbox and confidence
            for det in l_D:
                # Apply noise to bbox (add a small random offset to each coordinate)
                noise = [random.uniform(-bbox_noise_factor, bbox_noise_factor) for _ in range(4)]
                noisy_bbox = [val + noise[i] for i, val in enumerate(det["bbox"])]
                # Reclip the noisy bbox to ensure values remain within frame bounds
                noisy_bbox[0] = min(max(noisy_bbox[0], 0), width - 10)
                noisy_bbox[1] = min(max(noisy_bbox[1], 0), height - 10)
                noisy_bbox[2] = min(max(noisy_bbox[2], 0), width - 10)
                noisy_bbox[3] = min(max(noisy_bbox[3], 0), height - 10)
                det["bbox"] = noisy_bbox
                
                # Apply noise to confidence
                det["confidence"] = min(max(det["confidence"] + random.uniform(-conf_noise_factor, conf_noise_factor), 0), 1)
            
            detections = l_D.copy()
        else:
            # No previous detections, so create N random ones (here N is random between 1 and 5)
            N = random.randint(1, 5)
            detections = [generate_random_detection() for _ in range(N)]
            l_D = detections.copy()
    
        return detections

class ObjectDetector:

    def __init__(self):

        self.int_to_label = {
            0: "person",
            1: "birds",
            2: "parking meter",
            3: "stop sign",
            4: "street sign",
            5: "fire hydrant",
            6: "traffic light",
            7: "motorcycle",
            8: "bicycle",
            9: "LMVs",
            10: "HMVs",
            11: "animals",
            12: "poles",
            13: "barricades",
            14: "traffic cones",
            15: "mailboxes",
            16: "stones",
            17: "small walls",
            18: "bins",
            19: "furniture",
            20: "pot plant",
            21: "sign boards",
            22: "boxes",
            23: "trees",
        }
    
    def predict(self, image):

        with torch.no_grad():

            torch_image = torch.from_numpy( image ).permute(2, 0, 1).float() / 255.0
            torch_image = torch_image.to( device )
            torch_image = resize_image( torch_image, 640 )

            boxes, scores, labels, detectors = model.predict( torch_image, stride_slices = 32, confidence_threshold = 0.9, iou_threshold = 0.3 )
        
        return boxes.cpu().numpy(), scores.cpu().numpy(), labels.cpu().numpy()
    
    def detect(self, frame: np.ndarray):

        if frame is None:
            return []
        
        im_height, im_width = frame.shape[:2]

        # Perform detection
        boxes, scores, labels = self.predict( frame )

        detections = []
        for box, score, label in zip(boxes, scores, labels):

            x_center, y_center, width, height = box
            x_min = int( ( x_center - width / 2 ) * im_width )
            y_min = int( ( y_center - height / 2 ) * im_height )
            x_max = int( ( x_center + width / 2 ) * im_width )
            y_max = int( ( y_center + height / 2 ) * im_height )
            
            # Append detection
            detections.append({
                "bbox": [x_min, y_min, x_max, y_max],
                "class_id": int(label),
                "class_name": self.int_to_label[int(label)],
                "confidence": float(score)
            })

        return detections

# Thread function for capturing frames from RTMP
def rtmp_reader_thread(rtmp_url: str):
    reader = RTMPReader(rtmp_url, reconnect_delay=2.0, max_retries=-1)
    reader.running = True
    
    print(f"Starting RTMP reader thread for {rtmp_url}")
    
    last_frame_time = time.time()
    try:
        while reader.running:
            # Read frame with automatic reconnection handling
            ret, frame = reader.read()
            
            if not ret or frame is None:
                print("Failed to retrieve frame after reconnection attempts.")
                time.sleep(1)
                continue
            
            # Compute FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_frame_time)
            last_frame_time = current_time
            
            # Resize for web streaming (optional, adjust as needed)
            frame = cv2.resize(frame, (854, 480))
            
            # Add FPS text
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update latest frame with lock
            with frame_lock:
                global latest_frame
                latest_frame = frame.copy()
            
            # Try to add to queue without blocking if full
            try:
                if not frame_queue.full():
                    frame_queue.put_nowait(frame.copy())
            except queue.Full:
                pass  # Skip frame if queue is full
            
            # Throttle capture rate to reduce CPU usage
            time.sleep(0.01)  # Adjust for desired capture rate
            
    except Exception as e:
        print(f"Error in RTMP reader thread: {e}")
    finally:
        print("Stopping RTMP reader thread")
        reader.release()

# Thread function for object detection
def detection_thread():
    # detector = MockObjectDetector()
    detector = ObjectDetector()
    detector.running = True
    
    print("Starting object detection thread")
    
    try:
        while detector.running:
            # Get frame from queue, waiting up to 1 second
            try:
                frame = frame_queue.get(timeout=1.0)
                
                # Process frame with object detection
                detections = detector.detect(frame)
                
                # Update latest detections with lock
                with detection_lock:
                    global latest_detections
                    latest_detections = detections
                
                # Notify queue task is done
                frame_queue.task_done()
                
                # Try to add to detection queue without blocking
                try:
                    if not detection_queue.full():
                        detection_queue.put_nowait(detections)
                except queue.Full:
                    pass  # Skip if queue is full
                
                # Emit detections through Socket.IO
                socketio.emit('detections', json.dumps(detections))
                
            except queue.Empty:
                pass  # No frame available, continue waiting
                
            # Throttle detection rate slightly
            time.sleep(0.01)
                
    except Exception as e:
        print(f"Error in detection thread: {e}")
    finally:
        print("Stopping detection thread")
        detector.running = False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('drone.html')

@app.route('/video_feed')
def video_feed():
    """Endpoint for video frame as JPEG"""
    with frame_lock:
        if latest_frame is None:
            # Return a blank image if no frame is available
            blank_image = np.zeros((480, 854, 3), np.uint8)
            _, buffer = cv2.imencode('.jpg', blank_image)
            response = Response(
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
            return response
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', latest_frame)
        
        # Return as multipart response for continuous streaming
        return Response(
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n',
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

@app.route('/frame')
def get_frame():
    """API endpoint to get the latest frame as base64 encoded JPEG"""
    with frame_lock:
        if latest_frame is None:
            return jsonify({"error": "No frame available"}), 404
        
        # Encode frame as JPEG then base64
        _, buffer = cv2.imencode('.jpg', latest_frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        return jsonify({"image": encoded_image})

@app.route('/detections')
def get_detections():
    """API endpoint to get the latest detections"""
    with detection_lock:
        return jsonify(latest_detections)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def start_threads(rtmp_url):
    """Start the worker threads"""
    # Start RTMP reader thread
    rtmp_thread = threading.Thread(target=rtmp_reader_thread, args=(rtmp_url,))
    rtmp_thread.daemon = True
    rtmp_thread.start()
    
    # Start detection thread
    detect_thread = threading.Thread(target=detection_thread)
    detect_thread.daemon = True
    detect_thread.start()
    
    return rtmp_thread, detect_thread

if __name__ == '__main__':

    # RTMP URL - replace with your actual RTMP stream URL
    rtmp_url = "rtmp://192.168.3.2/live/drone"
    
    # Start background threads
    rtmp_thread, detect_thread = start_threads(rtmp_url)
    
    # Start the Flask-SocketIO server
    print("Starting web server...")  
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)