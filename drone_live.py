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
frame_queue = queue.Queue(maxsize=5)  # Smaller queue size to ensure fresher frames
detection_queue = queue.Queue(maxsize=5)

# Shared storage for latest data
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
detection_lock = threading.Lock()
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Thread control flags
rtmp_thread_running = False
detection_thread_running = False
thread_control_lock = threading.Lock()

# Thread references to allow force termination
current_rtmp_thread = None
current_detection_thread = None
thread_references_lock = threading.Lock()

# Set a timeout value for thread shutdown (seconds)
THREAD_SHUTDOWN_TIMEOUT = 3

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
    torch.compile( model )
    a = 10

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
        """
        Run model prediction with safeguards for thread interruption.
        Uses a separate thread with a timeout to prevent the model from hanging.
        """
        if detection_thread_exit_event.is_set():
            print("Skipping prediction due to exit event")
            return np.array([]), np.array([]), np.array([])
            
        # Define result placeholders
        result_boxes = np.array([])
        result_scores = np.array([])
        result_labels = np.array([])
        
        # Flag to indicate if prediction completed
        prediction_complete = threading.Event()
        prediction_error = [None]  # Use a list to store the error (if any) from the worker thread
        
        # Define the actual prediction function to run in a separate thread
        def run_prediction():
            try:
                if detection_thread_exit_event.is_set():
                    return
                
                with torch.no_grad():
                    # Convert image to tensor
                    torch_image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
                    torch_image = torch_image.to(device)
                    torch_image = resize_image(torch_image, 640)
                    
                    # Check for exit event again before running model
                    if detection_thread_exit_event.is_set():
                        return
                    
                    # Run model inference
                    nonlocal result_boxes, result_scores, result_labels
                    boxes, scores, labels, _ = model(
                        torch_image, 
                        stride_slices=256, 
                        confidence_threshold=0.7, 
                        iou_threshold=0.3
                    )
                    
                    # Check for exit event before copying results
                    if detection_thread_exit_event.is_set():
                        return
                    
                    # Convert results to numpy and store
                    result_boxes = boxes.cpu().numpy()
                    result_scores = scores.cpu().numpy()
                    result_labels = labels.cpu().numpy()
                    
            except Exception as e:
                prediction_error[0] = e
                print(f"Error in prediction thread: {e}")
            finally:
                # Always signal completion, even on error
                prediction_complete.set()
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        try:
            # Start prediction in a separate thread
            prediction_thread = threading.Thread(target=run_prediction, name="PredictionThread")
            prediction_thread.daemon = True
            prediction_thread.start()
            
            # Wait for prediction to complete with a timeout
            prediction_success = prediction_complete.wait(timeout=2.0)  # 2-second timeout
            
            # If the prediction didn't complete or there was an error, handle it
            if not prediction_success:
                print("Model prediction timed out, forcing stop")
                return np.array([]), np.array([]), np.array([])
                
            if prediction_error[0] is not None:
                print(f"Model prediction failed: {prediction_error[0]}")
                return np.array([]), np.array([]), np.array([])
                
            # Return the results
            return result_boxes, result_scores, result_labels
            
        except Exception as e:
            print(f"Error managing prediction: {e}")
            return np.array([]), np.array([]), np.array([])
        finally:
            # Ensure GPU memory is cleared
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def detect(self, frame: np.ndarray):

        if frame is None:
            return []
        
        im_height, im_width = frame.shape[:2]

        # Perform detection
        boxes, scores, labels = self.predict( frame )

        detections = []
        for box, score, label in zip(boxes, scores, labels):

            x_center, y_center, width, height = box
            # Convert normalized coordinates to actual pixel coordinates
            x_min = int( ( x_center - width / 2 ) * im_width )
            y_min = int( ( y_center - height / 2 ) * im_height )
            x_max = int( ( x_center + width / 2 ) * im_width )
            y_max = int( ( y_center + height / 2 ) * im_height )
            
            # Calculate for center format with actual pixel values
            pixel_x_center = (x_min + x_max) / 2
            pixel_y_center = (y_min + y_max) / 2
            pixel_width = x_max - x_min
            pixel_height = y_max - y_min
            
            # Append detection with center format (matching what the frontend expects)
            detections.append({
                "bbox": [pixel_x_center, pixel_y_center, pixel_width, pixel_height],
                "class_id": int(label),
                "class_name": self.int_to_label[int(label)],
                "confidence": float(score)
            })

        return detections

# Thread function for capturing frames from RTMP
def rtmp_reader_thread(rtmp_url: str):
    global rtmp_thread_running
    
    with thread_control_lock:
        rtmp_thread_running = True
    
    reader = RTMPReader(rtmp_url, reconnect_delay=2.0, max_retries=-1)
    reader.running = True
    
    print(f"Starting RTMP reader thread for {rtmp_url}")
    
    last_frame_time = time.time()
    try:
        while True:
            # Check if thread should stop
            with thread_control_lock:
                if not rtmp_thread_running:
                    print("RTMP thread received stop signal")
                    break
            
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
                        
            # Add FPS text
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Update latest frame with lock
            with frame_lock:
                global latest_frame
                latest_frame = frame.copy()
            
            # Clear queue if it's getting full to avoid backlog
            if frame_queue.qsize() > 3:  # If more than 3 frames are waiting
                try:
                    # Try to drain the queue
                    while not frame_queue.empty():
                        frame_queue.get_nowait()
                        frame_queue.task_done()
                except queue.Empty:
                    pass
            
            # Add the new frame to the queue
            try:
                frame_queue.put_nowait(frame.copy())
            except queue.Full:
                # If still full after clearing, skip this frame
                pass
            
            # Throttle capture rate to reduce CPU usage
            time.sleep(0.01)  # Adjust for desired capture rate
            
    except Exception as e:
        print(f"Error in RTMP reader thread: {e}")
    finally:
        print("Stopping RTMP reader thread")
        reader.release()
        
        # Ensure thread is marked as stopped
        with thread_control_lock:
            rtmp_thread_running = False

# Global variable to force detection thread termination
detection_thread_exit_event = threading.Event()

# Thread function for object detection
def detection_thread():
    global detection_thread_running
    
    # Reset exit event
    detection_thread_exit_event.clear()
    
    with thread_control_lock:
        detection_thread_running = True
    
    # detector = MockObjectDetector()
    detector = ObjectDetector()
    
    print("Starting object detection thread")
    
    try:
        while not detection_thread_exit_event.is_set():
            # Check if thread should stop
            with thread_control_lock:
                if not detection_thread_running:
                    print("Detection thread received stop signal")
                    break
            
            # Clear the queue and get only the latest frame
            processed_frame = None
            frames_processed = 0
            
            # Drain the queue to get the most recent frame
            try:
                # Use timeout to allow periodic checks for stop signals
                while not frame_queue.empty():
                    try:
                        frame = frame_queue.get_nowait()
                        frame_queue.task_done()
                        processed_frame = frame
                        frames_processed += 1
                    except queue.Empty:
                        break
                    
                    # Check for stop signal frequently
                    if not detection_thread_running or detection_thread_exit_event.is_set():
                        break
            except Exception as e:
                print(f"Error processing queue: {e}")
            
            # Check again for thread stop signal
            if not detection_thread_running or detection_thread_exit_event.is_set():
                print("Detection thread received stop signal during queue processing")
                break
                
            # If we processed frames, log how many we skipped
            if frames_processed > 1:
                print(f"Skipped {frames_processed-1} older frames")
            
            # Process only if we have a frame
            if processed_frame is not None:
                try:
                    # Process frame with object detection
                    detections = detector.detect(processed_frame)
                    
                    # Update latest detections with lock
                    with detection_lock:
                        global latest_detections
                        latest_detections = detections
                    
                    # Emit detections through Socket.IO
                    socketio.emit('detections', json.dumps(detections))
                except Exception as e:
                    print(f"Error processing detection: {e}")
            else:
                # Wait a bit if no frames are available, but check for stop signal
                for _ in range(5):  # 5 x 10ms = 50ms wait, with checks in between
                    if not detection_thread_running or detection_thread_exit_event.is_set():
                        break
                    time.sleep(0.01)
    except Exception as e:
        print(f"Error in detection thread: {e}")
    finally:
        print("Stopping detection thread")
        
        # Clear any detections
        with detection_lock:
            global latest_detections
            latest_detections = []
        
        # Ensure thread is marked as stopped
        with thread_control_lock:
            detection_thread_running = False
        
        print("Detection thread has stopped")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/drone')
def drone():
    """Serve the drone page and ensure threads are running"""
    # RTMP URL - update this as needed
    rtmp_url = "rtmp://192.168.3.2/live/drone"
    
    try:
        # Always stop existing threads and start fresh ones
        print("Initializing fresh threads for new drone page request")
        
        # First stop any existing threads
        stop_all_threads(force=True)
        
        # Start new threads
        start_threads(rtmp_url)
    except Exception as e:
        print(f"Error starting threads: {e}")
    
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

@socketio.on('leave_drone')
def handle_leave_drone(data):
    """Handle request to stop drone threads"""
    print('Client requested to stop drone threads')
    
    # Use a separate thread for the stopping operation to prevent blocking
    def stop_thread_task():
        try:
            # Stop all threads with force=True to ensure they stop
            stop_all_threads(force=True)
            print("Threads successfully stopped")
            socketio.emit('threads_stopped', {'status': 'success'})
        except Exception as e:
            print(f"Error stopping threads: {e}")
            socketio.emit('threads_stopped', {'status': 'error', 'message': str(e)})
    
    # Launch the thread stopping in the background
    stopping_thread = threading.Thread(target=stop_thread_task)
    stopping_thread.daemon = True
    stopping_thread.start()
    
    # Immediately return success to the client
    return {'status': 'success', 'message': 'Drone threads stop request received'}

def start_threads(rtmp_url):
    """Start the worker threads"""
    # Reset the thread control flags
    global rtmp_thread_running, detection_thread_running, current_rtmp_thread, current_detection_thread
    
    # First, ensure any existing threads are stopped
    stop_all_threads(force=True)
    
    # Clear the exit event to ensure detection thread can run
    detection_thread_exit_event.clear()
    
    # Reset thread control flags
    with thread_control_lock:
        rtmp_thread_running = True
        detection_thread_running = True
    
    # Start RTMP reader thread
    rtmp_thread = threading.Thread(target=rtmp_reader_thread, args=(rtmp_url,), name="RTMP-Thread")
    rtmp_thread.daemon = True
    
    # Start detection thread
    detect_thread = threading.Thread(target=detection_thread, name="Detection-Thread")
    detect_thread.daemon = True
    
    # Store references to the threads
    with thread_references_lock:
        current_rtmp_thread = rtmp_thread
        current_detection_thread = detect_thread
    
    # Start the threads
    rtmp_thread.start()
    detect_thread.start()
    
    print("Worker threads started")
    return rtmp_thread, detect_thread

def stop_all_threads(force=False):
    """Stop all running threads, with option to force termination"""
    global rtmp_thread_running, detection_thread_running, current_rtmp_thread, current_detection_thread
    
    print(f"Stopping all threads (force={force})")
    
    # Set the exit event immediately
    detection_thread_exit_event.set()
    
    # Set thread control flags to signal threads to stop
    with thread_control_lock:
        rtmp_thread_running = False
        detection_thread_running = False
    
    # Get references to current threads
    with thread_references_lock:
        rtmp_thread = current_rtmp_thread
        detect_thread = current_detection_thread
    
    if rtmp_thread and rtmp_thread.is_alive():
        print(f"Waiting for RTMP thread to stop...")
        rtmp_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
        if rtmp_thread.is_alive() and force:
            print("RTMP thread did not stop, leaving as daemon")
    
    if detect_thread and detect_thread.is_alive():
        print(f"Waiting for detection thread to stop...")
        detect_thread.join(timeout=THREAD_SHUTDOWN_TIMEOUT)
        if detect_thread.is_alive() and force:
            print("Detection thread did not stop, leaving as daemon")
    
    # Clear thread references
    with thread_references_lock:
        current_rtmp_thread = None
        current_detection_thread = None
    
    # Reset resources
    with frame_lock:
        global latest_frame
        latest_frame = None
    
    with detection_lock:
        global latest_detections
        latest_detections = []
    
    # Clear the queues
    try:
        while not frame_queue.empty():
            frame_queue.get_nowait()
            frame_queue.task_done()
    except queue.Empty:
        pass
    
    try:
        while not detection_queue.empty():
            detection_queue.get_nowait()
            detection_queue.task_done()
    except queue.Empty:
        pass
    
    # Clear GPU memory if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("All threads stopped and resources cleared")

if __name__ == '__main__':

    # RTMP URL - replace with your actual RTMP stream URL
    rtmp_url = "rtmp://192.168.3.2/live/drone"
    
    # Start background threads
    rtmp_thread, detect_thread = start_threads(rtmp_url)
    
    # Start the Flask-SocketIO server
    print("Starting web server...")  
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)