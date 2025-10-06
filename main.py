import cv2
import numpy as np
import time
from ultralytics import YOLO
import sys

yolomodel = 'yolov8n.pt' 

class AreaDetector:
    def __init__(self, model_path=yolomodel, area_coords=None):
        self.model = YOLO(model_path)
        self.area_coords = area_coords
        self.last_detection_time = 0
        self.detection_cooldown = 1
        
    def set_detection_area(self, coords):
        """Set the detection area coordinates"""
        self.area_coords = np.array(coords, dtype=np.int32)
    
    def point_in_area(self, point):
        """Check if a point is inside the detection area"""
        if self.area_coords is None:
            return False
        try:
            return cv2.pointPolygonTest(self.area_coords, point, False) >= 0
        except:
            return False
    
    def get_object_center(self, box):
        """Get the center point of a bounding box"""
        try:
            x1, y1, x2, y2 = box
            return (int((x1 + x2) / 2), int((y1 + y2) / 2))
        except:
            return (0, 0)
    
    def draw_detection_area(self, frame):
        """Draw the detection area on the frame"""
        if self.area_coords is not None and len(self.area_coords) >= 3:
            try:
                cv2.polylines(frame, [self.area_coords], True, (0, 255, 0), 2)
                cv2.putText(frame, 'Detection Area', 
                           (self.area_coords[0][0], self.area_coords[0][1] - 10),
                           cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error drawing area: {e}")
    
    def process_frame(self, frame):
        """Process a single frame for object detection"""
        try:
            results = self.model(frame, verbose=False)
            object_detected_in_area = False
        
            for result in results:
                boxes = None
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                
                if boxes is not None:
                    try:
                        if hasattr(boxes, 'xyxy'):
                            # New format with boxes object
                            box_data = boxes.xyxy
                            conf_data = boxes.conf
                            cls_data = boxes.cls
 
                        
                        # Convert to numpy if needed
                        if hasattr(box_data, 'cpu'):
                            box_data = box_data.cpu().numpy()
                        if hasattr(conf_data, 'cpu'):
                            conf_data = conf_data.cpu().numpy()
                        if hasattr(cls_data, 'cpu'):
                            cls_data = cls_data.cpu().numpy()
                        
                        # Process each detection
                        for i in range(len(box_data)):
                            try:
                                # Get coordinates and metadata
                                if len(box_data.shape) > 1:
                                    x1, y1, x2, y2 = box_data[i]
                                    confidence = float(conf_data[i]) if len(conf_data.shape) > 0 else float(conf_data)
                                    class_id = int(cls_data[i]) if len(cls_data.shape) > 0 else int(cls_data)
                                else:
                                    # Single detection case
                                    x1, y1, x2, y2 = box_data[:4]
                                    confidence = float(conf_data)
                                    class_id = int(cls_data)
                                
                                # Filter by confidence
                                if confidence > 0.5:
                                    # Get object center
                                    center = self.get_object_center([x1, y1, x2, y2])
                                    
                                    # Check if in detection area
                                    if self.point_in_area(center):
                                        object_detected_in_area = True
                                        # Red box for objects in area
                                        cv2.rectangle(frame, (int(x1), int(y1)), 
                                                    (int(x2), int(y2)), (0, 0, 255), 2)
                                        cv2.circle(frame, center, 5, (0, 0, 255), -1)
                                    else:
                                        # Blue box for objects outside area
                                        cv2.rectangle(frame, (int(x1), int(y1)), 
                                                    (int(x2), int(y2)), (255, 0, 0), 2)
                                    
                                    # Add label - handle different name formats
                                    try:
                                        if hasattr(self.model, 'names') and class_id < len(self.model.names):
                                            class_name = self.model.names[class_id]
                                        elif hasattr(result, 'names') and class_id < len(result.names):
                                            class_name = result.names[class_id]
                                        else:
                                            class_name = f'Class_{class_id}'
                                        
                                        label = f'{class_name}: {confidence:.2f}'
                                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                                   cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
                                    except:
                                        # Fallback label
                                        label = f'Object: {confidence:.2f}'
                                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                                   cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
                                        
                            except Exception as e:
                                print(f"Error processing detection {i}: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"Error processing boxes: {e}")
                        continue
            
            # Check detection with cooldown
            current_time = time.time()
            if object_detected_in_area and (current_time - self.last_detection_time) > self.detection_cooldown:
                print('Detected')
                self.last_detection_time = current_time
                
        except Exception as e:
            print(f"Error in frame processing: {e}")
            # Add error text to frame
            cv2.putText(frame, f"Processing Error: {str(e)[:50]}", (10, 50),
                       cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def test_camera(self):
        """Test camera access"""
        print("Testing camera access...")
        
        # Try different camera indices
        for i in range(5):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"Camera {i} works, Resolution: {frame.shape}")
                        cap.release()
                        return i
                cap.release()
            except:
                continue
        
        print("No cam")
        return None
    
    def run_camera(self, camera_index=0):
        """Run real time detection using camera"""
        
        # Test camera first
        try :
            working_camera = self.test_camera()
        except Exception as e:
            print(f" error: {e}")
            working_camera = None
        camera_index = working_camera
        
        print(f"Using camera index: {camera_index}")
        
        cap = cv2.VideoCapture(camera_index)
        
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            cap.set(cv2.CAP_PROP_FPS, 30)
        except:
            pass
        
        if not cap.isOpened():
            print("camera error")
            return
        
        # Get frame for default area setup
        ret, frame = cap.read()
        if not ret or frame is None:
            print(" initial frame error")
            cap.release()
            return
            
        if self.area_coords is None:
            h, w = frame.shape[:2]
            area_width = w // 2  # 1/3 of frame width
            area_height = h // 2  # 1/2 of frame height
            start_x = w - area_width - 15  # Right side with margin
            start_y = (h - area_height) // 2  # Vertically centered
            
            self.area_coords = np.array([
                [start_x, start_y],
                [start_x + area_width, start_y],
                [start_x + area_width, start_y + area_height],
                [start_x, start_y + area_height]
            ], dtype=np.int32)
            # print(f"Default detection area set on right side: {w}x{h} frame")
        
        print("Starting real-time detection...")
        print("Controls:")
        print("- Press 'q' or ESC to quit")
        print("- Press 'r' to reset detection area (click 4 points)")
        print("- Press 's' to save current frame")
        print("- Close window with X button to exit")
        
        # Area selection variables
        drawing_area = False
        temp_points = []
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing_area, temp_points
            
            if drawing_area and event == cv2.EVENT_LBUTTONDOWN:
                temp_points.append([x, y])
                print(f"Point {len(temp_points)}: ({x}, {y})")
                
                if len(temp_points) == 4:
                    self.area_coords = np.array(temp_points, dtype=np.int32)
                    drawing_area = False
                    temp_points = []
                    print(" detection area set")
        
        # Create window and set callback
        window_name = 'YOLO Area Detection'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Warning: Could not read frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Process frame
                if not drawing_area:
                    try:
                        frame = self.process_frame(frame)
                        self.draw_detection_area(frame)
                    except Exception as e:
                        print(f"Frame processing error: {e}")
                        cv2.putText(frame, "Processing Error", (10, 30),
                                   cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), 2)
                
                # Draw temporary points when setting area
                if drawing_area:
                    for i, point in enumerate(temp_points):
                        cv2.circle(frame, tuple(point), 5, (0, 255, 255), -1)
                        cv2.putText(frame, f'{i+1}', (point[0]+10, point[1]),
                                   cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255), 1)
                    
                    cv2.putText(frame, f'Click point {len(temp_points)+1}/4', 
                               (10, 30), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 255), 2)
                
                # Add FPS counter
                if frame_count % 30 == 0:  # Update every 30 frames
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    cv2.putText(frame, f'FPS: {fps:.1f}', (10, frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 255), 1)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Handle key presses with better responsiveness
                key = cv2.waitKey(1) & 0xFF
                
                # Check multiple times for better key detection
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    print("Quit key pressed - exiting")
                    break
                elif key == ord('r') or key == ord('R'):
                    drawing_area = True
                    temp_points = []
                    print("Reset key pressed - click 4 points to define new detection area...")
                elif key == ord('s') or key == ord('S'):
                    filename = f"frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Save key pressed - frame saved as {filename}")
                
                # Additional check for window close button
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed - exiting...")
                    break
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Runtime error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup completed")

def main():
    print("YOLO Area Detection System")
    
    try:
        # Create detector
        print("Initializing detector...")
        detector = AreaDetector()
        detector.run_camera()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()