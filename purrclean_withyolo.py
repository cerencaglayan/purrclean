from flask import Flask, Response
import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO('yolov8n.pt')

app = Flask(__name__)

@app.route('/')
def index():
    return Response("Webcam aktif! '/video_feed' adresinden izleyebilirsiniz.")

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamıyor")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Modeli kullanarak nesne tespiti
            results = model(frame)
            detections = results.boxes  # Bounding box bilgilerini al

            # Algılanan her nesne için çerçeve çiz ve skoru yaz
            for det in detections:
                x1, y1, x2, y2, conf, cls = det.xyxy[0], det.xyxy[1], det.xyxy[2], det.xyxy[3], det.conf, int(det.cls)
                label = f"{results.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)
