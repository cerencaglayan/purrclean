from flask import Flask, Response
import cv2
import serial
import time

app = Flask(__name__)

def arduino():
    arduino = serial.Serial('COM3', 9600, timeout=.1)
    time.sleep(2)

cat_detected = False
last_time_cat_seen = time.time()

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera açılamıyor")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Kedi tespiti
        cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cats = cat_cascade.detectMultiScale(gray, 1.3, 5, minSize=(75, 75))

        global cat_detected
        global last_time_cat_seen

        for (x, y, w, h) in cats:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cat_detected = True
            last_time_cat_seen = time.time()

        if cat_detected and (time.time() - last_time_cat_seen > 10):
           # arduino.write(b'1') şuanda arduino ile bağlantı yok 
            cat_detected = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "Webcam aktif! '/video_feed'"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, threaded=True)
