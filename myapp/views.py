import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings
import tensorflow as tf

# Buộc TensorFlow chỉ sử dụng CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Tải mô hình cảm xúc
MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp/model_filter.h5')
try:
    classifier = load_model(MODEL_PATH, compile=False)
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
except Exception as e:
    classifier = None
    class_labels = []
    print(f"Error loading model: {e}")

# Cấu hình nhận diện khuôn mặt
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

latest_prediction = "No prediction yet"

def gen_frames():
    """
    Phát trực tiếp video từ camera và gửi khuôn mặt đã nhận diện.
    """
    global latest_prediction

    # Mở camera
    cap = cv2.VideoCapture(2,cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Xử lý video: nhận diện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Vẽ khung quanh khuôn mặt
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Chuẩn bị khuôn mặt để dự đoán
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Dự đoán cảm xúc (kiểm tra nếu mô hình được tải thành công)
            if classifier:
                preds = classifier.predict(roi)[0]
                latest_prediction = class_labels[np.argmax(preds)]

        # Chuyển khung hình về định dạng JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

def emotion_detection_stream(request):
    """
    View để stream video nhận diện khuôn mặt.
    """
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_latest_prediction(request):
    """
    API để lấy kết quả dự đoán mới nhất.
    """
    global latest_prediction
    return JsonResponse({"prediction": latest_prediction})

def index(request):
    """
    Trang chủ.
    """
    return render(request, 'home.html')
