import os
import cv2
import numpy as np
import base64
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from django.http import JsonResponse
from django.shortcuts import render  # Thêm dòng này để render trang HTML
from django.conf import settings
import tensorflow as tf
import json

# Buộc TensorFlow chỉ sử dụng CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# Tải mô hình cảm xúc
MODEL_PATH = os.path.join(settings.BASE_DIR, 'myapp/model_filter.h5')
try:
    classifier = load_model(MODEL_PATH, compile=False)
    class_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    print("Model loaded successfully.")
except Exception as e:
    classifier = None
    class_labels = []
    print(f"Error loading model: {e}")

# Cấu hình nhận diện khuôn mặt
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Hàm xử lý yêu cầu từ trang chủ
def index(request):
    """
    Trang chủ cho ứng dụng (hiển thị camera stream).
    """
    return render(request, 'home.html')  # Trả về trang HTML index

# API xử lý frame từ frontend
def process_frame(request):
    if request.method == 'POST':
        try:
            # Tải dữ liệu frame từ frontend
            data = json.loads(request.body)
            frame_data = data.get('frame')

            if not frame_data:
                return JsonResponse({"error": "No frame data provided"}, status=400)

            # Decode base64 thành ảnh
            frame_bytes = base64.b64decode(frame_data.split(',')[1])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Chuyển sang ảnh grayscale để nhận diện khuôn mặt
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            emotions = []
            for (x, y, w, h) in faces:
                # Cắt và resize khuôn mặt
                roi_gray = gray[y:y + h, x:x + w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                if classifier:
                    preds = classifier.predict(roi)[0]
                    emotion = class_labels[np.argmax(preds)]
                    emotions.append(emotion)

            # Kiểm tra xem có cảm xúc nào được phát hiện không
            if emotions:
                # Chọn video YouTube dựa trên cảm xúc
                youtube_links = {
                    'happy': 'https://www.youtube.com/watch?v=QDXszKiPfQw',
                    'sad': 'https://www.youtube.com/watch?v=zEkg4GBQumc&list=RDEMX84_dI1fcouTTGRjFni1EA&index=7',
                    'angry': 'https://www.youtube.com/watch?v=PGRawdwMSBg',
                    'disgust': 'https://www.youtube.com/watch?v=MmbNfOQ4GKg',
                    'fear': 'https://www.youtube.com/watch?v=BhgVXinOqq0',
                    'surprise': 'https://www.youtube.com/watch?v=dLaJRXiDU24',
                    'neutral': 'https://www.youtube.com/watch?v=XY-eeKwIYUE',
                }

                # Chọn video YouTube theo cảm xúc đầu tiên trong danh sách
                selected_video = youtube_links.get(emotions[0], youtube_links['neutral'])

                return JsonResponse({"prediction": emotions[0], "youtube_link": selected_video})
            else:
                return JsonResponse({"error": "No emotion detected"}, status=200)
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            return JsonResponse({"error": f"Failed to process frame: {str(e)}"}, status=400)
    return JsonResponse({"error": "Invalid request method"}, status=405)
