from django.urls import path
from . import views

urlpatterns = [
    path('emotion-stream/', views.emotion_detection_stream, name='emotion-stream'),
    path('process-frame/', views.process_frame, name='process-frame'),
    path('get-prediction/', views.get_latest_prediction, name='get-prediction'),
    path('', views.index, name='index'),
]
