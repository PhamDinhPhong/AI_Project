from django.urls import path
from .views import emotion_detection_stream, get_latest_prediction, index

urlpatterns = [
    path('', index, name='index'),
    path('emotion-detection/', emotion_detection_stream, name='emotion_detection'),
    path('latest-prediction/', get_latest_prediction, name='latest_prediction'),
]
