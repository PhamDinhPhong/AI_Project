from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Đường dẫn đến trang chủ
    path('process-frame/', views.process_frame, name='process-frame'),  # Đường dẫn API xử lý frame
]
