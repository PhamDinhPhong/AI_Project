# Sử dụng image Python
FROM python:3.12

ENV CUDA_VISIBLE_DEVICES="-1"

# Cài đặt các thư viện yêu cầu cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy các file requirements và cài đặt các thư viện
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy mã nguồn vào container
COPY . .

# Chạy Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
