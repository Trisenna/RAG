FROM docker.1ms.run/python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install  -r requirements.txt
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY . .

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]