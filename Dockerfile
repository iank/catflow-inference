FROM python:3.10 as base

WORKDIR /app

# Install ffmpeg, libsm6, libxext6
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install torch cpu version specifically
ADD requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

# Install any needed packages specified in requirements.txt
ADD dist/catflow_inference-0.1.0-py3-none-any.whl /app
RUN pip install --no-cache-dir catflow_inference-0.1.0-py3-none-any.whl

EXPOSE 5054

CMD ["uvicorn", "catflow_inference.main:app", "--port", "5054", "--host", "0.0.0.0"]

