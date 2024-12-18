FROM  python:3.8.10
WORKDIR /app
COPY . /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install --upgrade pip
RUN pip install -r app/requirements.txt
