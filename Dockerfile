FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

WORKDIR /usr/src/project

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt --yes --no-install-recommends install \
    build-essential \
    cmake \
    git \
    libsdl2-dev \
    libboost-all-dev \
    libopenal-dev \
    zlib1g-dev \
    libjpeg-dev \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    libfluidsynth-dev \
    libgme-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY ./vizdoom ./vizdoom

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY doom.wad doom2.wad .

COPY main.py .

ENTRYPOINT ["python3", "main.py"]
