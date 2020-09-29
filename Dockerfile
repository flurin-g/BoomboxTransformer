FROM nvidia/cuda:10.2-cudnn8-runtime-centos8


COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . /BoomboxTransformer
WORKDIR /BoomboxTransformer

VOLUME /BoomboxTransformer/data

WORKDIR /BoomboxTransformer
RUN ["python", "trainer_main.py"]