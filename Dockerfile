FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

COPY requirements.txt /
RUN apt update && apt install -y \
    python3-pip

RUN pip3 install -r /requirements.txt

COPY . /BoomboxTransformer
WORKDIR /BoomboxTransformer

VOLUME /BoomboxTransformer/data

WORKDIR /BoomboxTransformer
CMD python trainer_main.py