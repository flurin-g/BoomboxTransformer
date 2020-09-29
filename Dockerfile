FROM pytorch/pytorch:1.6.0-cuda10.2-cudnn8-runtime

COPY requirements.txt /
RUN pip install -r /requirements.txt

COPY . /BoomboxTransformer
WORKDIR /BoomboxTransformer

VOLUME /BoomboxTransformer/data

WORKDIR /BoomboxTransformer
RUN ["python", "trainer_main.py"]