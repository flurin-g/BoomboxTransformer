FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.6

COPY requirements.txt /
RUN apt update && apt install -y \
    python3-pip

RUN pip3 install -r /requirements.txt

COPY . /BoomboxTransformer
WORKDIR /BoomboxTransformer

VOLUME /BoomboxTransformer/data

WORKDIR /BoomboxTransformer
CMD python3 trainer_main.py