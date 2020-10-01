FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.6

COPY requirements.txt /

RUN pip install -r /requirements.txt

COPY . /BoomboxTransformer
WORKDIR /BoomboxTransformer

VOLUME /BoomboxTransformer/data

WORKDIR /BoomboxTransformer
CMD python trainer_main.py