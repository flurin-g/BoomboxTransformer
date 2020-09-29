FROM rapidsai/rapidsai:cuda10.2-runtime-ubuntu18.04-py3.8

COPY requirements.txt /
RUN apt update && apt install -y

RUN pip3 install -r /requirements.txt

COPY . /BoomboxTransformer
WORKDIR /BoomboxTransformer

VOLUME /BoomboxTransformer/data

WORKDIR /BoomboxTransformer
CMD python3 trainer_main.py