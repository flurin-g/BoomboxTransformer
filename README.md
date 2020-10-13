# Run Tensorboard
- The command to run tensorboard is:  
`tensorboard --logdir data/output/lightning_logs`

# Setup containers
1. **Create volume**: When using the containerized version, first create a volume to hold the data, provide a name for easier reference, e.g.
`boombox-data`:  
` docker volume create --name boombox-data`
1. **Build the container**: Execute in the root directory, for Dockerfile to be found:  
`docker build . -t boombox`
2. **Start Container**: To start the container with the volume mounted at  with gpu-support, run:  
`docker run --gpus all -v boombox-data:/BoomboxTransformer/data -t boombox:latest`  
Note: To run without a specified volume, use:  
`docker run --gpus all -t boombox:latest`

# Configuration and Hyperparameters
Configuration handling is done with facebook Hydra, the configuration files are hierarchically structured, like so:
```
├── config.yaml
├── dataset
│   └── noisy_speech.yaml
├── hparams
│   ├── basic.yaml
│   └── lstm.yaml
├── lightning
│   ├── local.yaml
│   └── remote.yaml
└── logging
    └── tensorboard.yaml
```
- **Local vs remote**: In the `config.yaml` you can specify which setup you want to run, if you run locally without gpu, use `local`,
to run with gpu and half-precision training, enter `remote`
- **Number of gpu's**: The number of gpu's used by the model is set under `lightning/remote.yaml`, the default is 1
- **Dataset**: Urban8k and Librispeech are downloaded and meta-data is created per default, if you have downloaded them already
you can set `download: False` and `create_meta: False` under `dataset/noisy_speech.yaml` to save time
