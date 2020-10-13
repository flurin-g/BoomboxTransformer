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

