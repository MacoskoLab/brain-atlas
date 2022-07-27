FROM mambaorg/micromamba:0.24.0

COPY --chown=$MAMBA_USER:$MAMBA_USER brain-atlas /brain-atlas

RUN cd /brain-atlas \
       && micromamba install -n base -y -f environment.yml \
       && micromamba clean --all --yes

# need to install protobuf manually bc broken for some reason
# https://github.com/spesmilo/electrum/issues/7825
ARG MAMBA_DOCKERFILE_ACTIVATE=1 
RUN pip install --no-cache-dir protobuf==3.20.1; 

COPY  --chown=$MAMBA_USER:$MAMBA_USER  getcpu.sh       /getcpu.sh
# COPY --chown=$MAMBA_USER:$MAMBA_USER killallpython3 /killallpython3

USER root
# need bc for getcpu.sh
# procps so can remove all processes to force shutdown (ps and top)
RUN apt-get -y update \
    && apt-get -y install bc procps tmux sudo \
    && apt-get -y clean
    

# hail batch fails internally if not run as root, seems to work fine with conda
# USER $MAMBA_USER
USER root

RUN adduser $MAMBA_USER sudo \
     && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Useful commands
# ~/03_RCTD_Subtypes/033 Zeng Data/10_MBASS/14_hail_docker$
#     sudo docker build -t atlas .
#     sudo docker run --rm -ti atlas
#     sudo docker build --squash -t gcr.io/mouse-brain-atlas/jonahatlas .
#     sudo docker push gcr.io/mouse-brain-atlas/jonahatlas
