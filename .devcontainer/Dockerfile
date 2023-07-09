FROM mcr.microsoft.com/devcontainers/anaconda:0-3

USER root

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt apt.txt ./

RUN apt-get update && \
    apt-get upgrade -y && \
    xargs -a apt.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt
