FROM mcr.microsoft.com/devcontainers/anaconda:0-3

USER root

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt apt.txt ./

RUN apt-get update && \
    xargs -a apt.txt apt-get install -y && \
    pip install -r requirements.txt && \
    tlmgr init-usertree && \
    tlmgr install algorithm babel-english
