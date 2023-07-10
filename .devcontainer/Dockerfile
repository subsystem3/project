FROM mcr.microsoft.com/devcontainers/anaconda:0-3

USER root

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt apt.txt ./

RUN apt-get update && \
    xargs -a apt.txt apt-get install -y && \
    apt-get clean && \
    apt-get autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt && \
    tlmgr --no-ask install algorithm babel-english

RUN rm requirements.txt apt.txt
