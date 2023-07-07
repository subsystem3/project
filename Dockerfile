FROM ubuntu:latest

ENV DEBIAN_FRONTEND=noninteractive

COPY requirements.txt apt.txt ./

RUN apt-get update && \
    apt-get install -y python3.11 python3-pip && \
    xargs -a apt.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    pip3 install -r requirements.txt
