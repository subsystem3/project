#!/usr/bin/env bash

# print all commands for debugging
set -x

# check for changes in PowerPoint and assets
pptx_changed=$(git diff --name-only $1 $2 | grep '\.pptx$')
assets_changed=$(git diff --name-only $1 $2 | grep '^assets/')

if [ -n "$pptx_changed" ] || [ -n "$assets_changed" ]; then
    echo "Found changes in PowerPoint or assets, generating video."

    # install dependencies
    sudo apt-get update
    sudo apt-get -y install \
        ffmpeg \
        libreoffice \
        poppler-utils \
        bc

    python3 -m pip install --upgrade pip
    python3 -m pip install \
        gTTS  \
        moviepy \
        pdf2image  \
        python-pptx

    # generate video
    python3 pptx2video.py $3.pptx
else
    echo "No changes in PowerPoint or assets, skipping video generation."
fi

set +x
