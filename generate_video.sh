#!/usr/bin/env bash

# CHECK FOR VIDEO-RELATED CHANGES
pptx_changed=$(git diff --name-only $1 $2 | grep '\.pptx$')
assets_changed=$(git diff --name-only $1 $2 | grep '^assets/')

if [ -n "$pptx_changed" ] || [ -n "$assets_changed" ]; then
    echo "Found changes in PowerPoint or assets, generating video."

    # INSTALL DEPENDENCIES
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

    # REMOVE OLD VIDEO
    rm -f $3.mp4

    # GENERATE VIDEO
    python3 pptx2video.py $3.pptx

    # GET TOTAL VIDEO DURATION
    total_seconds=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $3.mp4)
    hours=$(bc <<< "${total_seconds}/3600")
    minutes=$(bc <<< "(${total_seconds}%3600)/60")
    seconds=$(printf "%.0f" $(bc <<< "${total_seconds}%60"))
    humanized_duration="${hours}h ${minutes}m ${seconds}s"
    echo "duration=$humanized_duration" >> "$GITHUB_OUTPUT"
else
    echo "No changes in PowerPoint or assets, skipping video generation."
fi
