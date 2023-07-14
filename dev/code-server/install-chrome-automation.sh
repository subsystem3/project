#!/usr/bin/env bash

# SET NONINTERACTIVE MODE TO SKIP PROMPTS
export DEBIAN_FRONTEND=noninteractive

sudo apt-get --yes update
sudo apt-get install wget unzip

# INSTALL CHROME
wget --quiet --output-document - https://dl.google.com/linux/linux_signing_key.pub | sudo apt-key add -
sudo echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" | sudo tee /etc/apt/sources.list.d/google-chrome.list
sudo apt-get --yes update
sudo apt-get --yes install google-chrome-stable

# INSTALL CHROMEDRIVER
CHROME_DRIVER_VERSION=$(curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE)
wget -N http://chromedriver.storage.googleapis.com/$CHROME_DRIVER_VERSION/chromedriver_linux64.zip -P ~/
sudo apt-get install -y unzip
unzip -o ~/chromedriver_linux64.zip -d ~/
rm ~/chromedriver_linux64.zip
sudo mv -f ~/chromedriver /usr/local/bin/chromedriver
sudo chown root:root /usr/local/bin/chromedriver
sudo chmod 0755 /usr/local/bin/chromedriver
