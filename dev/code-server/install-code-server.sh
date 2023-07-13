#!/usr/bin/env bash

# IMPORTANT: must create A record for SUBDOMAIN.DOMAIN.TLD

# Set your email and domain
export EMAIL="service@subsystem3.ai"
export DOMAIN="dev.subsystem3.ai"
export VS_CODE_PASSWORD="Hesitancy#Polygon0#Unless"
export PASSWORD="Hesitancy#Polygon0#Unless"

# set noninteractive mode to skip prompts
export DEBIAN_FRONTEND=noninteractive

# get public IP using DigitalOcean metadata service
export PUBLIC_IP=$(curl --silent http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address)

# update and upgrade
sudo apt update --yes
sudo apt upgrade --yes

# install dependencies
sudo apt install --yes curl snapd

# install certbot
sudo snap install --classic certbot

# Request and download the certificate
sudo certbot certonly --standalone --non-interactive --agree-tos --email $EMAIL -d $DOMAIN

# install code-server
curl -fsSL https://code-server.dev/install.sh | sh

# Create a new service file for code-server with the necessary parameters
sudo bash -c "cat > /etc/systemd/system/code-server.service" << EOF
[Unit]
Description=code-server
After=nginx.service

[Service]
Type=simple
User=user
Environment=PASSWORD=${VS_CODE_PASSWORD}
ExecStart=/usr/bin/authbind --deep /usr/bin/code-server --bind-addr $PUBLIC_IP:443 --cert /etc/letsencrypt/live/$DOMAIN/fullchain.pem --cert-key /etc/letsencrypt/live/$DOMAIN/privkey.pem --user-data-dir /home/user/.local/share/code-server
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# reload the systemd daemon to apply the changes
sudo systemctl daemon-reload

# enable and start the service
sudo systemctl stop code-server
sudo systemctl enable --now code-server
sudo systemctl restart code-server
sudo systemctl status code-server

sudo code-server --bind-addr $PUBLIC_IP:443 --cert /etc/letsencrypt/live/$DOMAIN/fullchain.pem --cert-key /etc/letsencrypt/live/$DOMAIN/privkey.pem &

# attempt certificate renewal when it will expire within 30 days
(crontab -l ; echo "0 */12 * * * sudo certbot renew --quiet") | crontab -
