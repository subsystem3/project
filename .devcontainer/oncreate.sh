#!/usr/bin/env bash
# Executes after the creating Codespace and cloning repo

get_info() {
    echo -e "-----------------------------"
    echo -e "Date: $(date)"
    echo -e "User: $(whoami)"
    echo -e "Home: $HOME"
    echo -e "Shell: $SHELL"
    echo -e "Working Directory: $(pwd)"
    echo -e "PATH: $PATH"
    echo -e "code: $(which code)"
    echo -e "Directories: $(ls -la $HOME)"
    echo -e "-----------------------------"
}

add_custom_welcome() {
    sudo cp .devcontainer/first-run-notice.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt
}

main() {
    get_info &
    add_custom_welcome &
    wait
}

main
