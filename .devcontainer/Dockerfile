FROM mcr.microsoft.com/devcontainers/miniconda:0-3

USER root

RUN apt-get update && \
    apt-get install -y texlive-latex-extra \
    conda init bash && \
    conda config --add channels conda-forge && \
    conda install -y \
    fastai \
    gensim \
    huggingface_hub \
    ipykernel \
    ipython \
    jupyter \
    keras \
    matplotlib \
    nbconvert \
    nltk \
    notebook \
    numpy \
    opencv \
    pandas \
    pandoc \
    plotly \
    pytorch \
    scikit-learn \
    scipy \
    seaborn \
    spacy \
    tabulate \
    tensorflow \
    torchvision \
    transformers
