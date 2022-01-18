FROM continuumio/miniconda3
ADD environment.yml /tmp/environment.yml

RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict

RUN conda env create -f /tmp/environment.yml

# Pull the environment name out of the environment.yml
RUN echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

COPY diseases.json ./diseases.json
COPY leaves.py ./leaves.py

CMD ["python", "leaves.py"]
