FROM continuumio/miniconda3:latest 

WORKDIR /app

COPY enviroment.yml enviroment.yml 

RUN conda config --add channels conda-forge

RUN conda env create --name="PDE_III" -f enviroment.yml

SHELL ["conda", "run", "-n", "PDE_III", "/bin/bash", "-c"]

COPY diseases.json /app/diseases.json
COPY leaves.py /app/leaves.py

ENTRYPOINT ["conda", "run", "-n", "PDE_III", "python3", "leaves.py"]
