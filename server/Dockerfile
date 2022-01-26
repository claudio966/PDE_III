# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster 

RUN apt-get update && apt-get install -y python3-opencv 
# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install pillow
EXPOSE 5000

COPY . .

CMD ["flask", "run", "--host=0.0.0.0"]
