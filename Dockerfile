FROM python:3.8-slim-buster

WORKDIR /app

COPY *.py ./
COPY requirements.txt requirements.txt
COPY data/ ./data/
COPY .git/ ./git/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

