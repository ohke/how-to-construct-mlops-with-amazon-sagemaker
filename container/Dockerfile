FROM python:3.9-slim AS dev

RUN apt-get update && \
    apt-get install -y build-essential nginx && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /root
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev && \
    rm -rf ~/.cache/pypoetry

WORKDIR /opt/program

FROM dev

COPY src /opt/program

RUN mv ./train.py ./train && \
    mv ./serve.py ./serve && \
    chmod +x ./train ./serve

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
