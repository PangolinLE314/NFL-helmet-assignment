FROM python:3.8.7-slim as poetryimg
ENV APP_DIR=/app \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    PATH="${PATH}:/root/.local/bin" \
    POETRY_VERSION=1.1.7
RUN mkdir /skillcorner

WORKDIR /skillcorner

RUN pip install poetry

# build poetry dependencies with custom gcc libraries
# poetry will natively create a dedicated venv in ${pwd}/.venv
FROM poetryimg as poetry-builder
RUN apt-get update -qq \
 && apt-get install -qq git gcc build-essential python-dev\
 && rm -rf /var/lib/apt/lists/*
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    CFLAGS="-fcommon" \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    POETRY_NO_ANSI=1

RUN apt-get -q update && \
	apt-get -q -y install --no-install-recommends \
       	ffmpeg libgl1 libglib2.0-0 \ 
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml poetry.lock ./

RUN poetry install 
COPY . /skillcorner
WORKDIR /skillcorner
CMD ["poetry", "run", "python", "-m", "app"]
