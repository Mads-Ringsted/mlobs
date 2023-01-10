FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY data/ data/
COPY main.py main.py
COPY data.py data.py
COPY model.py model.py

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
ENTRYPOINT ["python", "-u", "main.py"]
