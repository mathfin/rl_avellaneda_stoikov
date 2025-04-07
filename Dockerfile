FROM python:3.10

WORKDIR /src
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 torch --index-url https://download.pytorch.org/whl/cu118 --no-cache-dir

COPY requirements.txt /src
RUN pip install --no-cache-dir -r requirements.txt

COPY . /src
