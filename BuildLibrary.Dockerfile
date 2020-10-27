FROM python:3.8.5
COPY . /nasrl/source
WORKDIR /nasrl/source
RUN pip install -r requirements.txt
RUN pip wheel . -w /nasrl/install && pip install $(find /nasrl/install -type f -iname "*.whl")
