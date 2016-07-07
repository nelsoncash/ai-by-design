FROM gcr.io/tensorflow/tensorflow:latest-devel

RUN apt-get update

ADD . /app

EXPOSE 8000

VOLUME ["..:/app"]

WORKDIR /app/trainer

CMD ["/bin/bash", "-c", "python main.py"]
