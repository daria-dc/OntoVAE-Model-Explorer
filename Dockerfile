FROM python:3.10.6-alpine3.16

ENV APP_HOME /app 
WORKDIR $APP_HOME 

RUN apk add --update --no-cache py3-numpy py3-pandas
ENV PYTHONPATH=/usr/lib/python3.10/site-packages

COPY ./requirements.txt ./
RUN pip install -r ./requirements.txt

RUN mkdir ./assets
COPY ./assets/* ./assets/
COPY ./app.py ./

EXPOSE 8080

CMD python app.py