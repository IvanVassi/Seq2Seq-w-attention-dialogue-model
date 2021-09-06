FROM python:3

MAINTAINER Ivan Vassilenko

WORKDIR /app

COPY requirements.txt ./requirements.txt
# RUN apt-get install libgtk2.0-dev pkg-config -yqq 

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]