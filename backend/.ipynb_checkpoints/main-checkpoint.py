# backend/main.py
# docker-compose up -d --build
# docker-compose up
from celery import Celery
from celery.result import AsyncResult
import time
from datetime import timedelta
from flask import Flask, request
import json
import pickle
import re
from inference import predict, celeryapp


app = Flask(__name__)  # Основной объект приложения Flask


@app.route('/')
def hello():
    return "Hello, from Chatbot!"

@app.route('/bot', methods=["GET", "POST"])
def predict_handler():
    if request.method == 'POST':
        data = request.get_json(force=True)
        task = predict.delay(data["bot"])
        response = {
            "task_id": task.id
        }
        return json.dumps(response)
    
@app.route('/bot/<task_id>')
def predict_check_handler(task_id):
    task = predict.AsyncResult(task_id, app = celeryapp)
    if task.ready():
        response = {
            "status": "DONE",
            "result": task.result
        }
    else:
        response = {
            "status": "IN_PROGRESS"
        }
    return json.dumps(response)
    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)