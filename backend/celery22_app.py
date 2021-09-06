from celery import Celery
import inference


celery = Celery(
    'worker', 
    broker='redis://localhost:6379',
    backend='redis://localhost:6379'
)


@celery.task
def predict(seq):
    result = inference.evaluation(seq)
    return result