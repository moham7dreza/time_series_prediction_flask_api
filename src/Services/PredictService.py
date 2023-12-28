from src.Migrations import db
from src.Migrations.Predict import Predict


class PredictionService:
    def index(self):
        return Predict.query.all()

    def create(self, task_content):
        pred = Predict(task=task_content)
        db.session.add(pred)
        db.session.commit()

    def update(self, task_id, new_task_content):
        pred = Predict.query.get(task_id)
        pred.task = new_task_content
        db.session.commit()

    def destroy(self, task_id):
        task_to_delete = Predict.query.get(task_id)
        db.session.delete(task_to_delete)
        db.session.commit()
