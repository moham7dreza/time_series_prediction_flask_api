from src.Helper.Helper import Helper
from src.Migrations import db
from src.Migrations.Predict import Predict


class PredictionService:
    def index(self):
        return Predict.query.all()

    def create(self, request):
        pred = Predict(n_steps=request.get('n_steps'), datasets=Helper.implode(request.get('dataset')),
                       models=Helper.implode(request.get('model')),
                       prices=Helper.implode(request.get('price')), series=Helper.implode(request.get('serie')))
        db.session.add(pred)
        db.session.commit()
        return pred

    def update(self, task_id, new_task_content):
        pred = Predict.query.get(task_id)
        pred.task = new_task_content
        db.session.commit()

    def destroy(self, task_id):
        task_to_delete = Predict.query.get(task_id)
        db.session.delete(task_to_delete)
        db.session.commit()
