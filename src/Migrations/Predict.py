from src.Migrations import db


class Predict(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    series = db.Column(db.String(255), nullable=False)
    models = db.Column(db.String(255), nullable=False)
    prices = db.Column(db.String(255), nullable=False)
    databases = db.Column(db.String(255), nullable=False)
    n_steps = db.Column(db.String(255), nullable=False)
