echo Running Flask App...
cd ~/time_series_prediction_flask_api
python -m flask run &

echo Running React App...
cd ~/time_series_prediction_react 
npm start &