@echo off

echo Running Flask App...
start cmd /c "cd c:\CODEX\PYTHON\time_series_prediction_flask_api && python -m flask run"

echo Running React App...
start cmd /c "cd c:\CODEX\REACT\time_series_prediction_react && npm start"
