# Live Flood Forecast Dashboard with LSTM and GRU

This repository runs daily flood prediction models for the Huron River and Sandusky River watersheds and generates an interactive forecast dashboard. The baseline water-level forecast includes Random Forest, LSTM, and GRU models. Runoff-reduction and storage scenarios use the Random Forest model.

## Main Files

- `version_9.py` - daily flood prediction model and dashboard generator with Random Forest, LSTM, and GRU baseline forecasts
- `input_huron.csv` - historical daily water-level and precipitation input data for Huron River
- `input_sandusky.csv` - historical daily water-level and precipitation input data for Sandusky River
- `Watershed Shapefile/Huron River/data.geojson` - Huron River watershed boundary used in the interactive map
- `Watershed Shapefile/Sandusky River/download/layers` - Sandusky River watershed shapefile used in the interactive map
- `index.html` - generated interactive dashboard for GitHub Pages
- `requirements.txt` - Python dependencies for GitHub Actions, including TensorFlow for LSTM and GRU
- `.github/workflows/daily-flood-forecast.yml` - scheduled GitHub Actions workflow

## Automation

GitHub Actions runs the model every day at 1:00 AM Eastern during daylight saving time, regenerates the dashboard and output files, and commits updated outputs back to the repository.

The workflow can also be started manually from the repository's Actions tab.
