#!/bin/sh
set -e

case "$1" in
  train)
    exec python -m src.train
    ;;
  test)
    exec pytest tests/ -v
    ;;
  app)
    if [ ! -f models/fusion_model.keras ]; then
      echo "Modelos no encontrados en models/ — entrenando antes de arrancar el dashboard..."
      python -m src.train
    fi
    exec streamlit run app/app.py --server.address=0.0.0.0 --server.port=8501
    ;;
  *)
    exec "$@"
    ;;
esac
