# Variables
PYTHON = python3
APP_MODULE = app.main:app
PYTHONPATH = PYTHONPATH=.

# Comandos principales
.PHONY: all dataset train run clean

all: dataset train run

# Generar el dataset
dataset:
	$(PYTHONPATH) $(PYTHON) -c "from app.utils.dataset_generator import generate_dataset; generate_dataset()"

# Entrenar el modelo de obesidad
train-obesity:
	$(PYTHONPATH) $(PYTHON) app/utils/obesity/obesity_ensemble_model.py

# Entrenar el modelo de diabetes
train-diabetes:
	$(PYTHONPATH) $(PYTHON) app/utils/diabetes/diabetes_ensemble_model.py

# Entrenar el modelo de hipertensión
train-hypertension:
	$(PYTHONPATH) $(PYTHON) app/utils/hypertension/hypertension_ensemble_model.py

train-all: train-obesity train-diabetes train-hypertension
# Ejecutar el servidor con uvicorn en modo reload
run:
	$(PYTHONPATH) uvicorn $(APP_MODULE) --reload

# Limpiar archivos generados
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete 