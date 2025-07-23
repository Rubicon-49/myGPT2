install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

text:
	python -m pytest -vv test_mha.py

format:
	black *.py

lint:
	pylint *.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + ;\
	find . -type f -name "*.pyc" -delete

all: install lint format test