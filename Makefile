.PHONY: all dataset model demo test clean

all: dataset model
dataset:
	python src/build_dataset.py
model:
	python src/model.py
demo:
	python scripts/generate_demo_data.py
	python src/build_dataset.py
	python src/model.py
	python src/similarities.py --player "Demo Prospect 001"
test:
	pytest -q
clean:
	rm -rf data/processed/* outputs/*

