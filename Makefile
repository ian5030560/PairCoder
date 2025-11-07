start:
	python src/solve_dataset.py --dataset_name mbpp --split_name test --dir_path results

g4f:
	python -m g4f api --port 8000 --debug --no-gui

embed:
	python src/text_embedding/server.py

install:
	pip install -r requirements.txt

freeze:
	pip freeze > requirements.txt