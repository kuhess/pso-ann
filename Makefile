init:
	pip install -r requirements.txt

test: init
	py.test

example: init
	python example_mnist.py
	

.PHONY: init test example
