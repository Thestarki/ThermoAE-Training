run: install
	./src/runner.py

install: pyproject.toml
	poetry install

clean: 
	rm -rf 'find . -type d -name __pycache__'