include ../.env

format:
	poetry run black ./src

lint:
	poetry run black --check --diff ./src
	poetry run flake8 ./src

shell:
	env $(cat ../.env) poetry shell
