include ../.env

format:
	poetry run black ./src

lint:
	poetry run black --check --diff ./src
	poetry run flake8 ./src

build-and-push:
	./build_and_push.sh ${IMAGE_NAME}
