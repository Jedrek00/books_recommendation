# Books recommendation system

TODO add description of the project.

## Installation

To run code from this repository poetry is required. Check if poetry is installed on your machine using command:

```
poetry --version
```

If version of the poetry was displayed, you can install all required packages using command:

```
poetry install
```

or, if you don't want to install jupyter dependencies, use:

```
poetry install --without dev
```

To install poetry on your machine see [documentation](https://python-poetry.org/docs/cli/#install).

All code should be ran in the env created by the poetry. To ensure that you are running scripts in the env created by poetry run:

```
poetry shell
```