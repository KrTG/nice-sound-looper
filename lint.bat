isort src
black src
flake8 src
vulture --min-confidence 90 src
