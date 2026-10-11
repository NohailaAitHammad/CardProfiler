lien du daily meeting: https://docs.google.com/document/d/1yFMruIR_7E1-nZt0jh3gPXdz8Kqyrpf6vUKWu9lpYLw/edit?usp=sharing# CardProfiler



uv run python -m src.main
mlflow experiments restore --experiment-id 3
uv run mlflow experiments search --view deleted_only
uv run mlflow experiments delete --experiment-id 3