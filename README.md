# Chronis HSSM backbone

## Environment
Target Python is **3.12** (`.python-version`); scikit-learn 1.9.1 needs 3.11 or newer. Create a virtual environment with `python3.12 -m venv .venv`, activate it, then `pip install -r requirements.txt`.
Check it with `python scripts/check_env.py` (fails if an installed version differs from `requirements.txt`) and run the tests from the repository root with `pytest`. Libraries for other modules are in `requirements-optional.txt`.
