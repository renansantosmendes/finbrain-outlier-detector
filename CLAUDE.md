# CLAUDE.md

## About this project

`finbrain-outlier-detector` is the starting API for a graduate course lesson.
It trains an Isolation Forest per ticker on daily prices, stores the model
and serves it through FastAPI on Vercel. See README.md for the commands.

## About Python code

- Use `uv` as the package manager, for every install and every run.
- Write all docstrings in English, with `Parameters` and `Returns` sections
  and an `Example` section when the function is complex.
- Keep docstring lines within 79 characters (PEP 8 / PEP 257).
- Use descriptive English names for variables, functions, classes and methods.
- Use type hints on parameters and return values, but not on simple
  variable assignments inside function bodies.
- Do not add comments to the code. Keep only docstrings.
- Break any signature with two or more parameters across lines, one
  parameter per line.
- Every change to production code comes with unit tests under `tests/`,
  mirroring the structure of `src/outlier_detector/`.
- Keep `requirements.txt` in sync with `uv.lock` (see README).

## About the workflows

- `01 - Continuous Integration` (`ci.yml`) runs on every push and pull
  request: structure, lint, unit tests, API smoke test and security scan.
- `02 - Train Model` (`train.yml`) waits for the CI of the commit, trains the
  requested tickers and uploads `models/*.joblib` and `models/*.json` as the
  `trained-models` artifact.
- `03 - Deploy API to Vercel` (`deploy.yml`) takes the run id of a successful
  Train run, downloads its artifact into `models/`, deploys to Vercel and
  checks `/health` and `/models`.
- The three workflows run separately. Do not chain them with `needs` across
  files.

## About secrets and generated files

- Never hardcode tokens. Read them from GitHub secrets or from `.env`.
- Trained models, caches and coverage reports are generated files and must
  not be committed. Delete the caches that the tools leave behind (`__pycache__`,
  `.pytest_cache`, `.coverage`, `coverage.xml`, `junit-unit.xml`) when you finish.
- Never delete `.venv`, `.env` or the `models/` folder of the user.

## About documentation

On every change, check README.md, this file and the docstrings of the code
that changed, and update them in the same commit.
