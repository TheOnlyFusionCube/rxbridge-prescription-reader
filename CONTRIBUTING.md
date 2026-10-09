# Running RxBridge

## Backend

The API tests need docTR, which is not in the system Python. Use the project
virtualenv, which pins the versions that load the docTR and DistilBERT
checkpoints:

```bash
.venv/bin/python -m pytest tests/ -q      # 96 tests
.venv/bin/python -m uvicorn app.main:app --port 8000
```

If you rebuild `.venv`, install the pinned set:

```bash
python3 -m venv .venv
.venv/bin/pip install "numpy==1.26.4" "scipy==1.13.1" "python-doctr==1.0.1" \
  "transformers==4.46.3" "torch==2.11.0" fastapi uvicorn pydantic pytest \
  "opencv-python-headless" seqeval jiwer
```

The schema contract tests run under any interpreter with pydantic:

```bash
/opt/anaconda3/bin/python3 -m pytest tests/test_schema.py -q   # 69 tests
```

## Frontend

```bash
cd web
npm install
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000 npm run build
npx next start -p 3000
```

Leave `NEXT_PUBLIC_API_BASE` unset to run against the bundled sample schedule
(no backend needed).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q          # contract + API + attribution
cd web && npx playwright test                 # browser end-to-end
```
