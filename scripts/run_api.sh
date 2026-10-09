#!/bin/bash
cd /Users/faye/rxbridge
export RXBRIDGE_NER_MODEL=models/rxner2
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8010
