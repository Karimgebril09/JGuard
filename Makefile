.PHONY: backend

backend:
	python -m uvicorn backend.app.main:app --reload