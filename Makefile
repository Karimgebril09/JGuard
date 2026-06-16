.PHONY: backend dashboard dashboard-build

backend:
	python -m uvicorn backend.app.main:app --reload

dashboard-build:
	dotnet build ..\Dashboard\JGuard\JGuard.csproj -p:Platform=x64

dashboard:
	dotnet run --project ..\Dashboard\JGuard\JGuard.csproj -p:Platform=x64