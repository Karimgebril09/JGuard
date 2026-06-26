
.PHONY: backend dashboard-build dashboard

backend:
	python -m uvicorn backend.app.main:app

dashboard-build:
	dotnet build ./../Dashboard/JGuard/JGuard.csproj -p:Platform=x64

dashboard:
	dotnet run --project ./../Dashboard/JGuard/JGuard.csproj -p:Platform=x64

tca:
	python defenders/multi_turn/TCA/src/threat_model_train.py
	python defenders/multi_turn/TCA/src/toxicity_model_train.py
	python defenders/multi_turn/TCA/src/primitive_feature_extract_merge.py
	python defenders/multi_turn/TCA/src/risk_param_tune.py
	python defenders/multi_turn/TCA/src/feature_engineered.py
	python defenders/multi_turn/TCA/src/find_best_transformation.py
	python defenders/multi_turn/TCA/src/feature_selection_scale.py
