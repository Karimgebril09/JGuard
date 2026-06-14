"""Garak integration package."""

__all__ = ["GarakRunResult", "build_dataset_from_garak", "run_with_config_defaults"]


def __getattr__(name: str):
	if name in __all__:
		from . import garak_pipeline

		return getattr(garak_pipeline, name)
	raise AttributeError(name)
