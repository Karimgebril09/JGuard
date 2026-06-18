"""Promptfoo red-teaming integration package.

Mirrors the ``garak`` package interface so promptfoo can be used as a drop-in
red-teaming tool that emits the same attack-prompt dataset.
"""

__all__ = ["PromptfooRunResult", "build_dataset_from_promptfoo", "run_with_config_defaults"]


def __getattr__(name: str):
    if name in __all__:
        from . import promptfoo_pipeline

        return getattr(promptfoo_pipeline, name)
    raise AttributeError(name)
