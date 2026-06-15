__all__ = ["run_package_llm_extraction"]


def __getattr__(name: str):
    if name == "run_package_llm_extraction":
        from extraction.package_llm.run import run_package_llm_extraction

        return run_package_llm_extraction
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
