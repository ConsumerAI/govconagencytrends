from __future__ import annotations

# Bump when component behavior changes; included in cache keys and run diagnostics.
SCHEMA_VERSION = "package_extraction.v1"
PROMPT_VERSION = "package_extraction_prompt.v1"
CORPUS_BUILDER_VERSION = "package_corpus.v2"
FORM_EXTRACTOR_VERSION = "pdf_forms.v1"
VALIDATOR_VERSION = "evidence_match.v2"
CANONICALIZER_VERSION = "canonicalize.v2"
RESOLVER_VERSION = "resolve.v2"

FAST_SCHEMA_VERSION = "market_scope_fast.v2"
FAST_PROMPT_VERSION = "market_scope_fast_prompt.v2"
FAST_CORPUS_VERSION = "scope_corpus.v2"

PROFILE_VERSIONS: dict[str, dict[str, str]] = {
    "package_llm_full": {
        "profile": "package_llm_full",
        "schemaVersion": SCHEMA_VERSION,
        "promptVersion": PROMPT_VERSION,
        "corpusBuilderVersion": CORPUS_BUILDER_VERSION,
        "formExtractorVersion": FORM_EXTRACTOR_VERSION,
        "validatorVersion": VALIDATOR_VERSION,
        "canonicalizerVersion": CANONICALIZER_VERSION,
        "resolverVersion": RESOLVER_VERSION,
    },
    "market_scope_fast": {
        "profile": "market_scope_fast",
        "schemaVersion": FAST_SCHEMA_VERSION,
        "promptVersion": FAST_PROMPT_VERSION,
        "corpusBuilderVersion": FAST_CORPUS_VERSION,
        "formExtractorVersion": FORM_EXTRACTOR_VERSION,
        "validatorVersion": VALIDATOR_VERSION,
        "canonicalizerVersion": CANONICALIZER_VERSION,
        "resolverVersion": RESOLVER_VERSION,
    },
}
