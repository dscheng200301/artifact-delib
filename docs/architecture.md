# Architecture

HistoDelib separates fixture/data validation, normalized model requests, safe API infrastructure, modality-isolated methods, structured predictions, metrics, and CLI commands. Mock API clients implement the same normalized request/response schema as an OpenAI-compatible client. Paid calls are disabled by default, and generated fixture artifacts are not research results.
