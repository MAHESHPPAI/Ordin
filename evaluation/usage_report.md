# Token Usage and Evaluation Run Report

## Execution Summary
- **Requests Evaluated**: 250
- **Run Duration**: 1.71 seconds
- **Average Duration per Request**: 6.8 ms
- **Architecture**: 100% Deterministic Algorithmic Pipeline (Stages 0–7)

## Model Usage Statistics
- **Model Providers**: None (0 external AI models invoked)
- **Model Names**: None
- **Total Model Calls**: 0
- **Input Tokens**: 0
- **Output Tokens**: 0
- **Total Tokens**: 0
- **Average Tokens per Request**: 0.0
- **Total Estimated Cost**: $0.0000 USD
- **Per-Request Estimated Cost**: $0.0000 USD

## Compliance Verification
- **Deterministic Decision Path**: All safety checks, candidate generation, candidate selection, and explanations are produced by deterministic logic in `code/`.
- **Untrusted Content Handling**: Messages and OCR images are processed using regex pattern matching and verified OCR extractors; embedded prompt injections are filtered and ignored.
- **Privacy & Security**: Zero customer financial events, balances, or messages are transmitted to any cloud API or external service.
