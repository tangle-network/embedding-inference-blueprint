# Embedding Inference Blueprint

Use [README.md](README.md) for operator setup and [operator/Cargo.toml](operator/Cargo.toml) for the supported SDK dependencies.
Keep shared billing, payment validation, health, and metrics in the `tangle-inference-core` dependency.

## Verification

For HTTP changes, extend [e2e.rs](operator/tests/e2e.rs): start the actual server, replace only the external backend, and assert responses and payment rejection.
[lifecycle.rs](operator/tests/lifecycle.rs) calls the real handler with a backend substitute; it does not submit an on-chain job.
When changing chain integration, exercise submission, operator processing, and result recording through the production runner.
Use the SDK revision selected by the manifests and lockfile for test APIs and fixtures.
Test contract changes with actual deployments under `contracts/`.
Keep tests that detect a meaningful failure; choose verification for the changed behavior.
Report backend substitutes and skipped prerequisites when stating what a test proves.
