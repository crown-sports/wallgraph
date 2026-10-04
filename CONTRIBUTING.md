# Contributing to WallGraph

Small fixes can be proposed directly as pull requests. For larger changes, describe the input task, failure, expected behavior, and evaluation plan in an issue before implementation. Follow [community conduct](CODE_OF_CONDUCT.md); use [private reporting](SECURITY.md) for vulnerabilities.

## Development

```bash
git clone https://github.com/chrischen-coder/wallgraph.git
cd wallgraph
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,onnx]' onnx
ruff check .
ruff format --check .
pytest
python tools/release.py --check
python -m build
```

Training changes also need `.[train]` and the training tests. They must verify finite gradients, padding exclusion, and split isolation. CI exercises the core package on Python 3.10/3.12 and the training path on CPU; CUDA behavior needs separate actual-device evidence.

## Design and evidence

Keep the src layout and dependencies directed toward protocols and domain objects. Backends must validate dtype, shape, color/preprocessing, class IDs, finite output, and requested device. Geometry strategies must preserve original pixel coordinates and the `wallgraph/1` contract. Avoid import-time models, hidden downloads, and business-service dependencies.

Tests should expose meaningful contract failures or regressions, not repeat the implementation. For accuracy claims, state sample selection, train/validation/test separation, annotation meaning, macro/micro aggregation, model digest, and failures. For timing, state device, threads, warmup, repetitions, and initialization/I/O scope. Report regressions and uncertainty alongside improvements.

## Public-file boundary

Use generated geometry and minimal code reproductions. Do not commit real drawings, annotations, datasets, weights, per-case results, credentials, or internal endpoints. The repository allows one generated demo PNG; other fixtures must be created in tests. Keep private manifests and raw measurements outside the repository. Preserve source attribution and dependency notices.

Before submitting, run the checks above, update relevant docs and the changelog, and review the actual diff. By contributing, you confirm that you can license the submitted code under MIT and have retained required third-party notices. Release procedure and download contents are in [docs/releasing.md](docs/releasing.md).
