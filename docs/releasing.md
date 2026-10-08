# Releases and downloads

WallGraph is distributed on [GitHub Releases](https://github.com/crown-sports/wallgraph/releases). It has not been uploaded to PyPI. Initial 0.x releases are marked experimental pre-releases.

Each release supplies:

| Download | Contents |
| --- | --- |
| `wallgraph-source.zip` | Audited source, tests, docs, community files, CI, and the single generated diagram |
| `wallgraph-VERSION.tar.gz` | Python source distribution |
| `wallgraph-VERSION-py3-none-any.whl` | Installable Python code and metadata; no demo image or model |
| `SHA256SUMS` | Checksums for the three artifacts |

GitHub also generates source archives from the public tag. All downloads exclude real datasets, annotations, pretrained weights, private results, credentials, and previous application history. The MIT license covers this repository's code, not external model/data assets.

After downloading all three artifacts and `SHA256SUMS`, verify them with `sha256sum -c SHA256SUMS` on Linux, or `shasum -a 256 -c SHA256SUMS` on macOS. Install the wheel with `python -m pip install ./wallgraph-VERSION-py3-none-any.whl`, replacing VERSION with the release version. ONNX inference needs the optional runtime extra; training needs the training extra.

## Maintainer procedure

Use a clean standalone checkout of this repository. Update the package version and `__version__`, `CHANGELOG.md`, `CITATION.cff`, and validation notes before preparing a tag. For 0.x, document changes to schemas or configuration explicitly.

```bash
python -m pip install -e '.[dev,onnx,train]' onnx
ruff check .
ruff format --check .
pytest
python tools/render_demo.py --check
python tools/release.py --check
python -m build
python tools/release.py --output dist/wallgraph-source.zip
```

Review the exact repository tree and all archive members, including metadata. Use only the explicit public allowlist when extracting code from a private application; initialize fresh Git history and never push that application's repository. Check installation and demo execution using the built wheel in a fresh environment.

Push the reviewed source and let the matching commit pass CI. Tag that commit, create a draft release, upload all three artifacts and checksums, verify their hashes through the release asset API or a download, then publish the experimental release. No automated model or data upload is configured.

The checks workflow also supports manual runs. If a push did not start a usable run, use `gh workflow run ci.yml --repo crown-sports/wallgraph --ref main`. Confirm the run's commit SHA and all three jobs (Python 3.10, Python 3.12, and training) before tagging; a successful run on an older commit does not validate the release.

This process follows GitHub's [community profile](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/about-community-profiles-for-public-repositories), [release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository), and [Actions security](https://docs.github.com/en/actions/reference/security/secure-use) guidance. CI uses read-only repository permissions and pinned action commits.

The current main README uses `examples/simple.png` as a generated input/output comparison figure. `python tools/render_demo.py` regenerates it from `draw_demo()` and the actual default pipeline; CI checks its pixels and the six-path/four-node counts. The CLI demo still constructs its input in memory. The comparison figure is a visual explanation, not an inference input or a dataset sample. Tagged releases remain their original snapshots.
