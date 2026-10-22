# Where WallGraph fits

**User and decision:** a developer with wall labels needs paths and shared junction IDs for a drawing interface or downstream geometry. The model-free `vectorize` path is the shortest entry. With only scans, first bring a compatible model; [training/import instructions](training-guide.md) and [Chinese instructions](training-guide.zh-CN.md) are available, but public weights are not supplied.

| Current, checkable value | Evidence | Boundary |
| --- | --- | --- |
| Preserve supplied binary pixels and original coordinates unless cleanup is explicit | [`masks.py`](../src/wallgraph/masks.py), [`test_mask_import.py`](../tests/test_mask_import.py) | Membership scores from an imported mask are not detector probabilities |
| Expose paths, junctions and the closing edge of a ring | [`vectorize.py`](../src/wallgraph/vectorize.py), ring and T-junction tests in [`test_geometry.py`](../tests/test_geometry.py) / [`test_mask_import.py`](../tests/test_mask_import.py) | Skeletonization and simplification still depend on mask quality |
| Feed exact original-grid results into room/measurement tools | [Shared executable walkthrough](https://github.com/crown-sports/planportals/blob/main/docs/workflow.md) | No physical scale, opening semantics or CAD/BIM model is inferred here |

Use a short OpenCV/scikit-image script for a one-off mask operation. WallGraph packages the graph export and validation choices when an application must retain them consistently. [scikit-image's skeletonization examples](https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html) remain an appropriate starting point for the underlying operation. A general segmentation model is upstream: it proposes wall pixels; this package exposes what happened to those pixels. No comparative speed or accuracy advantage over those tools is established.

[Validation](validation.md) retains the unfavorable results too: downstream region PQ fell under the evaluated cleanup, and raw segmentation scored better on wall IoU. The private-data/model evaluation is documented but not independently reproducible from this repository; the generated examples and tests are reproducible.

**Next priorities:** publish permissioned continuity/opening failure cases with correction history, then validate whether graph and room errors improve together. Reusable contracts and tested failure handling are useful engineering today; proprietary data, unique algorithms, production reliability and a durable competitive advantage have not been demonstrated.

The only observed public request, [issue #1](https://github.com/crown-sports/wallgraph/issues/1), asks vaguely for import/training details. It motivates making these existing guides easier to find, not assuming a particular training feature or validated market demand.
