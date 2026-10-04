# Provenance and external assets

The wall backend contracts, command-line interface, training baseline, tiling, graph tracing, and output adapters were independently implemented for this repository. The release does not include the prior application's network/model source, business APIs, vendor code, OCR integration, dataset, server configuration, or Git history.

U-Net, skeletonization, distance transforms, connected components, and polyline simplification are established techniques. The implementation uses OpenCV, SciPy, and scikit-image; their licenses continue to apply to those dependencies.

The reference application identified [CubiCasa5k](https://github.com/CubiCasa/CubiCasa5k) and contained YOLOv5 files marked GPL-3.0. Those source files are not part of this repository. CubiCasa5k's [license](https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE) is CC BY-NC 4.0; the new repository's MIT license grants no rights to external datasets or weights.

External ONNX models were used through a generic inference interface in an authorized private environment. Their training exposure, class semantics, provenance, and redistribution rights remain separate concerns. Neither the private reference model nor the newly trained research weights are distributed here.

Software citations should identify this repository and the version used. A software release is not a claim that the underlying algorithms were invented here.
