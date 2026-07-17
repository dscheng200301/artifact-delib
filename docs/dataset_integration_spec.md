# Dataset Integration Specification

Future manifests must provide an image path, caption, `TRUE`/`MISCAPTIONED`/`OUT_OF_CONTEXT` label, original-image group ID, source, license, domain, conflict type, data version, and split. Import validation must reject missing/unreadable images, blank captions, invalid labels, duplicate IDs, path traversal, and group leakage across train/validation/test. No formal dataset is selected or downloaded in this phase.
