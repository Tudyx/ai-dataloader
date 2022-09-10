[![CI](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml)
[![Coverage](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml)

# dataloader_rs

A rust port of pytorch dataloader library.

## limitation 

For now support only single threaded map style dataset.
Random sampler n'a pas de version avec replacement.

Dataloader requière une library de multidimensionnal array qui a les opération suivante:
- stack
- transpose
- Any number of dimension
  
ndarray est la seule actuellement et il y a pas de version sur le GPU (Not on the roadmap yet https://github.com/rust-ndarray/ndarray/issues/840)

## TODO:

- finir la doc
- préparer le post sur reddit
- publier
- Rexeporter les symbols
- voir comment Serde fait pour tester ses différent  type supporté.

### Low priority

- trait for batchSampler
- collect_fn comme closure -> unstable
- RandomSampler avec replacement
- multithreading
