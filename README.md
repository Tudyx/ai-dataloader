[![CI](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml)
[![Coverage](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml)

# dataloader_rs

A rust port of pytorch dataloader library.

## limitation 

For now support only single threaded map style dataset.
Random samler n'a pas de version avec replacement.


## TODO:
- cleanup le default collect de vector
- voir si je converti les string ou pas
- clarifier l'histoire de niveau de récursion autorisé
- finir la doc
- préparer le post sur reddit
- publier

## Low priority

- trait for batchSampler
- collect_fn comme closure -> unstable
- macro default collect pour unpack les vec de vec/slice -> faisable avec seulement une déclarative macro
- RandomSampler avec replacement
- multithreading
- iterable dataset
