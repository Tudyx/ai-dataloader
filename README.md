[![CI](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/ci.yml)
[![Coverage](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml/badge.svg)](https://github.com/Tudyx/dataloader_rs/actions/workflows/codecov.yml)

# dataloader_rs

A rust port of pytorch dataloader library.

## limitation 

For now support only single threaded map style dataset.
Random samler n'a pas de version avec replacement.


## TODO:
- ajout de TU (copier _test_squential de test_dataloader)
- ajout de TU default collect copier de test_dataloader (faire + macro + macro dans les test)
- utilisation de Index au lieu de GetItem
- RandomSampler avec replacement
- trait for batchSampler
- collect_fn comme closure -> unstable
- macro default collect pour unpack les vec de vec/slice
- default collect pour les tuple et hasmap
- multithreading
- guide for pytorch user (comme le guide pour numpy user de ndarray)
