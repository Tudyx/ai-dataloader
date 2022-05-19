# dataloader_rs

A rust port of pytorch dataloader library.


For now support only single threaded map style dataset.

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