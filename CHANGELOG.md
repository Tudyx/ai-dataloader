# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.3.0] - 2022-01-11
### Added
- more strict compiler warnings
- `Dataloader` for iterable datasets. A dalaloader working for any type that implements `IntoIterator`
- shuffling support for iterable dataset
- project folder reorganization
- default collate now support reference


## [0.2.1] - 2022-27-09
### Fixed
- Fix link in the doc
- Add missing documentation field in package metadata
## [0.2.0] - 2022-27-09
### Added
- Indexable dataset
- `DefaultCollate` function
- `Dataloader` and `DataloaderBuilder`
- Sequential and random `Sampler`


[Unreleased]: https://github.com/Tudyx/ai-dataloader/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Tudyx/ai-dataloader/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/Tudyx/ai-dataloader/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Tudyx/ai-dataloader/compare/v0.1.0...v0.2.0