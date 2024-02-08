# Point-to-hyperplane nearest neighbours search (P2HNNS)
A Python library for efficient Point-to-hyperplane nearest neighbours search (P2HNNS) using locality sensitive hashing (LSH) approaches.
The library implements the 5 different methods described below.

- Bilinear Hyperplane (BH) hashing 
- Embedding Hyperplane (EH) hashing
- Multilinear Hyperplane (MH) hashing
- Nearest Hyperplane (NH) hashing
- Furthest Hyperplane (FH) hashing

The implementation is based on the original code of [HuangQiang](https://github.com/HuangQiang/P2HNNS) (implemented in C++) and [stepping1st](https://github.com/stepping1st/hyperplane-hash/tree/master)(implemented in Java).

The original papers proposing each method will be explicitly provided in section [Resources](#resouces).

## Installation
The library can be installed via the pip package manager using the following command
```
pip install P2HNNS
```

## Documentation
Extensive documentation for using the library is available via [Read the Docs](https://p2hnns.readthedocs.io/en/latest/index.html)

## Tests
Unit tests are written using the pytest framework for all functionalities of the library. Tests are located in the [/tests](/tests/) directory.

## Resources
- [Point-to-Hyperplane Nearest Neighbor Search Beyond the Unit Hypersphere, SIGMOD 2021](https://dl.acm.org/doi/abs/10.1145/3448016.3457240) (Original FH and NH paper)
- [Compact Hyperplane Hashing with Bilinear Functions, ICML 2012](https://arxiv.org/abs/1206.4618) (Original BH paper)
- [Hashing Hyperplane Queries to Near Points with Applications to Large-Scale Active Learning, NeurIPS 2010](https://proceedings.neurips.cc/paper/2010/hash/470e7a4f017a5476afb7eeb3f8b96f9b-Abstract.html) (Original EH paper)
- [Multilinear Hyperplane Hashing,CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/papers/Liu_Multilinear_Hyperplane_Hashing_CVPR_2016_paper.pdf) (Original MH paper)
- [Original C++ implementation by Huang Qiang](https://github.com/HuangQiang/P2HNNS)
- [Java implementation](https://github.com/stepping1st/hyperplane-hash)

## License
The library is licensed under the MIT Software license. You can see more details in the [LICENSE file](./LICENSE).