## Context-Aware Embeddings (CAE)

The module produces context-aware embeddings for text documents (utilizes their surrounding features, a.k.a. context). So far, only unsupervised version of the algorithm is implemented.

The concept behind is relatively easy. We basically average documents words vectors, normalize the sum-vector and extract from the obtained embeddings their projection to the principal components which construction is guided through hybrid SVD and additional knowledge of documents context.


The work inspired by the following papers: 

- [HybridSVD: When Collaborative Information is Not Enough](https://arxiv.org/pdf/1802.06398.pdf) [KDD'18] 
- [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx) [ICLR'17] [(code)](https://github.com/PrincetonML/SIF)

Dependences:

- [cholmod](https://github.com/scikit-sparse/scikit-sparse/) (python wrapper for sparse Choletsky decomposition)
- scipy (svd routine)
- numpy

### References

The link to the corresponding paper will be announced later.


