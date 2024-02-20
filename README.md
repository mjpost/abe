The code for ensembling is brewing in `ensemble.py`.
I (MJP) have copied stub code there which we simply need to extend.

## TODO
- script to convert Marian models to Huggingface

## Generalizing models

We need a number of abstractions to support fully ensembling any model

- models may have different preprocessing (e.g., tokenization)
- models may required a forced BOS token (e.g., language to select)
- models may apply post-processing before realizing the target language string (e.g., Marian's casing)