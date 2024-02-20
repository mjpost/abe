The code for ensembling is brewing in `ensemble.py`.
I (MJP) have copied stub code there which we simply need to extend.

## TODO
- [ ] Ensemble the same model twice (passed in as two models)
- [ ] Ensemble two models with a shared vocabulary
- [ ] Build a model bundle class
- [ ] Ensemble two models with different vocabularies
- [ ] Add script to convert Marian models to Huggingface
- [ ] Create a Marian model class for HF (maybe already exists?)

## Generalizing models

We need a number of abstractions to support fully ensembling any model

- models may have different preprocessing (e.g., tokenization)
- models may required a forced BOS token (e.g., language to select)
- models may apply post-processing before realizing the target language string (e.g., Marian's casing)

## Design CLI

echo This is a test | gensemble -m facebook/m2m facebook/nllb meta/llama /path/to/model/bundle --target-lang fr

## Generalized model wrapper

We need something like this:

model name -> params(
    - model name to download from huggingface
    - associated tokenizer
    - does it need BOS token (at instantiation or decoding time)
    - max length
    - etc
)