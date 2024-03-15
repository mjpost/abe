The code for ensembling is brewing in `ensemble.py`.
I (MJP) have copied stub code there which we simply need to extend.

## Setup

Installation:

    python3 -m venv venv
    . venv/bin/activate
    pip install -r requirements.txt

Testing:

    # This will interpolate two identical models (nllb)
    echo "This is a test." | ./ensemble.py -b 10

    # This adds noise to the logits of the first model, so interpolotion makes a bit more sense
    echo "This is a test." | ./ensemble.py -b 10 --noise 2

    # German
    echo "This is a test." | ./ensemble.py -b 5 -t deu_Latn

## TODO
- [x] Ensemble the same model twice (passed in as two models)
- [ ] Ensemble two models with a shared vocabulary
- [x] Build a model bundle class
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

## Files

- Definition of generate(): /Users/mattpost/src/transformers/src/transformers/generation/utils.py

  This is a version that's generate over beam search, sampling, constrained search, etc.
  Actually calls beam_search() in the same file.

## Models

- "facebook/m2m100_418M": 128104 tokens
- "facebook/nllb-200-distilled-600M": 256204 tokens
- "facebook/nllb-moe-54b": 256204 tokens 

## Usage

    echo "This is a test." | ./ensemble.py -t fr -m facebook/m2m100_418M facebook/nllb-200-distilled-600M --num-beams 5 -l 10
    echo "This is a test." | ./ensemble.py -t fr -m facebook/m2m100_418M facebook/m2m100_1.2B --num-beams 5 -l 10

