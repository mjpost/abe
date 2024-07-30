The code for ensembling is brewing in `ensemble.py`.
I (MJP) have copied stub code there which we simply need to extend.

## Test

    echo "This is test." | python ensembling/translate.py
    - m2m100 -> C'est un test.
    - nllb-200-distilled-600M -> Il s'agit d'un test.



    echo "This is a test." | python ensembling/ensemble.py -t fr -m facebook/m2m100_418M facebook/nllb-200-distilled-600M --num-beams 5 -l 10
    echo "This is a test." | python ensembling/ensemble.py -t fr -m facebook/m2m100_418M facebook/m2m100_1.2B --num-beams 5 -l 10



## Setup

Installation:

    python3 -m venv venv
    . venv/bin/activate
    

Testing:

    # This will interpolate two identical models (nllb)
    echo "This is a test." | ensembling/ensemble.py -b 10

    # This adds noise to the logits of the first model, so interpolotion makes a bit more sense
    echo "This is a test." | ensembling/ensemble.py -b 10 --noise 2

    # German
    echo "This is a test." | ensembling/ensemble.py -b 5 -t deu_Latn

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



2   45   68   0    0      input IDs

-1   0    1   0   -1      positions

              X    X      position mask

1    1    1   0    0      attention mask             

## Notes

Huggingface beam search pops items off the beam when they are finished, and continues generating using normal beam search, until there are enough complete items. Another strategy would be to keep those items in the beam and ignore them.

## Unit Testing

We wrote a script to collect scores from M2M100. It runs like this:

```
sacrebleu -t wmt14 -l en-fr --echo src ref | python unit-tests.py
```

We'll use this to determine logprobs for unit tests.

## Code Review:
Ensemble.py
1. Line 570: Random gaussian noise to logits processor only to model[0]
2. Line 578: Comment that says normally we will use beam search, but we will have to implement it. Does it mean implementing naive beam search? Not sure.
3. Line 202: Should we generalize to n-models? If so, can it be on the lower priority?
4. Line 212: Probably where we should set the first decoder token. Can confirm that it is not set here. </s> + <lang_id>? 
5. Line 213: set_input returns encoder_input_ids, self.encoder_outputs
6. Line 261: Add preprocesser abstraction ToDO - commented by Matt. What needs to be done?
7. 

