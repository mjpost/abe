

## Test

    paste <(echo "This is a test." | ensembling/build/src-tgt "__en__" "__fr__") \
            <(echo "This is a test." | ensembling/build/src-tgt "eng_Latn" "fra_Latn")
            | python ensembling/ensemble.py --models facebook/m2m100_418M facebook/nllb-200-distilled-600M -> C'est un test.

## Setup

Installation:

    python3 -m venv venv
    . venv/bin/activate
    
We should fill this out with the requirements etc. We also need to add our unit tests here.

## Rachel Changes

- everything that wasn't used has been removed (extraneous model functions)
- loading and calls are now model-agnostic (hopefully). Everything is loaded with the `AutoModelWithLMHead` function (apparently this model will soon/eventually be deprecated) and `AutoTokenizer`
- batch size now works (I would continue to test with batch=1, sensitive to memory issues)
- I moved a lot of the search code to its own file `search.py` to clear up `ensemble.py` which just handles the high level stuff
- There's a new function `get_sorted_output_extensions` which handles the logic for what the model produces. Right now that's just either skipping (via pad/stall) or sorting. In the future, if there's more ways to optimize this, we can constrain the search.
- I exit the search early when there is at least one beam to continue with and the search depth exceeds 10k--this significantly speeds up the model when it gets stuck sometimes



## TODO
- [ ] Ensemble the same model twice (passed in as two models)
- [ ] Ensemble two models with a shared vocabulary
- [ ] Ensemble two models with different vocabularies
- [ ] Add script to convert Marian models to Huggingface
- [ ] Create a Marian model class for HF (maybe already exists?)


### Unit Tests
- [ ] given some output, run it through individual models to get the token level scores

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

