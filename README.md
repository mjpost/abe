

## Test

    paste <(echo "This is a test." | ensembling/build/src-tgt "__en__" "__fr__") \
            <(echo "This is a test." | ensembling/build/src-tgt "eng_Latn" "fra_Latn") \
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
- Source and Target tokenizers are different objects, but right now there is no logic to separate them for models where this is not the same
- I removed gaussian noise---easy to add back in if we want to try that again

### JSON inputs

As part of the generalization, I have offloaded the preprocessing to be in the input. The input should be a tab-separated file/stream. The N-th column should be the input to the N-th model. There are 4 possible keys: 

- `encoder_bos_tokens` (the source language code),
- `encoder_inputs` (the source input),
- `decoder_bos_tokens` (the target language code),
- and `decoder_inputs` (the language model prompt)

There is a script, `build/src-tgt`, that will build the appropriate inputs for a traditional multilingual MT-model style input. 

Outputs now look like:

```
{
    "sequence": "Maman a toujours dit que la vie est une boîte de chocolats", 
    "scores": [-7.699125289916992, -8.633319854736328], 
    "combined_score": -8.16622257232666, 
    "token_scores": [
        [-0.6921238303184509, -1.6668376922607422, -0.18656253814697266, -0.09190487861633301, -0.12645173072814941, -0.13176846504211426, -0.17235255241394043, -1.3625507354736328, -0.5222439765930176, -0.3566570281982422, -0.10730648040771484, -0.3345036506652832, -0.03524971008300781, -0.8017196655273438, -0.07809257507324219, -1.0327997207641602], 
        [-2.7465639114379883, -0.12948155403137207, -0.9309823513031006, -0.23229646682739258, -0.18974018096923828, -0.21661853790283203, -0.3390951156616211, -0.18550491333007812, -0.31716394424438477, -0.47531604766845703, -0.8708596229553223, -0.17315959930419922, -0.03824901580810547, -0.1769704818725586, -0.054709434509277344, -0.6767921447753906, -0.8798165321350098]
    ], 
    "tokens": [
        ["</s>", "fra_Latn", "▁Maman", "▁a", "▁toujours", "▁dit", "▁que", "▁la", "▁vie", "▁est", "▁une", "▁bo", "îte", "▁de", "▁cho", "cola", "ts", "</s>"],
        ["</s>", "__fr__", "▁M", "aman", "▁a", "▁toujours", "▁dit", "▁que", "▁la", "▁vie", "▁est", "▁une", "▁bo", "î", "te", "▁de", "▁chocol", "ats", "</s>"]
    ],
    "token_ids": [
        [2, 256057, 184065, 9, 37413, 1329, 340, 82, 6651, 613, 3335, 403, 145334, 79, 1925, 15440, 468, 2],
        [2, 128028, 100, 1905, 8, 19884, 793, 27, 14, 7315, 769, 1269, 502, 9469, 123, 6, 110309, 817, 2]
    ]
}

```

## TODO
- [ ] Ensemble the same model twice (passed in as two models)
- [ ] Ensemble two models with a shared vocabulary
- [ ] Ensemble two models with different vocabularies
- [ ] Add script to convert Marian models to Huggingface
- [ ] Create a Marian model class for HF (maybe already exists?)
- [ ] Add a decoder-only language model such as `Unbabel/TowerInstruct-Mistral-7B-v0.2`

### Unit Tests
- [ ] given some file, for each line in file, get output, run it through individual models to get the token level scores, compare (there are some helpful functions in `utils.py` for tokenization)
- [ ] given some file, for each line in file, run it through individual models. if the translations are the same, ensure the ensemble translation is *also* the same.

## Design CLI

echo This is a test | gensemble -m facebook/m2m facebook/nllb meta/llama /path/to/model/bundle --target-lang fr


## Files

- Definition of generate(): /Users/mattpost/src/transformers/src/transformers/generation/utils.py

  This is a version that's generate over beam search, sampling, constrained search, etc.
  Actually calls beam_search() in the same file.
        

## Notes

Huggingface beam search pops items off the beam when they are finished, and continues generating using normal beam search, until there are enough complete items. Another strategy would be to keep those items in the beam and ignore them.

## Unit Testing

We wrote a script to collect scores from M2M100. It runs like this:

```
sacrebleu -t wmt14 -l en-fr --echo src ref | python unit-tests.py
```

We'll use this to determine logprobs for unit tests.

## Code Review:
