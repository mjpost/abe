# Repository Structure

## Ensembling Code

All code to do our method of ensembling can be found in the `ensembling` directory. The important files include `ensemble.py` which contains the main function; `models.py` which has the model wrappers for each model to maintain it's own hidden state; `search.py` which has our cube-pruning-esque search algorithm. `utils.py` has some functions to help with tokenization.


## Data

### Inputs

For all our experiments, we use WMT24 data (en-XX, but mostly en-de).
The raw inputs can be found in `refs`. These were made via commands such as:

```
sacrebleu -t wmt24 -l en-de --echo src > wmt24.en-de.en
sacrebleu -t wmt24 -l en-de --echo ref > wmt24.en-de.de
```

#### Creation

These inputs are unsegmented (multiple sentences per line) which can make some machine translation models add or remove content. To circumvent these issues, we first segment these files into sentences. We then translate, and then reconcatenate. This requires an intermediate file (the sentences with the associated line numbers). We create this using `ersatz`:

```
cat wmt24.en-de.en | awk '{print NR "\t" $0}' | ersatz -m en -C 1 > wmt24.en-de.en.sentences
```

Our ensembling code requires `jsonl` inputs. We provide several scripts to automatically create these from plain text inputs. All scripts are in `ensembling/build/`

1. `bilingual-no-tags` creates inputs for a traditional encoder-decoder model which takes the input line as encoder input and has no additional special tags. We use these for our Marian `en-de` models.
2. `empty` creates an empty input. This would be used for a traditional decoder-only model that does not take prompts.
3. `prompt` creates input for both LLAMA and Tower specifically for translation. This is highly constrained to the set of languages we cover but we provide both 0-shot and 3-shot options. Calling looks like `echo "This is a test." | python ensembling/build/prompt llama3-0-shot English German`
4. `src-tgt` creates input for both M2M and NLLB by taking the source language token and the target language token. Calling looks like `echo "This is a test." | python ensembling/build/src-tgt eng_Latn deu_Latn`


The processed inputs (`jsonl`) can be found in `input_data`. They are labelled by model, and language pair.


# Outputs




# Scoring is