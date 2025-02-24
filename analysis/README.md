Used to generate Table 3.

The evaluate-bilingual.py file requires our custom models (for instance [rewicks/baseline_en-de_64k_ep25](https://huggingface.co/rewicks/baseline_en-de_64k_ep25)):

```
paste <(cat ../refs/wmt24.en-de.en.sentences | cut -f2) ../baselines/simple-translations/outputs/en-de/sentences/m2m100_1.2B ../baselines/simple-translations/outputs/en-de/sentences/nllb-200-3.3B ../translations/wmt24/en-de/sentences/m2m100_1.2B+nllb-200-3.3B | python evaluate-multilingual.py facebook/m2m100_1.2B facebook/nllb-200-3.3B
```


 The evaluate-multilingual.py runs as:

```
paste <(cat ../refs/wmt24.en-de.en.sentences | cut -f2) ../baselines/simple-translations/outputs/en-de/sentences/m2m100_1.2B ../baselines/simple-translations/outputs/en-de/sentences/nllb-200-3.3B ../translations/wmt24/en-de/sentences/m2m100_1.2B+nllb-200-3.3B | python evaluate-multilingual.py facebook/m2m100_1.2B facebook/nllb-200-3.3B
```
