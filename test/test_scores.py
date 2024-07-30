from collections import namedtuple
import pytest

# import ..ensemble.models.get_model_bundle
from ensembling.models import get_model_bundle, Bundle
from ensembling.ensemble import ensemble_beam_search


EPSILON = 1e-8


test_inputs = [
    (["facebook/m2m100_418M", "facebook/nllb-200-distilled-600M"], "de", "This is a test.", "Das ist ein Test.", -5.491171031462455 ), #m2m100 (-5.681196157753042) nllb (-5.301145905171868)
    (["facebook/m2m100_418M", "facebook/nllb-200-distilled-600M"], "fr", "This is a test.", "C'est un test.", -7.940011596468024), #m2m100 (-10.358879269814091) nllb (-5.521143923121955)
]

@pytest.mark.parametrize("model_list, target_lang, source, reference, score", test_inputs)
def test_ensemble_scores(model_list, target_lang, source, reference, score):

    models = []
    for model_name in model_list:
        models.append(get_model_bundle(model_name, target_language=target_lang))

    outputs = ensemble_beam_search(source, models, num_beams=5, max_length=50)
    translated_text = outputs[0]
    beam_score = outputs[1].item()

    print(translated_text, beam_score)

    assert translated_text == reference
    print(translated_text == reference)

    assert abs(beam_score - score) < EPSILON
    print(abs(beam_score - score) < EPSILON)

