# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys
import torch
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"

import torch
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    pad_token_id=128001
)


# Translate the following text from English into German. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: White nationalists, gun fetishists, pro-dictator right wing.. there\'s the danger. Deport the white supremacists.
# The German Segment: Weiße Nationalisten, Waffenfetischisten, Diktator-Freundliche vom rechten Flügel … da liegt die Gefahr. Deportiert die weißen Rassisten.
# Translate the following text from English into German. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: It\'s 2023 and I still see people with iPhones actively avoid using Apple Maps.
# The German Segment: Es ist 2023 und ich sehe immer noch Menschen mit iPhones, die aktiv die Verwendung von Apple Maps vermeiden.<|eot_id|>\
# Translate the following text from English into German. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English Segment: #CNN needs to just close down. Yesterday\'s fiasco with #MangoMoron was an outrageous fiasco. Who at that pathetic, dying network was responsible for vetting the so-called \"independent voter\" in the audience? Which cable news \"executive\" made the call to give the fat, lying criminal this much air-time?<|eot_id|>\
# The German Segment: #CNN muss einfach den Laden dichtmachen. Das Fiasko von gestern mit #MangoMoron war einfach ungeheuerlich. Wer war bei diesem jämmerlichen, im Sterben liegenden Sender eigentlich für die Überprüfung der sogenannten „unabhängigen Wähler“ im Publikum verantwortlich? Welcher Kabelnachrichten-„Verantwortliche“ hat denn die Entscheidung getroffen, dem fetten, lügenden Kriminellen so viel Sendezeit zu geben?
# Translate the following segment into ```TARGET```. Do not add any additional content. Do not add parentheticals. Only provide the translation. The ```SOURCE``` segment: ```INPUT_TEXT```<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe ```TARGET``` segment:"

shots = [
    {"role": "user", "content": "Translate the following text from English into German. The English Segment: White nationalists, gun fetishists, pro-dictator right wing.. there's the danger"},
    {"role": "assistant", "content": "Weiße Nationalisten, Waffenfetischisten, Diktator-Freundliche vom rechten Flügel … da liegt die Gefahr. Deportiert die weißen Rassisten."},

    {"role": "user", "content": "Translate the following text from English into German. The English Segment: It's 2023 and I still see people with iPhones actively avoid using Apple Maps."},
    {"role": "assistant", "content": "Es ist 2023 und ich sehe immer noch Menschen mit iPhones, die aktiv die Verwendung von Apple Maps vermeiden."},

    {"role": "user", "content": "Translate the following text from English into German. The English Segment: #CNN needs to just close down. Yesterday's fiasco with #MangoMoron was an outrageous fiasco. Who at that pathetic, dying network was responsible for vetting the so-called \"independent voter\" in the audience? Which cable news \"executive\" made the call to give the fat, lying criminal this much air-time?"},
    {"role": "assistant", "content": "#CNN muss einfach den Laden dichtmachen. Das Fiasko von gestern mit #MangoMoron war einfach ungeheuerlich. Wer war bei diesem jämmerlichen, im Sterben liegenden Sender eigentlich für die Überprüfung der sogenannten „unabhängigen Wähler“ im Publikum verantwortlich? Welcher Kabelnachrichten-„Verantwortliche“ hat denn die Entscheidung getroffen, dem fetten, lügenden Kriminellen so viel Sendezeit zu geben?"}
]


for line in sys.stdin:
    line = line.strip()
    messages = [
        {"role": "user", "content": f"Translate the following segment into German. Do not add any additional content. Do not add parentheticals. Only provide the translation. The English segment: {line}"}    
    ]
    outputs = pipe(
        shots + messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1]['content'].replace('\n', ' ').strip())