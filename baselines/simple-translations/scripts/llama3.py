# https://huggingface.co/docs/transformers/en/model_doc/nllb

import sys
import torch
from transformers import pipeline
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = sys.argv[1]
target = sys.argv[2]

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

shots = {

    "cs": [
        {"role": "user", "content": "Translate the following text from English into Czech. The English Segment: The only way to restore service was to do a AC power reset."},
        {"role": "assistant", "content": "Jediný způsob, jak obnovit službu, bylo provést reset napájení."},

        {"role": "user", "content": "Translate the following text from English into Czech. The English Segment: Then we have to come at a quarter past eight already, so that we have the 30 minutes extra to make sure that everything is running and chase people."},
        {"role": "assistant", "content": "Pak musíme přijít už ve čtvrt na devět, abychom měli 30 minut navíc na to ujistit se, že všechno běží, a abychom nahnali lidi."},

        {"role": "user", "content": "Translate the following text from English into Czech. The English Segment: He added that the decision to withdraw the appeal was more \"to prevent the politics of Hindutva [Hindu nationalism] from being ridiculed in the face of severe criticism from all quarters.\""},
        {"role": "assistant", "content": "Dodal, že rozhodnutí stáhnout výzvu bylo spíše proto, „aby se politika hindutvy (hinduistického nacionalismu) nezesměšnila tváří v tvář tvrdé kritice ze všech stran“."}
    ],

    "de": [
        {"role": "user", "content": "Translate the following text from English into German. The English Segment: White nationalists, gun fetishists, pro-dictator right wing.. there's the danger"},
        {"role": "assistant", "content": "Weiße Nationalisten, Waffenfetischisten, Diktator-Freundliche vom rechten Flügel … da liegt die Gefahr. Deportiert die weißen Rassisten."},

        {"role": "user", "content": "Translate the following text from English into German. The English Segment: It's 2023 and I still see people with iPhones actively avoid using Apple Maps."},
        {"role": "assistant", "content": "Es ist 2023 und ich sehe immer noch Menschen mit iPhones, die aktiv die Verwendung von Apple Maps vermeiden."},

        {"role": "user", "content": "Translate the following text from English into German. The English Segment: #CNN needs to just close down. Yesterday's fiasco with #MangoMoron was an outrageous fiasco. Who at that pathetic, dying network was responsible for vetting the so-called \"independent voter\" in the audience? Which cable news \"executive\" made the call to give the fat, lying criminal this much air-time?"},
        {"role": "assistant", "content": "#CNN muss einfach den Laden dichtmachen. Das Fiasko von gestern mit #MangoMoron war einfach ungeheuerlich. Wer war bei diesem jämmerlichen, im Sterben liegenden Sender eigentlich für die Überprüfung der sogenannten „unabhängigen Wähler“ im Publikum verantwortlich? Welcher Kabelnachrichten-„Verantwortliche“ hat denn die Entscheidung getroffen, dem fetten, lügenden Kriminellen so viel Sendezeit zu geben?"}
    ],

    "es": [
        {"role": "user", "content": "Translate the following text from English into Spanish. The English Segment: In June, the Commission published the results of a public consultation on the proposals which found broad support for calling the assembly a Welsh Parliament."},
        {"role": "assistant", "content": "En junio, la Comisión publicó los resultados de una consulta pública sobre las propuestas, en donde se obtuvo un amplio apoyo para llamar a la asamblea un Parlamento de Gales."},

        {"role": "user", "content": "Translate the following text from English into Spanish. The English Segment: Waters' statement quickly drew criticism online, including from former White House press secretary Ari Fleischer."},
        {"role": "assistant", "content": "La declaración de Walters provocó rápidamente críticas en Internet, incluyendo una del anterior secretario de prensa de la Casa Blanca Ari Fleischer."},

        {"role": "user", "content": "Translate the following text from English into Spanish. The English Segment: It was a third Elite League defeat of the season for Adam Keefe's men, who had come from behind to beat Dundee 2-1 in Belfast on Friday night."},
        {"role": "assistant", "content": "Fue la tercera derrota de la temporada de la Elite League para el equipo de Adam Keefe, quienes tuvieron que jugar desde una posición en desventaja para vencer a Dundee 2 a 1 en Belfast el viernes en la noche."}
    ],

    "uk": [
        {"role": "user", "content": "Translate the following text from English into Ukrainian. The English Segment: Will a community blossom around the user-generated content piece of the game?"},
        {"role": "assistant", "content": "Чи буде привабливим частина гри, де користувачі самі створюють свій контент?"},

        {"role": "user", "content": "Translate the following text from English into Ukrainian. The English Segment: That takes effort."},
        {"role": "assistant", "content": "Це вимагає зусиль."},

        {"role": "user", "content": "Translate the following text from English into Ukrainian. The English Segment: Shortly after, Balenciaga dropped photos for its spring 2023 campaign that featured a page from a 2008 Supreme Court case involving \"virtual child pornography\" in the background."},
        {"role": "assistant", "content": "Невдовзі після цього Balenciaga опублікував фотографії для своєї рекламної кампанії навесні 2023 року, на задньому плані яких була сторінка зі справи у Верховному суді 2008 року про «віртуальну дитячу порнографію»."}
    ],

    "ru": [
        {"role": "user", "content": "Translate the following text from English into Russian. The English Segment: The hacked up version of Jedi Knight was crashing because it was calling a function off the end of a vtable."},
        {"role": "assistant", "content": "Взломанная версия Jedi Knight вылетала из-за того, что вызывала функцию с конца виртуальной таблицы."},

        {"role": "user", "content": "Translate the following text from English into Russian. The English Segment: Who at that pathetic, dying network was responsible for vetting the so-called \"independent voters\" in the audience?"},
        {"role": "assistant", "content": "Кто на этом жалком умирающем телеканале отвечал за проверку так называемых «независимых избирателей» среди зрителей?"},

        {"role": "user", "content": "Translate the following text from English into Russian. The English Segment: The suspect is charged with two counts of felony first-degree murder and two counts of felony attempted murder, Chicago Police superintendent David Brown announced."},
        {"role": "assistant", "content": "Подозреваемый обвиняется по двум пунктам по делу об убийстве первой степени и двум пунктам по делу о покушении на убийство, сообщил Дэвид Браун, начальник полиции Чикаго."}
    ]
}

prefixes = {
    "cs": "Translate the following text from English into Czech. The English Segment: ",
    "de": "Translate the following text from English into German. The English Segment: ",
    "es": "Translate the following text from English into Spanish. The English Segment: ",
    "uk": "Translate the following text from English into Ukrainian. The English Segment: ",
    "ru": "Translate the following text from English into Russian. The English Segment: "
}


# shots = [
#     {"role": "user", "content": "Translate the following text from English into German. The English Segment: White nationalists, gun fetishists, pro-dictator right wing.. there's the danger"},
#     {"role": "assistant", "content": "Weiße Nationalisten, Waffenfetischisten, Diktator-Freundliche vom rechten Flügel … da liegt die Gefahr. Deportiert die weißen Rassisten."},

#     {"role": "user", "content": "Translate the following text from English into German. The English Segment: It's 2023 and I still see people with iPhones actively avoid using Apple Maps."},
#     {"role": "assistant", "content": "Es ist 2023 und ich sehe immer noch Menschen mit iPhones, die aktiv die Verwendung von Apple Maps vermeiden."},

#     {"role": "user", "content": "Translate the following text from English into German. The English Segment: #CNN needs to just close down. Yesterday's fiasco with #MangoMoron was an outrageous fiasco. Who at that pathetic, dying network was responsible for vetting the so-called \"independent voter\" in the audience? Which cable news \"executive\" made the call to give the fat, lying criminal this much air-time?"},
#     {"role": "assistant", "content": "#CNN muss einfach den Laden dichtmachen. Das Fiasko von gestern mit #MangoMoron war einfach ungeheuerlich. Wer war bei diesem jämmerlichen, im Sterben liegenden Sender eigentlich für die Überprüfung der sogenannten „unabhängigen Wähler“ im Publikum verantwortlich? Welcher Kabelnachrichten-„Verantwortliche“ hat denn die Entscheidung getroffen, dem fetten, lügenden Kriminellen so viel Sendezeit zu geben?"}
# ]


for line in sys.stdin:
    line = line.strip()
    messages = [
        {"role": "user", "content": f"{prefixes[target]} {line}"}    
    ]
    outputs = pipe(
        shots[target] + messages,
        max_new_tokens=256,
        num_beams = 5,
    )
    print(outputs[0]["generated_text"][-1]['content'].replace('\n', ' ').strip())