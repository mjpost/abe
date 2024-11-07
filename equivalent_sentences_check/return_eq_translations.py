import json
import os

def load_input(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_translations(output_file_list):
    translations = []
    for output_file in output_file_list:
        with open(output_file, "r") as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            translations.append(lines)
    return translations

english_inputs = load_input("eng_Latn.devtest")
translations = load_translations(["fra_Latn_M2M418.devtest", "fra_Latn_NLLB600.devtest"])

matched_english_inputs = []
for i in range(len(translations[0])):
    if translations[0][i] == translations[1][i]:
        #print("Matched")
        #print(translations[0][i], translations[1][i])
        matched_english_inputs.append(english_inputs[i])

'''
Export matched_english_inputs to a file
'''

with open("matched_english_inputs", "w") as f:
    for line in matched_english_inputs:
        f.write(line + "\n")
    f.close()
    
