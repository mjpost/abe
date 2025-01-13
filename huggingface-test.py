from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

en_text = "This is a test."
long_en_text = "This is a significantly longer text."


model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

tokenizer.src_lang = "en"


encoded_en = tokenizer(en_text, return_tensors="pt")
outputs = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("fr"), output_scores=True, output_hidden_states=True, return_dict_in_generate=True, num_beams=1)

print("Translation of first sentence:")
print(tokenizer.decode(outputs.sequences[0], skip_special_tokens=True))
print("First ten probabilities of the first sequence for each token without padding")
for tok in outputs.scores:
    print(" ".join([str(_.item()) for _ in tok[0, :10]]))


nocache_outputs = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("fr"), output_scores=True, output_hidden_states=True, return_dict_in_generate=True, num_beams=1, use_cache=False)

print("\n\nTranslation of first sentence:")
print(tokenizer.decode(nocache_outputs.sequences[0], skip_special_tokens=True))
print("First ten probabilities of the first sequence for each token without caching or padding")
for tok in nocache_outputs.scores:
    print(" ".join([str(_.item()) for _ in tok[0, :10]]))

batch_en = tokenizer([en_text, long_en_text], return_tensors="pt", padding=True)
long_outputs = model.generate(**batch_en, forced_bos_token_id=tokenizer.get_lang_id("fr"), output_scores=True, output_hidden_states=True, return_dict_in_generate=True, num_beams=1)

print("\n\nTranslation of first sentence:")
print(tokenizer.decode(long_outputs.sequences[0], skip_special_tokens=True))
print("First ten probabilities of the first sequence for each token with padding")
for tok in long_outputs.scores:
    print(" ".join([str(_.item()) for _ in tok[0, :10]]))

