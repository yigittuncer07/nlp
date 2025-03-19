from transformers import pipeline, AutoTokenizer
from llama_ner_infer_engine import UnllamaNerInferenceEngine
import time

# Examples: 
examples = [['3.lük', 'maçında', 'Slovenya', 'Millî', 'Basketbol', "Takımı'nı", 'yendikleri', 'maçta', '23', 'sayı', ',', '6', 'ribaund', ',', '2', 'blok', 'istatistikleriyle', 'oynamış', 've', '12', 'faul', 'yaptırmıştır', '.'], ["'", "''", 'Denizlispor', "''", "'"], ['Hami', 'Mandıralı', '36', ',', 'Orhan', 'Çıkırıkçı', '46', ',', '48', ',', 'Arçil', 'Arveladze', '70'], ['San', 'Antonio', 'Spurs', '(', "Milwaukee'den", ')'], ['Divandere', '(', 'Dîwandere', ')'], ['Büyük', 'Ermenistan', 'kurma', 'girişimleri', 'sona', 'ermiştir', '.'], ['YÖNLENDİRME', '2010-11', '1.', 'Lig'], ['David', 'Ferrer', '(', '16', ')'], ['Hollywood', 'için', 'senaryo', 'yazmaktadır', 've', 'yazdıkları', 'çok', 'satılmaktadır', '.'], ['Türk', 'Silahlı', "Kuvvetleri'nde", 'normal', 'şartlar', 'altında', 'görev', 'süresi', '4', 'yıldır', '.']]
# words = "Deli şapkacı Ayşeye şarkıyı duyup duymadığını sordu, Ayşe ise benzerini duyduğunu söyledi. \"Devam ettiğini biliyorsun değil mi?\" diyerek devam etti şapkacı...".split()

id2label = { 0: "B-LOC", 1: "B-ORG", 2: "B-PER", 3: "I-LOC", 4: "I-ORG", 5: "I-PER", 6: "O" }

MODEL_NAME = 'turkishnlp/UNllama3.2-3b-instruct-ner-wikiann'

# Named entity recognition pipeline, passing in a specific model and tokenizer
engine = UnllamaNerInferenceEngine(MODEL_NAME, id2label=id2label)


start = time.time()
for example in examples:
    labels = engine.infer_labels(example)
    for word, label in zip(example, labels):
        print(f"{word}: {label}")
end = time.time()

print(f"Elapsed time: {end-start} seconds, average time: {(end-start)/len(examples)} seconds")

