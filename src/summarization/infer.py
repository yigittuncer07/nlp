import torch
from transformers import pipeline
user="""Bir köylü kadın ikinci defa evlendi. Mutfakta birinci kocasını
düşündü. Birinci kocası şimdiki kocasından daya iyiydi. Bu arada
bir dilenci kapıyı çalmadan içeri girdi. Köylü kadından bir şeyler
vermesini rica etti. Çünkü dilenci çok açtı. Köylü kadın dilenciye:
“Nereden geliyorsun?” diye sordu. Dilenci “Çanakkale’den
geliyorum” diye cevap verdi. Köylü kadın Ça- nakkale’yi duymadı
ve bunu cennet gibi anladı. Kadın: “Birinci kocam geçen yıl öldü.
Cennette onu gördün mu?” diye sordu. Dilenci: “Tabii, sevgili bayan,
kocanızı taniyorum. Kocanız orada gömleksiz, şapkasız büyük bir
elbiseyle dolaşıyor” dedi. Köylü kadın: Ah! Benim zavallı kocam
dedi ve dilenciye: “Cennete ne zaman dönüyorsun?” diye sordu.
Dilenci: “Yarın hazırlanıp on dört gün sonra cennete dönecegim.”
dedi. Köylü kadın: “Birinci kocama bir şeyler götürür müsün?” diye
sordu. Dilenci; “Tabii” diye cevap verdi ve “Deli bir kadın! Belki
Ölü kocasına para ve elbise gönderir"""
system="Sen bir metin özetleme asistanısın. Sana verilen metinleri analiz et ve önemli noktaları içeren kısa özetler yaz"
# model_id = "turkishnlp/llama3.2-1b-sum-finetune-final"
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1]) 