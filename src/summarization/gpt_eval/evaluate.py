#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from tqdm import tqdm

# Azure setup
api_key = os.getenv("AZURE_INFERENCE_CREDENTIAL", '')
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

client = ChatCompletionsClient(
    endpoint='https://azure-openai-verius.openai.azure.com/openai/deployments/gpt-4o',
    credential=AzureKeyCredential(api_key)
)

# GPT prompt
EVAL_PROMPT_TEMPLATE = """
Küçük dil modelleri tarafından üretilen özetleri değerlendiren bir değerlendiricisin. Verilen orijinal metin ve özet göz önünde bulundurularak, özetin orijinal metne uyumu, açıklığı, tutarlılığı ve genel uyumu temel alınarak 1 ile 5 arasında bir puan ver.

**Değerlendirme Kriterleri:**

**5 - Mükemmel:**  
Açık, akıcı ve çoğunlukla doğru. Orijinal metnin temel fikrini iyi bir şekilde yansıtıyor. Önemli hata veya gariplik yok.

**4 - İyi:**  
Genel olarak açık ve ilgili. Bazı noktaları atlamış veya küçük hatalar içerebilir, ancak tutarlı ve anlaşılır.

**3 - Orta:**  
Orijinal fikrin kısmen yakalanması. Belirsiz, tekrarlı veya biraz konu dışı olabilir; ancak yine de okunabilir ve yanıltıcı değil.

**2 - Zayıf:**  
Dilbilgisi açısından hatalı, tutarsız veya karışık. Orijinal anlamı önemli ölçüde yanlış yansıtabilir.

**1 - Kabul Edilemez:**  
Saçma, büyük ölçüde orijinal metinden alakasız veya dil veya içerik açısından ciddi şekilde hatalı.

---

### Örnek 1
**Orijinal Metin:**  
YARGITAY İLAMI Taraflar arasındaki 4650 sayılı Kanunla değişik 2942 sayılı Kamulaştırma Kanununun 10. maddesine dayanan kamulaştırma bedelinin tespiti ve kamulaştırılan taşınmazın davacı idare adına tescili davasından dolayı yapılan yargılama sonunda: Davanın kabulüne dair verilen yukarıda gün ve sayıları yazılı hükmün Yargıtay’ca incelenmesi, davacı idare vekilince verilen dilekçe ile istenilmiş olmakla, dosyadaki belgeler okunup uyuşmazlık anlaşıldıktan sonra gereği görüşülüp düşünüldü: - K A R A R – Dava, 4650 sayılı Kanunla değişik 2942 sayılı Kamulaştırma Kanununun 10. maddesine dayanan kamulaştırma bedelinin tespiti ve kamulaştırılan taşınmazın davacı idare adına tescili istemine ilişkindir. Mahkemece davanın kabulüne karar verilmiş, hüküm davacı idarevekilince temyiz edilmiştir. Bilirkişi incelemesi yaptırılmıştır. Alınan rapor hüküm kurmaya elverişli değildir. Şöyle ki; Farklı kamulaştırma kapsamında aynı bölgeden (Bucakkışla Köyü) gelen dosyalarda Karaman 1. Asliye Hukuk Mahkemesi’nin 2013/596 Esas 2014/172 Karar sayılı dosyası 18 HD’nin 2015/13372 Esas - 2015/13563 sayılı kararı ile Mersin, Tarsus, Ermenek, Erdemli, Adana, Silifke ve Mut Gıda, Tarım ve Hayvancılık Müdürlüklerinden getirtilen nar verilerinin ortalamasının alınması suretiyle kamulaştırma bedelinin belirlenmesi gerekçesiyle bozulmuştur. Somut olayda kamulaştırma belgeleri 19.06.2014 tarihinde mahkemeye verilmiş olduğundan değerlendirmenin 2014 yılına ait yukarıda sözü edilen çevre il ve ilçelerin nar verilerinin ortalamasının alınması suretiyle 2942 sayılı Yasanın 4650 sayılı Yasayla değişik 11. maddesinin birinci fıkrasının (f) bendi uyarınca dava konusu taşınmazın kamulaştırma bedelinin resmi verilere ve gerçeğe uygun biçimde yöntemince tespiti için bilirkişi kurulundan ek rapor alınması ve oluşacak sonuca göre karar verilmesi gerektiği düşünülmeden (gerçeğe uygun ve inandırıcı bulunmayan) bilirkişi kurulunca salt 2013 yılı Karaman Gıda, Tarım ve Hayvancılık İl Müdürlüğü verilerini değerlendirmeye alan rapora itibarla karar verilmesi, Doğru görülmemiştir. Davacı idare vekilinin temyiz itirazları yerinde olduğundan hükmün açıklanan nedenlerle H.U.M.K.nun 428. maddesi gereğince BOZULMASINA, 29/01/2018 gününde oybirliğiyle karar verildi.

**Özet:**  
Dava konusu taşınmazın kamulaştırma bedelinin daha önce farklı kamulaştırma kapsamında aynı bölgeden gelen dosyalarda belirtilen illere ait tarım arazilerinde yetiştiği belirtilen nar verilerinin dava açıldığı tarihteki ortalamasının alınması suretiyle resmi verilere ve gerçeğe uygun biçimde yöntemince tespiti gerektiği gözetilmelidir.

**Değerlendirme:**  
5

---

### Örnek 2
**Orijinal Metin:**  
TÜRK MİLLETİ ADINA YARGITAY İLAMI Mahalli mahkemece verilen hükümler temyiz edilmekle dosya incelenerek gereği düşünüldü: Dairemizin 2020/3762 Esas sırasında kayıtlı dava dosyası ile temyize konu bu dava dosyası arasında sanıklar ve suçlar yönünden fiili ve hukuki bağlantı olduğundan birlikte ele alınarak yapılan incelemede; 1-Sanıklar hakkında tefecilik suçundan kurulan mahkumiyet hükümlerine yönelik temyiz itirazlarının incelenmesinde; Yapılan yargılamaya, toplanıp karar yerinde gösterilen delillere, mahkemenin soruşturma sonuçlarına uygun olarak oluşan kanaat ve takdirine, incelenen dosya içeriğine göre yerinde görülmeyen sair temyiz itirazlarının reddine, Telekomünikasyon ekipman parçaları satışı yapan sanıkların belirli bir faiz karşılığında muhtelif kişilerin nakit ihtiyacını karşılamak veya kredi kartı borçlarını ötelemek amacıyla POS cihazlarından ödünç para vererek tefecilik suçunu işledikleri iddia ve kabul edilen olayda, UYAP kayıtlarına göre; sanıklar hakkında 10/02/2014 tarihli iddianameyle açılan kamu davalarında, 2008 yılında işledikleri iddia edilen tefecilik suçundan Ankara 3. Asliye Ceza Mahkemesince 12/05/2015 tarih ve 2014/137 Esas, 2015/403 sayılı Karar ile mahkumiyetlerine karar verildiği, keza temyize konu bu dosyadaki suç tarihinin 2009 ve 2010 yılları, iddianame tarihinin ise 15/01/2016 olması ve sanıklar hakkında ilk derece mahkemeleri ile istinaf mahkemelerinde aynı suçtan davalar bulunması karşısında, dosyalar arasında sanıklar yönünden hukuki ve fiili irtibat bulunduğu, sanıkların eylemlerinin kül halinde zincirleme tefecilik suçunu oluşturabileceği gözetilip, anılan dosyaların getirtilerek incelenmesinden, mümkünse dosyaların birleştirilmesinden, kesinleşmiş ise onaylı suretlerinin getirtilmesi suretiyle iddianame ve suç tarihlerine göre hukuki kesintinin gerçekleşip gerçekleşmediğinin, suçun teselsül edip etmediğinin, zincirleme şekilde işlenmiş olması durumunda mahsup hükümlerinin uygulanma imkanı olup olmadığının tartışılmasından sonra hasıl olacak sonuca göre sanıkların hukuki durumlarının takdir ve tayin edilmesi gerektiği gözetilmeden eksik inceleme ile yazılı şekilde hükümler kurulması, 2-Sanıklar hakkında 5464 sayılı Kanuna muhalefet suçundan kurulan mahkumiyet hükümlerine yönelik temyiz itirazlarının incelenmesinde ise; Suç tarihi itibariyle yürürlükte bulunan TCK’nın 241. maddesi ile 5464 sayılı Banka Kartları ve Kredi Kartları Kanununun 36. maddesinde aynı tür ve miktarda cezalar öngörülmesi nedeniyle hangi yasa ile uygulama yapılması gerekeceği sorununa ilişkin olarak; POS tefeciliği olayında, her ne kadar görünürde bir satım akdi mevcut olsa ve suçun işlenmesinde kredi kartı araç olarak kullanılsa da, tarafların gerçek niyeti bir faiz anlaşması yapmaktan ibarettir. Üye işyeri sahibi olan fail, kart hamili ile yapmış olduğu faiz anlaşması üzerine işyerinde kurulu POS cihazı üzerinden kart hamilinin kartından -faiz ve anlaşmaya konu ödünç para miktarının toplamından oluşan- bedeli çekerek alacağını teminat altına almakta, sonra çektiği tutardan daha azını (anlaşmaya konu ödünç para miktarını) kart hamiline nakit olarak ödemektedir. Ödünç paranın verilmesi, görünürdeki muvazaalı bir satım akdine dayanmaktadır. Buradaki muvazaa, nispi muvazaa olup; 6098 sayılı Türk Borçlar Kanununun 19. maddesi uyarınca nispi muvazaa hallerinde görünürdeki işlem, tarafların gerçek iradelerini yansıtmadığından geçersiz olacak, tarafların gerçek iradelerini yansıtan alttaki gizli işlem hukuki sonuç doğuracaktır. POS tefeciliğinde tarafların gerçek iradelerini (kastlarını) yansıtmayan görünürdeki satım işlemi geçersiz olmakla birlikte temel de gerçekleştirilmek istedikleri gizli işlem (karz akdi/ödünç sözleşmesi) varlığını muhafaza edecektir. Bu açıklamalar ışığında olay değerlendirildiğinde; POS tefeciliğinde failin kastı, tefecilik suretiyle yarar sağlamaya dönük olup, amaç suç tefeciliktir. Fail, amaçladığı bu suçu işleme yolunda birden fazla hareket gerçekleştirmekte ve bu hareketlerden alacağını teminatlı hale getirmeye dönük bir kısım hareketlerle 5464 sayılı Kanunun 36. maddesinde tanımlanan suçu da işlemekte ise de; söz konusu birden fazla hareket, hukuksal anlamda “tek bir fiili” oluşturmaktadır. Yargıtay Ceza Genel Kurulunun 06/07/2010 tarih ve 2010/8-51 Esas, 2010/162 sayılı Kararında vurgulandığı üzere “TCK’nın 44. maddesi ile kanun koyucu ‘erime sistemini’ benimsemiş olup”, POS tefeciliğinde failin suç yolunda gerçekleştirdiği bir kısım hareketlerle işlediği 5464 sayılı Kanunun 36. maddesine muhalefet suçu, kastının dönük olduğu tefecilik fiilindeki teklik nedeniyle, bu fiilin içinde erimektedir. Bu halde sanıklar hakkında işlemeyi amaçladıkları, diğer bir ifade ile kastlarının dönük olduğu tefecilik suçundan uygulama yapılmalıdır. Kaldı ki Türk Borçlar Kanunu hükümleri de nazara alındığında, maddi gerçeği hedefleyen Ceza Hukukunun, eylemin nitelendirilmesinde görünürdeki işleme değil, tarafların nihai olarak gerçekleştirmek istedikleri (kast) gizli işleme (ödünç sözleşmesi) göre sonuca gidilmelidir. Aktarılan bu açıklama ve değerlendirmeler ışığında somut olayda sanıkların tefecilik suretiyle kazanç sağlamaya yönelik kastları ve atılı suçlara ilişkin eylemlerin korudukları hukuki yararlar dikkate alındığında hukuksal anlamda fiilin sadece tefecilik suçuna vücut vereceği gözetilerek, 5464 sayılı Kanunun 36. maddesinde düzenlenen suç yönünden ceza verilmesine yer olmadığına dair kararlar verilmesi gerekirken, yanılgılı değerlendirme ve oluşa uygun düşmeyen gerekçelerle yazılı şekilde hükümler kurulması, 3-Kabule göre de; Sanıklar hakkında 5237 sayılı Türk Ceza Kanununun 62. maddesi uyarınca, sanıkların geçmişi, sosyal ilişkileri, fiilden sonraki ve yargılama süresindeki davranışları, cezanın failin geleceği üzerindeki etkileri gibi hususları içeren takdiri indirim nedenlerinin varlığının tartışılıp karar yerinde gösterilmemesi, Anayasa Mahkemesinin 08/10/2015 tarih ve E. 2014/140;  K. 2015/85 sayılı iptal Kararının 24/11/2015 tarih ve 29542 sayılı Resmi Gazete’de yayımlanarak yürürlüğe girmiş olması nedeniyle TCK’nın 53. maddesiyle ilgili olarak yeniden değerlendirme yapılmasında zorunluluk bulunması, Kanuna aykırı, sanıklar müdafii ile o yer Cumhuriyet savcısının temyiz itirazları bu nedenlerle yerinde görülmüş olduğundan, 5320 sayılı Kanunun 8/1. maddesi de gözetilerek 1412 sayılı Ceza Muhakemeleri Kanununun 321 ve 326/son maddeleri uyarınca hükümlerin BOZULMASINA, 02/02/2021 tarihinde oy birliğiyle karar verildi.

**Özet:**  
Telekomünikasyon ekipman parçaları satışı yapan sanıkların belirli bir faiz karşılığında muhtelif kişilerin nakit ihtiyacını karşılamak veya kredi kartı borçlarını ötelemek amacıyla POS cihazlarından ödünç para vererek tefecilik suçunu işledikleri iddia ve kabul edilen olayda; sanıklar hakkında daha önce iddianame tarihinden öncesini kapsayan tefecilik eylemi nedeniyle verilmiş bir mahkumiyet kararı yanında ilk derece ve istinaf mahkemelerinde aynı suçtan açılmış davalar bulunması karşısında, sanıkların eylemlerinin kül halinde zincirleme tefecilik suçunu oluşturabileceği dikkate alınarak anılan dosyalar getirtilip incelendikten, mümkünse birleştirildikten, kesinleşmiş ise onaylı suretleri celp edilerek iddianame ve suç tarihlerine göre hukuku kesintisinin gerçekleşip gerçekleşmediği, zincirleme suç hükümlerinin ve buna bağlı olarak mahsubun uygulanma imkanı olup olmadığının tartışılması yanında, POS tefeciliği eyleminde failin kastı tefecilik suretiyle yarar sağlamaya dönük olduğundan sanıklar hakkında tefecilik suçundan uygulama yapılırken, 5464 sayılı Kanunun 36. maddesinde düzenlenen suç yönünden ceza verilmesine yer olmadığı kararı verilmesi gerektiği gözetilmelidir.

**Değerlendirme:**  
5

---

### Değerlendirilmesi gereken örnek:

**Orijinal Metin:**  
{}

**Özet:**  
{}

Sadece 1 ile 5 arasında tek bir sayı döndür. Açıklama veya yorum ekleme.
**Değerlendirme:**  
"""

# Paths
input_dir = Path("inferences")
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# Iterate over files
for file_path in input_dir.glob("*.json"):
    with open(file_path) as f:
        items = json.load(f)

    updated_items = []

    for item in tqdm(items, desc=f"Processing {file_path.name}"):
        prompt_text = EVAL_PROMPT_TEMPLATE.format(item["input"], item["model_response"])

        payload = {
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 10,
            "temperature": 0,
            "top_p": 1,
            "stop": []
        }

        try:
            response = client.complete(payload)
            score_str = response.choices[0].message.content.strip()
            print(f"Response: {score_str}")
            item["gpt_score"] = int(score_str)
        except Exception as e:
            print(f"Error on item: {e}")
            item["gpt_score"] = None

        updated_items.append(item)

    out_path = output_dir / file_path.name
    with open(out_path, "w") as f:
        json.dump(updated_items, f, indent=4, ensure_ascii=False)

    print(f"Saved to {out_path}")
