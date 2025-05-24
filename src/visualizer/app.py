from flask import Flask, request, render_template, jsonify, session
from config import id2label, MODEL_PATH
from llama_ner_infer_engine import UnllamaNerInferenceEngine
from ner_highlighter.color_utils import *
from ner_highlighter.highlighter import highlight_text
import requests

app = Flask(__name__)
app.secret_key = 'deftones12'  

ner_engine = UnllamaNerInferenceEngine(MODEL_PATH, id2label)

def query_summarizer(text):
    payload = {
        "system_prompt": "Sen deneyimli bir hukukçusun ve görevin, verilen içtihat metinlerini sadece karar metnine dayanarak özetlemektir. Özet, avukatların dava sürecinde kullanımı için teknik ve tarafsız bir şekilde hazırlanmalıdır. Özetin kişisel yorum, değerlendirme veya çıkarım içermemelidir. Özette yargıtay kararında geçen dava tarihlerini, miktarları, karar tarihini, kanun ve üst mahkeme atıflarını özet içinde birebir muhafaza et. Sadece karar metnine sadık kal, ek bilgi ekleme. Cevap olarak sadece paragraf şeklindeki özetini ver.",
        "user_prompt": text,
        "temperature": 1.0,
        "max_tokens": 512,
        "seed": 48  # Added fixed seed
    }
    try:
        response = requests.post("http://localhost:5005/infer", json=payload)
        response.raise_for_status()
        full_text = response.json().get("response", "[Boş Yanıt]")
        
        # Split at the word 'model' and return only the second part
        if "model" in full_text:
            return full_text.split("model", 1)[-1].strip()
        else:
            return full_text.strip()
    except Exception as e:
        return f"[Hata: {e}]"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        submit_button = request.form.get("submit_button")
        if submit_button == "ner":
            session["ner_input"] = request.form.get("ner_input", "").strip()
            session["sum_input"] = request.form.get("sum_input", "").strip() # Keep sum_input
            if session["ner_input"]:
                tokens = session["ner_input"].split()
                labels = ner_engine.infer_labels(tokens)
                entity_types = extract_entity_types(id2label)
                entity_colors = generate_entity_colors(entity_types)
                label_colors = generate_label_colors(id2label, entity_colors)
                session["ner_result"] = highlight_text(tokens, labels, label_colors)
            else:
                session["ner_result"] = ""
            # Ensure sum_result is preserved from session if it exists
            if "sum_result" not in session:
                session["sum_result"] = ""

        elif submit_button == "sum":
            session["sum_input"] = request.form.get("sum_input", "").strip()
            session["ner_input"] = request.form.get("ner_input", "").strip() # Keep ner_input
            if session["sum_input"]:
                session["sum_result"] = query_summarizer(session["sum_input"])
            else:
                session["sum_result"] = ""
            # Ensure ner_result is preserved from session if it exists
            if "ner_result" not in session:
                session["ner_result"] = ""
    
    # Retrieve values from session, defaulting to empty strings if not found
    ner_input = session.get("ner_input", "")
    sum_input = session.get("sum_input", "")
    ner_result = session.get("ner_result", "")
    sum_result = session.get("sum_result", "")

    entity_types = extract_entity_types(id2label)
    entity_colors = generate_entity_colors(entity_types)
    # label_colors = generate_label_colors(id2label, entity_colors) # This is already generated when ner is processed

    return render_template("index.html",
                           ner_input=ner_input,
                           sum_input=sum_input,
                           ner_result=ner_result,
                           sum_result=sum_result,
                           entity_types=entity_types,
                           entity_colors=entity_colors)

@app.route("/clear_outputs", methods=["POST"])
def clear_outputs():
    session.pop("ner_result", None)
    session.pop("sum_result", None)
    # Optionally, clear inputs too if desired
    # session.pop("ner_input", None)
    # session.pop("sum_input", None)
    return jsonify(status="success")

@app.route("/api/colors")
def get_colors():
    entity_types = extract_entity_types(id2label)
    entity_colors = generate_entity_colors(entity_types)
    label_colors = generate_label_colors(id2label, entity_colors)
    return jsonify({"entity_colors": entity_colors, "label_colors": label_colors})

if __name__ == "__main__":
    app.run(debug=True, port=8001, use_reloader=False)

