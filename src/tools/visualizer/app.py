from flask import Flask, request, render_template
from tools.llama_ner_infer_engine import  UnllamaNerInferenceEngine
app = Flask(__name__)

label_colors = {}
id2label = {
    0: "O",
    1: "B-CARDINAL",
    2: "B-DATE",
    3: "B-EVENT",
    4: "B-FAC",
    5: "B-GPE",
    6: "B-LANGUAGE",
    7: "B-LAW",
    8: "B-LOC",
    9: "B-MONEY",
    10: "B-NORP",
    11: "B-ORDINAL",
    12: "B-ORG",
    13: "B-PERCENT",
    14: "B-PERSON",
    15: "B-PRODUCT",
    16: "B-QUANTITY",
    17: "B-TIME",
    18: "B-TITLE",
    19: "B-WORK_OF_ART",
    20: "I-CARDINAL",
    21: "I-DATE",
    22: "I-EVENT",
    23: "I-FAC",
    24: "I-GPE",
    25: "I-LANGUAGE",
    26: "I-LAW",
    27: "I-LOC",
    28: "I-MONEY",
    29: "I-NORP",
    30: "I-ORDINAL",
    31: "I-ORG",
    32: "I-PERCENT",
    33: "I-PERSON",
    34: "I-PRODUCT",
    35: "I-QUANTITY",
    36: "I-TIME",
    37: "I-TITLE",
    38: "I-WORK_OF_ART"
}
# Define a color scheme for the labels

@app.route("/", methods=["GET", "POST"])
def index():
    highlighted_text = ""
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        tokens = input_text.split()
        labels = ner_engine.infer_labels(tokens)

        # Build highlighted text
        highlighted_text = []
        for token, label in zip(tokens, labels):
            color = label_colors.get(label, "#000000")  # Default to black if label not found
            highlighted_text.append(f'<span style="color: {color};">{token}</span>')
        highlighted_text = " ".join(highlighted_text)

    return render_template("index.html", input_text=input_text, highlighted_text=highlighted_text)


if __name__ == "__main__":
    MODEL_PATH="turkishnlp/UNllama3.2-1b-instruct-ner"
    global ner_engine
    ner_engine = UnllamaNerInferenceEngine(MODEL_PATH, id2label)


    prefix_colors = {
        "B-": "#f4a261",  # Orange for beginning entities
        "I-": "#2a9d8f",  # Teal for inside entities
        "O": "#000000",   # Black for non-entities
    }

    # Generate label_colors from id2label
    for label_id, label in id2label.items():
        if label == "O":
            label_colors[label] = prefix_colors["O"]
        elif label.startswith("B-"):
            label_colors[label] = prefix_colors["B-"]
        elif label.startswith("I-"):
            label_colors[label] = prefix_colors["I-"]
        else:
            label_colors[label] = "#FFFFFF"  # Default color for unknown labels


    app.run(debug=True)
