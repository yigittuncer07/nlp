from flask import Flask, request, render_template, jsonify
import sys
import colorsys
sys.path.append("..") # Adds higher directory to python modules path.
from llama_ner_infer_engine import UnllamaNerInferenceEngine

app = Flask(__name__)

id2label = { 0: "B-LOC", 1: "B-ORG", 2: "B-PER", 3: "I-LOC", 4: "I-ORG", 5: "I-PER", 6: "O" }

def extract_entity_types():
    entity_types = set()
    for label in id2label.values():
        if label != "O":
            entity_type = label.split("-")[1]
            entity_types.add(entity_type)
    return list(entity_types)

def generate_entity_colors(entity_types):
    """Generate visually distinct colors for each entity type with good contrast"""
    entity_colors = {}
    
    num_entities = len(entity_types)
    for i, entity_type in enumerate(entity_types):
        hue = i / num_entities
        saturation = 0.6  # Slightly reduced saturation for better contrast
        value = 0.95      # Higher value (brightness) for better readability
        
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        entity_colors[entity_type] = hex_color
    
    return entity_colors

def generate_label_colors(entity_colors):
    """Generate colors for all B- and I- prefixed labels"""
    label_colors = {}
    
    for label in id2label.values():
        if label == "O":
            label_colors[label] = "#000000"  # Black
        else:
            prefix, entity_type = label.split("-")
            base_color = entity_colors[entity_type]
            
            # Make B- (beginning) slightly darker and I- (inside) slightly lighter
            # But ensure they're still readable
            if prefix == "B":
                label_colors[label] = darken_color(base_color, 0.1)  # Reduced darkening factor
            else:  # I- prefix
                label_colors[label] = lighten_color(base_color, 0.1)  # Reduced lightening factor
    
    return label_colors

def has_good_contrast(hex_color):
    """Check if a background color provides good contrast with black text"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    return luminance > 0.7  # Threshold for good contrast with black text

def darken_color(hex_color, factor=0.1):
    """Darken a hex color by a factor, ensuring good contrast with black text"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    r = max(0, int(r * (1 - factor)))
    g = max(0, int(g * (1 - factor)))
    b = max(0, int(b * (1 - factor)))
    
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    if luminance < 0.7:
        factor = factor * 0.5  # Reduce the darkening effect
        r = max(0, int(int(hex_color[0:2], 16) * (1 - factor)))
        g = max(0, int(int(hex_color[2:4], 16) * (1 - factor)))
        b = max(0, int(int(hex_color[4:6], 16) * (1 - factor)))
    
    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def lighten_color(hex_color, factor=0.1):
    """Lighten a hex color by a factor"""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    # Lighten
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    
    # Convert back to hex
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

@app.route("/", methods=["GET", "POST"])
def index():
    highlighted_text = ""
    input_text = ""
    
    entity_types = extract_entity_types()
    entity_colors = generate_entity_colors(entity_types)
    label_colors = generate_label_colors(entity_colors)
    
    if request.method == "POST":
        input_text = request.form.get("input_text", "").strip()
        tokens = input_text.split()
        labels = ner_engine.infer_labels(tokens)

        highlighted_spans = []
        for token, label in zip(tokens, labels):
            if label == "O":
                highlighted_spans.append(f'<span>{token}</span>')
            else:
                color = label_colors.get(label, "#000000")
                entity_type = label.split("-")[1]  # Extract LOC, ORG, PER, etc.
                
                text_color = "#000000"  # Default: black text
                if not has_good_contrast(color):
                    text_color = "#FFFFFF"  # White text for dark backgrounds
                
                highlighted_spans.append(
                    f'<span class="entity {label}" style="background-color: {color}; color: {text_color};" '
                    f'title="{label}">{token}</span>'
                )
        
        highlighted_text = " ".join(highlighted_spans)

    # Pass color information to the template
    return render_template(
        "index.html", 
        input_text=input_text, 
        highlighted_text=highlighted_text,
        entity_types=entity_types,
        entity_colors=entity_colors
    )

@app.route("/api/colors", methods=["GET"])
def get_colors():
    entity_types = extract_entity_types()
    entity_colors = generate_entity_colors(entity_types)
    label_colors = generate_label_colors(entity_colors)
    
    return jsonify({
        "entity_colors": entity_colors,
        "label_colors": label_colors
    })

if __name__ == "__main__":
    MODEL_PATH="turkishnlp/UNllama3.2-1b-instruct-ner-wikiann"
    global ner_engine
    ner_engine = UnllamaNerInferenceEngine(MODEL_PATH, id2label)
    
    app.run(debug=True)