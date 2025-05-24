import colorsys

def extract_entity_types(id2label):
    return list({label.split("-")[1] for label in id2label.values() if label != "O"})

def generate_entity_colors(entity_types):
    colors = {}
    for i, entity in enumerate(entity_types):
        hue = i / len(entity_types)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.95)
        colors[entity] = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
    return colors

def has_good_contrast(hex_color):
    r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    luminance = (0.299*r + 0.587*g + 0.114*b)/255
    return luminance > 0.7

def darken_color(hex_color, factor=0.1):
    r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    r, g, b = [max(0, int(c * (1 - factor))) for c in (r, g, b)]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def lighten_color(hex_color, factor=0.1):
    r, g, b = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]
    r = min(255, int(r + (255 - r) * factor))
    g = min(255, int(g + (255 - g) * factor))
    b = min(255, int(b + (255 - b) * factor))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def generate_label_colors(id2label, entity_colors):
    label_colors = {}
    for label in id2label.values():
        if label == "O":
            label_colors[label] = "#000000"
        else:
            prefix, entity = label.split("-")
            base = entity_colors[entity]
            label_colors[label] = darken_color(base) if prefix == "B" else lighten_color(base)
    return label_colors

