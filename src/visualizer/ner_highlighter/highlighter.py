from .color_utils import has_good_contrast

def highlight_text(tokens, labels, label_colors):
    spans = []
    for token, label in zip(tokens, labels):
        if label == "O":
            spans.append(f"<span>{token}</span>")
        else:
            bg = label_colors[label]
            fg = "#000000" if has_good_contrast(bg) else "#FFFFFF"
            spans.append(
                f'<span class="entity" style="background-color: {bg}; color: {fg};" title="{label}">{token}</span>'
            )
    return " ".join(spans)

