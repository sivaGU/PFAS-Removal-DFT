"""Figure style constants"""

BLUE = "#1f77b4"
ORANGE = "#B35400"
GREEN = "#176B43"
PURPLE = "#5D4E9C"
GREY = "#555555"
FONT = "DejaVu Sans"
EXPORT_DPI = 600


def _linear_channel(value):
    value /= 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def luminance(rgb):
    channels = [_linear_channel(round(float(v) * 255.0)) for v in rgb[:3]]
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def readable_text_color(rgba):
    return "black" if luminance(rgba[:3]) >= 0.179 else "white"
