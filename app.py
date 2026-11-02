import base64
import hashlib
import io
import json
import os
from pathlib import Path
import uuid

from dotenv import load_dotenv
from openai import OpenAI
from pillow_heif import register_heif_opener
import rawpy
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Stillroom · Photo Editor", page_icon="◫", layout="wide", initial_sidebar_state="expanded")
load_dotenv(Path(__file__).with_name(".env.local"))
register_heif_opener(thumbnails=False)

# Seed curve controls before any widgets or callbacks read from session state.
for _channel in ("master", "red", "green", "blue"):
    for _point, _value in (("shadows", 64), ("midtones", 128), ("highlights", 192)):
        st.session_state.setdefault(f"curve_{_channel}_{_point}", _value)

SUGGESTION_CONTROL_ORDER = [
    "brightness", "contrast", "saturation",
    "tint", "reds", "greens", "blues",
    "dehaze", "denoise",
    "curve_shadows", "curve_midtones", "curve_highlights",
    "rotation",
]
SUGGESTION_LABELS = {
    "brightness": "Brightness", "contrast": "Contrast", "saturation": "Saturation",
    "tint": "Tint", "reds": "Reds", "greens": "Greens", "blues": "Blues",
    "dehaze": "Dehaze", "denoise": "Denoise",
    "curve_shadows": "Curve · Shadows", "curve_midtones": "Curve · Midtones",
    "curve_highlights": "Curve · Highlights", "rotation": "Rotation",
}
SUGGESTION_RANGES = {
    "brightness": (0.0, 2.0, "float"), "contrast": (0.0, 2.0, "float"),
    "saturation": (0.0, 2.0, "float"), "tint": (-100, 100, "int"),
    "reds": (0.0, 2.0, "float"), "greens": (0.0, 2.0, "float"),
    "blues": (0.0, 2.0, "float"), "dehaze": (0, 100, "int"),
    "denoise": (0, 100, "int"), "curve_shadows": (0, 255, "int"),
    "curve_midtones": (0, 255, "int"), "curve_highlights": (0, 255, "int"),
    "rotation": (0, 270, "rotation"),
}
ADVICE_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "control": {"type": "string", "enum": SUGGESTION_CONTROL_ORDER},
                    "value": {"type": "number"},
                    "reason": {"type": "string"},
                },
                "required": ["control", "value", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["message", "suggestions"],
    "additionalProperties": False,
}


def configured_api_key():
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        secret_key = None
    return secret_key or os.environ.get("OPENAI_API_KEY", "")


def normalize_suggestions(items):
    cleaned = []
    seen = set()
    for item in items:
        control = item.get("control")
        value = item.get("value")
        if control not in SUGGESTION_RANGES or control in seen or isinstance(value, bool):
            continue
        try:
            low, high, kind = SUGGESTION_RANGES[control]
            value = max(low, min(high, float(value)))
            if kind == "int":
                value = int(round(value))
            elif kind == "float":
                value = round(value, 2)
            else:
                value = min((0, 90, 180, 270), key=lambda step: abs(step - value))
        except (TypeError, ValueError):
            continue
        cleaned.append({
            "control": control,
            "value": value,
            "reason": str(item.get("reason", "")).strip(),
        })
        seen.add(control)
    return sorted(cleaned, key=lambda item: SUGGESTION_CONTROL_ORDER.index(item["control"]))


def apply_suggested_edits(suggestions, curve_channel, suggestion_id):
    for suggestion in suggestions:
        control = suggestion["control"]
        if control.startswith("curve_"):
            point = control.removeprefix("curve_")
            key = f"curve_{curve_channel}_{point}"
        else:
            key = control
        st.session_state[key] = suggestion["value"]
    st.session_state[f"ai_suggestion_applied_{suggestion_id}"] = True


def ask_photo_advisor(api_key, image, question, history, settings):
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG", quality=86, optimize=True)
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")

    request_input = [{
        "role": "developer",
        "content": (
            "You are a relaxed, helpful photo-editing buddy with a good photographer's eye. Sound casual, "
            "warm, and natural, like you're looking at the photo together. Avoid formal reports, canned praise, "
            "and bare diagnoses such as 'the photo is underexposed.' Instead, point out one specific thing you "
            "notice, then give a clear next move using the editor's actual control names and a modest suggested "
            "amount or direction. Briefly say what that change should improve. For example: 'The sunset colors "
            "are lovely; the foreground is getting a little lost, though. Try lifting Shadows to about 80 and "
            "Brightness just a touch, maybe 1.05, so the foreground opens up without flattening the sky.' "
            "Give one or two useful steps at a time, not a long checklist. If the image already looks good, say "
            "what is working and offer an optional tweak. Match the user's level of detail, use plain language, "
            "and be honest when something is subjective or unclear. Do not claim to have changed the photo. "
            "Return a JSON object with a casual user-facing message and a suggestions array. Each suggestion must "
            "use one real editor control, provide a target value, and briefly explain why. Suggest only useful "
            "changes, and keep the list short. For tone-curve suggestions, use the currently selected curve "
            "channel. Suggest values within these ranges: Brightness/Contrast/Saturation/Reds/Greens/Blues 0–2; "
            "Tint -100–100; Dehaze/Denoise 0–100; curve points 0–255; Rotation 0, 90, 180, or 270."
        ),
    }]
    for item in history[-10:]:
        history_text = item["content"]
        if item.get("suggestions"):
            summary = "; ".join(
                f"{SUGGESTION_LABELS[suggestion['control']]} to {suggestion['value']}"
                for suggestion in item["suggestions"]
            )
            history_text += f"\nSuggested slider changes: {summary}."
        request_input.append({"role": item["role"], "content": history_text})
    request_input.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": f"{question}\n\nCurrent editor settings: {settings}"},
            {"type": "input_image", "image_url": image_data_url, "detail": "high"},
        ],
    })
    response = OpenAI(api_key=api_key).responses.create(
        model="gpt-5.5",
        input=request_input,
        text={
            "format": {
                "type": "json_schema",
                "name": "photo_edit_advice",
                "strict": True,
                "schema": ADVICE_SCHEMA,
            }
        },
        max_output_tokens=500,
    )
    return json.loads(response.output_text)


def apply_color_effects(image, tint, reds, greens, blues, dehaze, denoise):
    """Apply color channel balance plus lightweight haze/noise reduction."""
    red, green, blue = image.split()

    # Tint moves the balance toward magenta (positive) or green (negative).
    magenta = max(0, tint) / 100
    green_tint = max(0, -tint) / 100
    red = red.point(lambda value: max(0, min(255, int(value * reds + magenta * 24))))
    green = green.point(lambda value: max(0, min(255, int(value * greens + green_tint * 24 - magenta * 18))))
    blue = blue.point(lambda value: max(0, min(255, int(value * blues + magenta * 18))))
    image = Image.merge("RGB", (red, green, blue))

    if dehaze:
        dehazed = ImageOps.autocontrast(image, cutoff=1)
        image = Image.blend(image, dehazed, dehaze / 100)
    if denoise:
        # Keep the median kernel odd and cap it to bound work on large photos.
        kernel = 3 if denoise < 45 else 5 if denoise < 80 else 7
        softened = image.filter(ImageFilter.MedianFilter(size=kernel))
        image = Image.blend(image, softened, denoise / 100)
    return image


def tone_curve_lut(shadows, midtones, highlights):
    """Build a smooth-ish piecewise-linear tone curve through five fixed anchors."""
    anchors = [(0, 0), (64, shadows), (128, midtones), (192, highlights), (255, 255)]
    lut = []
    segment = 0
    for value in range(256):
        while segment < len(anchors) - 2 and value > anchors[segment + 1][0]:
            segment += 1
        x0, y0 = anchors[segment]
        x1, y1 = anchors[segment + 1]
        amount = (value - x0) / (x1 - x0)
        lut.append(max(0, min(255, round(y0 + amount * (y1 - y0)))))
    return lut


def apply_tone_curves(image, curves):
    master, red_curve, green_curve, blue_curve = curves
    channels = image.split()
    result = []
    for channel, curve in zip(channels, (red_curve, green_curve, blue_curve)):
        channel = channel.point(master)
        result.append(channel.point(curve))
    return Image.merge("RGB", tuple(result))


def open_uploaded_image(image_bytes, is_raw, half_size=False):
    if is_raw:
        with rawpy.imread(io.BytesIO(image_bytes)) as raw:
            rgb = raw.postprocess(half_size=half_size, use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb)
    return ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")


@st.cache_data(show_spinner=False)
def make_preview(image_bytes, is_raw, brightness, contrast, saturation, tint, reds, greens, blues, dehaze, denoise, curves, rotation):
    """Decode and edit a small preview so slider reruns stay lightweight."""
    image = open_uploaded_image(image_bytes, is_raw, half_size=is_raw)
    image.thumbnail((1800, 1400), Image.Resampling.LANCZOS)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    image = apply_color_effects(image, tint, reds, greens, blues, dehaze, denoise)
    image = apply_tone_curves(image, curves)
    if rotation:
        image = image.rotate(-rotation, expand=True)
    # Quantization gives a compact dominant-color palette with bounded work.
    palette_image = image.quantize(colors=8, method=Image.Quantize.FASTOCTREE)
    color_counts = sorted(palette_image.getcolors(maxcolors=image.width * image.height) or [], reverse=True)
    palette = palette_image.getpalette()
    total = sum(count for count, _ in color_counts) or 1
    colors = []
    for count, index in color_counts[:8]:
        offset = index * 3
        red, green, blue = palette[offset:offset + 3]
        colors.append((f"#{red:02X}{green:02X}{blue:02X}", count / total * 100))
    return image, colors


def make_full_resolution_export(image_bytes, is_raw, brightness, contrast, saturation, tint, reds, greens, blues, dehaze, denoise, curves, rotation):
    image = open_uploaded_image(image_bytes, is_raw)
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    image = ImageEnhance.Color(image).enhance(saturation)
    image = apply_color_effects(image, tint, reds, greens, blues, dehaze, denoise)
    image = apply_tone_curves(image, curves)
    if rotation:
        image = image.rotate(-rotation, expand=True)
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), image.size


def reset_adjustments():
    defaults = {
        "brightness": 1.0, "contrast": 1.0, "saturation": 1.0, "tint": 0,
        "reds": 1.0, "greens": 1.0, "blues": 1.0, "dehaze": 0,
        "denoise": 0, "rotation": 0, "curve_channel": "Master",
    }
    st.session_state.update(defaults)
    for channel in ("master", "red", "green", "blue"):
        for point, value in (("shadows", 64), ("midtones", 128), ("highlights", 192)):
            st.session_state[f"curve_{channel}_{point}"] = value


def guide_popover(label, title, description):
    """A small contextual guide button for an editor area."""
    with st.popover(label, help=f"Learn about {title}"):
        st.markdown(f"**{title}**")
        st.write(description)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
:root { --ink:#e9e8e3; --muted:#92938f; --line:#2b2c29; --panel:#171816; --accent:#d4f06a; }
.stApp { background:#10110f; color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:#171816; border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.3rem; }
.block-container { padding-top:2.2rem; max-width:1500px; }
header[data-testid="stHeader"] { background:transparent; }
.brand { font:500 12px 'DM Mono',monospace; letter-spacing:.14em; color:var(--accent); text-transform:uppercase; }
.hero { display:flex; justify-content:space-between; align-items:end; border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:24px; }
.hero h1 { font-size:clamp(30px,4vw,48px); line-height:1; letter-spacing:-.055em; margin:9px 0 0; font-weight:600; }
.hero p { color:var(--muted); margin:0 0 3px; font-size:14px; }
.eyebrow { font:11px 'DM Mono',monospace; color:var(--muted); letter-spacing:.11em; text-transform:uppercase; }
.canvas-head { display:flex; justify-content:space-between; margin:4px 0 12px; color:var(--muted); font:11px 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stFileUploader"] { border:1px dashed #454740; border-radius:12px; background:#1c1e19; padding:10px; }
[data-testid="stFileUploader"] section { border:0; }
[data-testid="stFileUploader"] section > div { padding:18px 8px; }
[data-testid="stFileUploader"] button { background:var(--accent); color:#151710; border:0; border-radius:6px; font-weight:600; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--accent); border-color:var(--accent); }
.stButton button, .stDownloadButton button { border-radius:7px; font-weight:600; }
.stButton button[kind="primary"] { background:var(--accent); color:#151710; border:0; }
.sidebar-title { font-size:20px; font-weight:600; letter-spacing:-.04em; margin:0 0 4px; }
.sidebar-note { color:var(--muted); font-size:12px; margin-bottom:26px; }
.empty { min-height:60vh; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; border:1px solid var(--line); border-radius:12px; background:radial-gradient(ellipse at center,#1c1e1a 0%,#141512 64%); padding:48px; }
.empty-icon { font-size:36px; color:var(--accent); margin-bottom:18px; }
.empty h2 { font-size:24px; letter-spacing:-.04em; margin:0 0 8px; }
.empty p { max-width:370px; color:var(--muted); line-height:1.65; font-size:14px; margin:0; }
.meta { display:flex; justify-content:space-between; color:var(--muted); font:10px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; margin-top:11px; }
[data-testid="stImage"] img { border-radius:8px; }
.st-key-floating_ai_advice { position:fixed !important; right:20px; bottom:18px; z-index:9999; width:min(400px, calc(100vw - 32px)) !important; max-height:72vh; overflow-y:auto; background:rgba(23,24,22,.97); border:1px solid #454740; border-radius:12px; box-shadow:0 14px 42px #0009; padding:8px 12px; }
.st-key-floating_ai_advice [data-testid="stExpander"] { background:#171816; border:0; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("dark_mode", True):
    st.markdown("""
    <style>
    :root { --ink:#252620; --muted:#676a62; --line:#d4d5cc; --panel:#ffffff; --accent:#8aa92e; }
    .stApp { background:#f5f5f0; color:#252620; }
    [data-testid="stSidebar"] { background:#e9eae3; border-right:1px solid #d4d5cc; }
    [data-testid="stHeader"] { background:rgba(245,245,240,.92); }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] { color:#252620; }
    .hero p, .sidebar-note, .eyebrow, .canvas-head, .meta { color:#676a62; }
    [data-testid="stExpander"], [data-testid="stPopover"] > div,
    [data-baseweb="popover"] > div { background:#fff; color:#252620; border-color:#d4d5cc; }
    [data-testid="stFileUploader"] { background:#f7f7f2; border-color:#bfc1b7; }
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea {
        background:#fff; color:#252620; border-color:#c7c9bf;
    }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
    [data-testid="stPopover"] button { color:#252620; border-color:#c7c9bf; }
    [data-testid="stButton"] button[kind="primary"], [data-testid="stDownloadButton"] button[kind="primary"] {
        background:#d4f06a; color:#202315; border:0;
    }
    .empty { background:radial-gradient(ellipse at center,#fff 0%,#eeefe8 72%); border-color:#d4d5cc; }
    .empty h2 { color:#252620; }
    .empty p { color:#676a62; }
    .st-key-floating_ai_advice { background:rgba(255,255,252,.98); border-color:#c7c9bf; box-shadow:0 14px 42px #20231524; }
    .st-key-floating_ai_advice [data-testid="stExpander"] { background:#fff; }
    table { color:#252620 !important; }
    svg { background:#fff !important; border-color:#d4d5cc !important; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    for channel in ("master", "red", "green", "blue"):
        for point, value in (("shadows", 64), ("midtones", 128), ("highlights", 192)):
            st.session_state.setdefault(f"curve_{channel}_{point}", value)
    st.markdown('<div class="brand">STILLROOM / 01</div><div style="height:14px"></div><div class="sidebar-title">Make it yours.</div><div class="sidebar-note">A quiet space to tune your image.</div>', unsafe_allow_html=True)
    st.toggle("Dark mode", value=True, key="dark_mode", help="Switch between the dark and light editor themes.")
    upload = st.file_uploader("Drop a photo here", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "heic", "heif", "dng", "nef", "nrw", "cr2", "cr3", "arw", "srf", "sr2", "raf", "rw2", "orf", "pef", "ptx", "srw", "3fr", "ari", "bay", "cap", "iiq", "eip", "erf", "mef", "mos", "mrw", "obm", "rwl", "x3f"], help="Drag a photo here or choose a file. HEIC/HEIF iPhone photos convert automatically; RAW files use camera white balance.", label_visibility="visible")
    with st.popover("Editor guide", icon="❔", use_container_width=True):
        st.markdown("### How to use Stillroom")
        st.markdown("**Upload:** Drop in a photo or use Browse files. HEIC/HEIF converts automatically; supported camera RAW files are developed for editing.\n\n**Canvas:** See the current photo as you edit. Changes preview automatically; the full-resolution PNG is prepared when you choose the export button.\n\n**Light & color:** Brightness changes overall light, Contrast changes separation between tones, and Saturation changes color intensity.\n\n**Color balance:** Tint shifts green/magenta. Reds, Greens, and Blues adjust those color channels.\n\n**Detail:** Dehaze adds contrast to hazy images. Denoise smooths speckled noise; high values can soften detail.\n\n**Tone curve:** Choose Master to affect overall tones, or a color channel to affect only that color. Shadows, Midtones, and Highlights set the curve at dark, middle, and bright values.\n\n**Orientation:** Rotate the photo in 90° steps.\n\n**Magic Eraser:** Paint over a small distraction and choose Remove painted area. Undo eraser edits restores the uploaded image.\n\n**AI photo advice:** Ask for editing suggestions based on the displayed photo. Apply changes only with the confirmation button. API usage may incur charges.\n\n**Dominant colors:** View a table of the photo’s most common colors, their HEX codes, and approximate share of the image.")
    st.markdown("<div style='height:20px'></div><div class='eyebrow'>Adjustments</div>", unsafe_allow_html=True)
    with st.expander("Light & color", expanded=True):
        guide_popover("ⓘ", "Light & color", "Brightness lifts or lowers the whole image. Contrast makes light and dark areas farther apart or closer together. Saturation controls how vivid the colors look. The neutral value for each is 1.0.")
        st.caption("Start at 1.00 for the original look. Move a slider a little at a time and watch the canvas.")
        brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.01, key="brightness", help="1.0 is the original brightness.")
        contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.01, key="contrast", help="1.0 is the original contrast.")
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.01, key="saturation", help="1.0 is the original color saturation.")
    with st.expander("Color balance"):
        guide_popover("ⓘ", "Color balance", "Tint shifts the image toward green or magenta. Reds, Greens, and Blues change the strength of each color channel; 1.0 leaves that channel unchanged.")
        st.caption("Use Tint to correct a green or magenta cast. The color sliders start neutral at 1.00.")
        tint = st.slider("Tint", -100, 100, 0, 1, key="tint", help="Shift the color balance toward green or magenta.")
        reds = st.slider("Reds", 0.0, 2.0, 1.0, 0.01, key="reds", help="Adjust the red channel. 1.0 is unchanged.")
        greens = st.slider("Greens", 0.0, 2.0, 1.0, 0.01, key="greens", help="Adjust the green channel. 1.0 is unchanged.")
        blues = st.slider("Blues", 0.0, 2.0, 1.0, 0.01, key="blues", help="Adjust the blue channel. 1.0 is unchanged.")
    with st.expander("Detail"):
        guide_popover("ⓘ", "Detail", "Dehaze restores contrast in hazy or flat scenes. Denoise reduces speckled noise, but stronger settings can smooth fine texture.")
        st.caption("Increase Dehaze for a hazy scene. Raise Denoise gently; too much can blur fine details.")
        dehaze = st.slider("Dehaze", 0, 100, 0, 1, key="dehaze", help="Recover contrast in hazy or flat-looking photos.")
        denoise = st.slider("Denoise", 0, 100, 0, 1, key="denoise", help="Reduce image noise with a median filter.")
    with st.expander("Tone curve"):
        guide_popover("ⓘ", "Tone curve", "Choose Master for overall tone, or Red, Green, or Blue for one channel. Move Shadows, Midtones, and Highlights to change dark, middle, and bright values. The small graph previews the curve shape.")
        st.caption("Choose a channel, then move Shadows, Midtones, or Highlights. Values near the defaults keep the original tone; the graph shows your curve.")
        curve_channel = st.selectbox("Channel", ["Master", "Red", "Green", "Blue"], key="curve_channel")
        channel_key = curve_channel.lower()
        shadows = st.slider("Shadows", 0, 255, key=f"curve_{channel_key}_shadows")
        midtones = st.slider("Midtones", 0, 255, key=f"curve_{channel_key}_midtones")
        highlights = st.slider("Highlights", 0, 255, key=f"curve_{channel_key}_highlights")
        current_points = [(0, 255), (64, shadows), (128, midtones), (192, highlights), (255, 0)]
        svg_points = " ".join(f"{round(x / 255 * 120 + 8, 1)},{round(128 - y / 255 * 120 - 4, 1)}" for x, y in current_points)
        st.markdown(
            f"<svg viewBox='0 0 136 136' style='width:100%;max-width:220px;background:#11120f;border:1px solid #2b2c29;border-radius:8px'><path d='M8 8V128H128' fill='none' stroke='#454740' stroke-width='1'/><polyline points='{svg_points}' fill='none' stroke='#d4f06a' stroke-width='2'/><circle cx='38.1' cy='{round(128 - shadows / 255 * 120 - 4, 1)}' r='3' fill='#d4f06a'/><circle cx='68.5' cy='{round(128 - midtones / 255 * 120 - 4, 1)}' r='3' fill='#d4f06a'/><circle cx='98.6' cy='{round(128 - highlights / 255 * 120 - 4, 1)}' r='3' fill='#d4f06a'/></svg>",
            unsafe_allow_html=True,
        )
    with st.expander("Orientation"):
        guide_popover("ⓘ", "Orientation", "Rotate the photo clockwise in 90-degree steps. This also rotates the final exported image.")
        st.caption("Choose Original unless the photo needs turning. Rotation is included in the exported image.")
        rotation = st.selectbox("Rotation", [0, 90, 180, 270], key="rotation", format_func=lambda x: "Original" if x == 0 else f"{x}° clockwise")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.button("Reset adjustments", use_container_width=True, on_click=reset_adjustments)
    with st.expander("AI connection"):
        typed_api_key = st.text_input(
            "OpenAI API key (optional if configured locally)",
            type="password",
            key="photo_advice_api_key",
            help="A key entered here stays in this browser session. It is not saved to the app files.",
        )
        if typed_api_key.strip() or configured_api_key():
            st.caption("API key ready. Photo questions use billable API tokens.")
        else:
            st.caption("Enter a key here or configure OPENAI_API_KEY locally to enable advice.")

eraser_slot = st.sidebar.empty()
palette_slot = st.sidebar.empty()

st.markdown('<div class="hero"><div><div class="brand">A SIMPLE PHOTO EDITOR</div><h1>See the whole picture.</h1></div><p>Drop in a photo. Find your frame.</p></div>', unsafe_allow_html=True)

if upload is None:
    st.markdown('''<div class="empty"><div class="empty-icon">＋</div><h2>Your canvas is waiting.</h2><p>Drop a photo into the upload area on the left to bring it into view. JPG, PNG, WEBP, TIFF and RAW are supported. iPhone HEIC/HEIF photos convert automatically.</p></div>''', unsafe_allow_html=True)
else:
    try:
        original_image_bytes = upload.getvalue()
        original_upload_id = hashlib.sha256(original_image_bytes).hexdigest()
        extension = Path(upload.name).suffix.lower()
        is_heic = extension in {".heic", ".heif"}
        if is_heic:
            heic_image = ImageOps.exif_transpose(Image.open(io.BytesIO(original_image_bytes))).convert("RGB")
            converted_jpeg = io.BytesIO()
            heic_image.save(converted_jpeg, format="JPEG", quality=95, optimize=True)
            image_bytes = converted_jpeg.getvalue()
            st.caption("HEIC/HEIF detected · converted to JPEG for editing")
        else:
            image_bytes = original_image_bytes
        is_raw = extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
        if st.session_state.get("magic_erased_upload_id") == original_upload_id:
            image_bytes = st.session_state["magic_erased_source"]
            is_raw = False
        curve_values = [
            (st.session_state[f"curve_{channel}_shadows"], st.session_state[f"curve_{channel}_midtones"], st.session_state[f"curve_{channel}_highlights"])
            for channel in ("master", "red", "green", "blue")
        ]
        curves = [tone_curve_lut(*values) for values in curve_values]
        preview, colors = make_preview(image_bytes, is_raw, brightness, contrast, saturation, tint, reds, greens, blues, dehaze, denoise, curves, rotation)
        width, height = preview.size
        st.markdown(f'<div class="canvas-head"><span>Canvas / {upload.name}</span><span>Preview · {width} × {height} px</span></div>', unsafe_allow_html=True)
        guide_popover("ⓘ Canvas guide", "Photo canvas", "This large preview shows how your edits look. Slider changes update the preview automatically. Use Prepare full-resolution PNG near the bottom of the page when you are ready to save the finished image.")
        st.caption("Your edits appear in this preview. Scroll below the tools and choose Prepare full-resolution PNG to save a copy.")
        st.image(preview, use_container_width=True)
        image_id = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get("ai_chat_image_id") != image_id:
            st.session_state["ai_chat_image_id"] = image_id
            st.session_state["ai_chat_messages"] = []

        st.markdown("<div class='eyebrow'>AI PHOTO ADVICE</div>", unsafe_allow_html=True)
        guide_popover("ⓘ AI guide", "AI photo advice", "Ask a question about the photo or what to adjust next. The assistant sees the current preview and settings, then can suggest slider values. Nothing changes until you choose Apply these changes. Requests use your OpenAI API key and may incur usage charges.")
        st.caption("Ask for feedback on the photo and your current edits. Suggestions follow the sidebar order and only apply when you click the button.")
        with st.form("photo_advice_form", clear_on_submit=True):
            question = st.text_input(
                "Ask the photo coach",
                placeholder="What would you adjust next?",
                label_visibility="collapsed",
            )
            ask_button = st.form_submit_button("Ask", type="primary")
        api_key = typed_api_key.strip() or configured_api_key()
        if ask_button and not question.strip():
            st.info("Type a question about the photo first.")
        elif ask_button and not api_key:
            st.error("Add an OpenAI API key in the sidebar's AI connection section first.")
        elif ask_button:
            curve_key = f"curve_{curve_channel.lower()}"
            curve_shadows = st.session_state[f"{curve_key}_shadows"]
            curve_midtones = st.session_state[f"{curve_key}_midtones"]
            curve_highlights = st.session_state[f"{curve_key}_highlights"]
            settings = (
                f"brightness {brightness:.2f}, contrast {contrast:.2f}, saturation {saturation:.2f}, "
                f"tint {tint}, reds {reds:.2f}, greens {greens:.2f}, blues {blues:.2f}, "
                f"dehaze {dehaze}, denoise {denoise}, {curve_channel} curve shadows "
                f"{curve_shadows}, midtones {curve_midtones}, "
                f"highlights {curve_highlights}, rotation {rotation} degrees"
            )
            try:
                with st.spinner("Looking at your photo…"):
                    advice = ask_photo_advisor(
                        api_key,
                        preview,
                        question.strip(),
                        st.session_state["ai_chat_messages"],
                        settings,
                    )
                answer = str(advice.get("message", "")).strip()
                if not answer:
                    answer = "I couldn't produce a text reply for that request. Please try again."
                suggestions = normalize_suggestions(advice.get("suggestions", [])[:8])
                st.session_state["ai_chat_messages"].extend([
                    {"role": "user", "content": question.strip()},
                    {
                        "role": "assistant",
                        "content": answer,
                        "suggestions": suggestions,
                        "curve_channel": curve_channel.lower(),
                        "suggestion_id": uuid.uuid4().hex,
                    },
                ])
            except Exception as exc:
                st.error(f"I couldn't get photo advice ({type(exc).__name__}). Check the key, API access, and connection, then try again.")

        messages = st.session_state["ai_chat_messages"]
        latest_assistant_index = max(
            (index for index, item in enumerate(messages) if item["role"] == "assistant"),
            default=-1,
        )
        for index, message in enumerate(messages):
            if index == latest_assistant_index:
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                suggestions = message.get("suggestions", [])
                if message["role"] == "assistant" and suggestions:
                    st.markdown("**Suggested adjustments**")
                    for suggestion in suggestions:
                        label = SUGGESTION_LABELS[suggestion["control"]]
                        value = suggestion["value"]
                        display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                        st.markdown(f"- **{label} → {display_value}** — {suggestion['reason']}")
                    suggestion_id = message["suggestion_id"]
                    if st.session_state.get(f"ai_suggestion_applied_{suggestion_id}"):
                        st.caption("These suggestions are applied.")
                    else:
                        st.button(
                            "Apply these changes",
                            key=f"apply_ai_suggestions_{suggestion_id}",
                            type="primary",
                            on_click=apply_suggested_edits,
                            args=(suggestions, message.get("curve_channel", "master"), suggestion_id),
                        )

        if latest_assistant_index >= 0:
            latest_advice = messages[latest_assistant_index]
            with st.container(key="floating_ai_advice"):
                with st.expander("Latest AI advice · open", expanded=False):
                    st.markdown(latest_advice["content"])
                    suggestions = latest_advice.get("suggestions", [])
                    if suggestions:
                        st.markdown("**Suggested adjustments**")
                        for suggestion in suggestions:
                            label = SUGGESTION_LABELS[suggestion["control"]]
                            value = suggestion["value"]
                            display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
                            st.markdown(f"- **{label} → {display_value}** — {suggestion['reason']}")
                        suggestion_id = latest_advice["suggestion_id"]
                        if st.session_state.get(f"ai_suggestion_applied_{suggestion_id}"):
                            st.caption("These suggestions are applied.")
                        else:
                            st.button(
                                "Apply these changes",
                                key=f"apply_floating_ai_{suggestion_id}",
                                type="primary",
                                on_click=apply_suggested_edits,
                                args=(suggestions, latest_advice.get("curve_channel", "master"), suggestion_id),
                            )

        # The eraser controls live in the sidebar. Populate this placeholder only
        # after rendering the canvas and AI conversation so eraser issues cannot
        # prevent the photo prompt from appearing below the image.
        with eraser_slot.container():
            with st.expander("Magic Eraser", expanded=False):
                guide_popover("ⓘ", "Magic Eraser", "Paint across a small object or distraction. The app fills the painted area using nearby pixels. Smaller selections on simple backgrounds usually look best. Use Undo eraser edits to start over.")
                st.caption("Paint over a small object, then choose Remove painted area. For best results, use short strokes and a brush just wider than the object.")
                brush_size = st.slider("Brush size", 4, 120, 28, 2, key="magic_eraser_brush")
                try:
                    canvas_result = st_canvas(
                        fill_color="rgba(0, 0, 0, 0)",
                        stroke_width=brush_size,
                        stroke_color="#ff4d7d",
                        background_image=preview.convert("RGBA"),
                        height=preview.height,
                        width=preview.width,
                        drawing_mode="freedraw",
                        key=f"magic_eraser_canvas_{original_upload_id[:12]}_{rotation}",
                        update_streamlit=True,
                    )
                except Exception as eraser_error:
                    canvas_result = None
                    st.warning(f"The Magic Eraser couldn't load ({type(eraser_error).__name__}). Your photo and AI prompt are still available.")
                erase_col, undo_col = st.columns(2)
                with erase_col:
                    erase_clicked = st.button("Remove painted area", type="primary", key="magic_eraser_apply")
                with undo_col:
                    undo_clicked = st.button("Undo eraser edits", key="magic_eraser_undo", disabled=st.session_state.get("magic_erased_upload_id") != original_upload_id)
                if undo_clicked:
                    st.session_state.pop("magic_erased_source", None)
                    st.session_state.pop("magic_erased_upload_id", None)
                    st.rerun()
                if erase_clicked:
                    canvas_pixels = canvas_result.image_data if canvas_result is not None else None
                    if canvas_pixels is None:
                        st.warning("Paint over an area of the photo first.")
                    else:
                        red, green, blue = canvas_pixels[:, :, :3].transpose(2, 0, 1)
                        brush_pixels = (red > 180) & (green < 145) & (blue > 55) & (blue < 190)
                        if not np.any(brush_pixels):
                            st.warning("Paint over an area of the photo first.")
                        else:
                            try:
                                painted_mask = Image.fromarray(brush_pixels.astype(np.uint8) * 255, mode="L")
                                if rotation:
                                    painted_mask = painted_mask.rotate(rotation, expand=True)
                                source_image = open_uploaded_image(image_bytes, is_raw)
                                painted_mask = painted_mask.resize(source_image.size, Image.Resampling.NEAREST)
                                mask_array = np.asarray(painted_mask, dtype=np.uint8)
                                rgb_array = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
                                radius = max(3, min(12, round(max(source_image.size) / max(preview.size) * 3)))
                                repaired = cv2.inpaint(cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR), mask_array, radius, cv2.INPAINT_TELEA)
                                repaired_image = Image.fromarray(cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))
                                repaired_buffer = io.BytesIO()
                                repaired_image.save(repaired_buffer, format="PNG", optimize=True)
                                st.session_state["magic_erased_source"] = repaired_buffer.getvalue()
                                st.session_state["magic_erased_upload_id"] = original_upload_id
                                st.rerun()
                            except Exception as eraser_error:
                                st.warning(f"The painted area couldn't be removed ({type(eraser_error).__name__}). Try a smaller brush selection.")

        st.markdown('<div class="meta"><span>STILLROOM · FAST PREVIEW</span><span>FULL RESOLUTION EXPORT</span></div>', unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td><span style='display:inline-block;width:22px;height:16px;border-radius:4px;background:{hex_color};border:1px solid #ffffff30;vertical-align:middle'></span></td><td><code>{hex_color}</code></td><td>{share:.1f}%</td></tr>"
            for hex_color, share in colors
        )
        with palette_slot.container():
            with st.expander("Dominant colors", expanded=False):
                guide_popover("ⓘ", "Dominant colors", "This table lists the most common colors in the current edited preview, their HEX color codes, and the approximate percentage of the image represented by each swatch.")
                st.caption("Use the HEX values to identify colors in the edited photo. Percentages are approximate.")
                st.markdown(
                    f"<table style='width:100%;border-collapse:collapse;margin-top:8px;font:12px \'DM Mono\',monospace;color:#e9e8e3'><thead><tr style='color:#92938f;text-align:left'><th style='padding:8px 12px;border-bottom:1px solid #2b2c29'>SWATCH</th><th style='padding:8px 12px;border-bottom:1px solid #2b2c29'>HEX</th><th style='padding:8px 12px;border-bottom:1px solid #2b2c29'>IMAGE SHARE</th></tr></thead><tbody>{rows}</tbody></table>",
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        if st.button("Prepare full-resolution PNG", type="primary"):
            with st.spinner("Preparing full-resolution image…"):
                export_bytes, (export_width, export_height) = make_full_resolution_export(
                    image_bytes, is_raw, brightness, contrast, saturation, tint, reds, greens, blues, dehaze, denoise, curves, rotation
                )
            st.download_button(
                f"Download PNG · {export_width} × {export_height}",
                data=export_bytes,
                file_name=f"{Path(upload.name).stem}-edited.png",
                mime="image/png",
                type="primary",
            )

    except Exception as exc:
        st.error(f"We couldn't open that image: {exc}")

