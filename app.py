import base64
import hashlib
import io
import json
from pathlib import Path
import urllib.error
import urllib.request
import uuid

from openai import OpenAI
from pillow_heif import register_heif_opener
import rawpy
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(page_title="Stillroom · Photo Editor", page_icon="◫", layout="wide", initial_sidebar_state="expanded")
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


def clear_ai_key_validation():
    st.session_state.pop("ai_key_validation", None)


def validate_ai_key(provider, api_key):
    endpoint = (
        "https://openrouter.ai/api/v1/key"
        if provider == "OpenRouter free"
        else "https://api.openai.com/v1/models"
    )
    request = urllib.request.Request(
        endpoint,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        if response.status != 200:
            raise urllib.error.HTTPError(endpoint, response.status, "Key check failed", response.headers, None)


def parse_photo_advice(output):
    raw_output = str(output or "").strip()
    candidate = raw_output
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    parsed = None
    for text in (candidate, candidate[candidate.find("{"):candidate.rfind("}") + 1]):
        if not text or "{" not in text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break

    if parsed is None:
        return {"message": raw_output, "suggestions": []}
    message = str(parsed.get("message", "")).strip() or raw_output
    suggestions = parsed.get("suggestions", [])
    return {
        "message": message,
        "suggestions": suggestions if isinstance(suggestions, list) else [],
    }


def ask_photo_advisor(provider, api_key, image, question, history, settings):
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG", quality=86, optimize=True)
    image_data_url = "data:image/jpeg;base64," + base64.b64encode(image_buffer.getvalue()).decode("ascii")

    system_prompt = (
        "You are a relaxed, helpful photo-editing buddy with a good photographer's eye. Sound casual, "
        "warm, and natural, like you're looking at the photo together. Avoid formal reports, canned praise, "
        "and bare diagnoses such as 'the photo is underexposed.' Instead, point out one specific thing you "
        "notice, then give a clear next move using the editor's actual control names and a modest suggested "
        "amount or direction. Briefly say what that change should improve. For example: 'The sunset colors "
        "are lovely; the foreground is getting a little lost, though. Try lifting Shadows to about 80 and "
        "Brightness just a touch, maybe 1.05, so the foreground opens up without flattening the sky.' "
        "Give one or two useful steps at a time, not a long checklist. If the image already looks good, say "
        "what is working and offer an optional tweak. You can also answer general questions about photography, "
        "the current photo, and how this editor works. Explain settings such as Brightness, Contrast, Saturation, "
        "Tint, color channels, Dehaze, Denoise, and Tone curve in plain language, with a short example when useful. "
        "For definitions or general questions, answer directly and return an empty suggestions array. Only suggest "
        "slider changes when the user asks for editing advice or a change would clearly help; keep those suggestions "
        "short and optional. Match the user's level of detail, be honest when something is subjective or unclear, "
        "and do not claim to have changed the photo. Return a JSON object with a casual user-facing message and a "
        "suggestions array. Each suggestion must use one real editor control, provide a target value, and briefly "
        "explain why. For tone-curve suggestions, use the currently selected curve channel. Suggest values within "
        "these ranges: Brightness/Contrast/Saturation/Reds/Greens/Blues 0–2; Tint -100–100; Dehaze/Denoise 0–100; "
        "curve points 0–255; Rotation 0, 90, 180, or 270."
    )
    request_input = [{"role": "developer", "content": system_prompt}]
    history_items = []
    for item in history[-10:]:
        history_text = item["content"]
        if item.get("suggestions"):
            summary = "; ".join(
                f"{SUGGESTION_LABELS[suggestion['control']]} to {suggestion['value']}"
                for suggestion in item["suggestions"]
            )
            history_text += f"\nSuggested slider changes: {summary}."
        history_items.append({"role": item["role"], "content": history_text})
    request_input.extend(history_items)
    request_input.append({
        "role": "user",
        "content": [
            {"type": "input_text", "text": f"{question}\n\nCurrent editor settings: {settings}"},
            {"type": "input_image", "image_url": image_data_url, "detail": "high"},
        ],
    })
    if provider == "OpenRouter free":
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        model = "openrouter/free"
    else:
        client = OpenAI(api_key=api_key)
        model = "gpt-5.5"
    response = client.responses.create(
        model=model,
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
    return parse_photo_advice(response.output_text)


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
@font-face { font-family:'Newsreader'; src:url('app/static/Newsreader-VF.woff2') format('woff2'); font-style:normal; font-weight:300 800; font-display:swap; }
:root { --ink:#f0eee8; --muted:#aaa99f; --line:#3a3b34; --panel:#21221d; --accent:#d4b87b; --action:#e0cf9d; --action-ink:#29261d; --surface:#151512; --sidebar:#1c1c18; --upload:#1d211d; --upload-line:#727c69; --ai:#c3b3d6; --tint:#b5a0d1; --red:#df8981; --green:#91b68a; --blue:#8caec2; --teal:#81aaa4; }
.stApp { background:var(--surface); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.3rem; }
.block-container { padding-top:2.2rem; max-width:1500px; }
header[data-testid="stHeader"] { background:transparent; }
.brand { font:600 11px 'DM Sans',sans-serif; letter-spacing:.13em; color:var(--accent); text-transform:uppercase; }
.hero { display:flex; justify-content:space-between; align-items:end; gap:24px; border-bottom:1px solid var(--line); padding-bottom:20px; margin-bottom:24px; }
.hero h1, .empty h2, .sidebar-title { font-family:'Newsreader',Georgia,serif; font-optical-sizing:auto; font-weight:500; }
.hero h1 { font-size:clamp(38px,4.2vw,56px); line-height:1.04; letter-spacing:-.025em; text-wrap:balance; margin:9px 0 0; }
.hero p { color:var(--muted); margin:0 0 3px; font-size:15px; line-height:1.5; text-wrap:pretty; max-width:34ch; }
.eyebrow { font:11px 'DM Mono',monospace; color:var(--muted); letter-spacing:.11em; text-transform:uppercase; }
.canvas-head { display:flex; justify-content:space-between; margin:4px 0 12px; color:var(--muted); font:11px 'DM Mono',monospace; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stFileUploader"] { border:1px dashed var(--upload-line); border-radius:12px; background:var(--upload); padding:10px; }
[data-testid="stFileUploader"] section { border:0; }
[data-testid="stFileUploader"] section > div { padding:18px 8px; }
[data-testid="stFileUploader"] button { background:var(--action); color:var(--action-ink); border:0; border-radius:6px; font-weight:600; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--accent); border-color:var(--accent); }
.st-key-tint [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--tint); border-color:var(--tint); }
.st-key-reds [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"], [class*="st-key-curve_red_"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--red); border-color:var(--red); }
.st-key-greens [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"], [class*="st-key-curve_green_"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--green); border-color:var(--green); }
.st-key-blues [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"], [class*="st-key-curve_blue_"] [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--blue); border-color:var(--blue); }
.st-key-dehaze [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"], .st-key-denoise [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:var(--teal); border-color:var(--teal); }
.stButton button, .stDownloadButton button { border-radius:8px; font-weight:600; transition:background-color .15s ease, border-color .15s ease; }
.stButton button[kind="primary"] { background:var(--action); color:var(--action-ink); border:0; }
.sidebar-title { font-size:25px; line-height:1.1; letter-spacing:-.02em; margin:0 0 4px; }
.sidebar-note { color:var(--muted); font-size:13px; line-height:1.5; margin-bottom:26px; }
.empty { min-height:58vh; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; border:1px dashed var(--line); border-radius:12px; background:var(--panel); padding:clamp(24px,5vw,48px); }
.empty-icon { width:40px; height:40px; color:var(--muted); margin-bottom:18px; }
.empty h2 { font-size:clamp(29px,3vw,36px); line-height:1.12; letter-spacing:-.02em; text-wrap:balance; margin:0 0 10px; }
.empty p { max-width:45ch; color:var(--muted); line-height:1.65; font-size:15px; text-wrap:pretty; margin:0; }
.st-key-prepare_edited_png button { width:100%; }
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:10px; background:var(--panel); }
[data-testid="stExpander"] summary { font-weight:600; }
.palette-table { width:100%; border-collapse:collapse; margin-top:8px; font:12px 'DM Mono',monospace; color:var(--ink); }
.palette-table th, .palette-table td { padding:9px 8px; text-align:left; border-bottom:1px solid var(--line); }
.palette-table th { color:var(--muted); font:600 10px 'DM Sans',sans-serif; letter-spacing:.08em; text-transform:uppercase; }
.palette-table tr:last-child td { border-bottom:0; }
.meta { display:flex; justify-content:space-between; color:var(--muted); font:10px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.08em; margin-top:11px; }
[data-testid="stImage"] img { border-radius:8px; }
.st-key-ai_advice_panel { position:sticky !important; top:1rem; align-self:flex-start; max-height:calc(100vh - 2rem); overflow-y:auto; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
.ai-panel-title { color:var(--ink); font-size:19px; font-weight:600; letter-spacing:-.01em; margin-bottom:4px; }
.ai-panel-note { color:var(--muted); font-size:13px; line-height:1.5; margin-bottom:14px; }
.st-key-ai_advice_panel [data-testid="stChatMessage"] { padding:10px 12px; }
.ai-label { color:var(--ai); }
hr { border-color:var(--line); }
::selection { background:color-mix(in srgb, var(--ai) 35%, transparent); color:var(--ink); }
:focus-visible { outline:2px solid var(--ai); outline-offset:2px; }
* { scrollbar-color:var(--muted) var(--surface); scrollbar-width:thin; }
@media (max-width:760px) {
    .block-container { padding-top:1.4rem; }
    .hero { align-items:flex-start; flex-direction:column; gap:10px; padding-bottom:16px; }
    .hero h1 { font-size:clamp(36px,9vw,48px); }
    .empty { min-height:48vh; }
    .st-key-ai_advice_panel { position:static !important; max-height:none; overflow:visible; }
}
</style>
""", unsafe_allow_html=True)

if not st.session_state.get("dark_mode", True):
    st.markdown("""
    <style>
    :root { --ink:#2a2b25; --muted:#62645b; --line:#d8d5ca; --panel:#fffdf8; --accent:#755c35; --action:#e7d9b1; --action-ink:#332d20; --surface:#f3f0e7; --sidebar:#eae8df; --upload:#f4f5ed; --upload-line:#929e7d; --ai:#735b8c; --tint:#765b92; --red:#ae5750; --green:#42744c; --blue:#3e6683; --teal:#326f69; }
    .stApp { background:var(--surface); color:var(--ink); }
    [data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--line); }
    [data-testid="stHeader"] { background:rgba(244,244,237,.94); }
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] { color:var(--muted); }
    .hero p, .sidebar-note, .eyebrow, .canvas-head, .meta { color:var(--muted); }
    [data-testid="stExpander"], [data-testid="stPopover"] > div,
    [data-baseweb="popover"] > div { background:var(--panel); color:var(--ink); border-color:var(--line); }
    [data-testid="stFileUploader"] { background:var(--upload); border-color:var(--upload-line); }
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input, [data-testid="stTextArea"] textarea {
        background:#fff; color:#252620; border-color:#c7c9bf;
    }
    [data-testid="stButton"] button, [data-testid="stDownloadButton"] button,
    [data-testid="stPopover"] button { color:#252620; border-color:#c7c9bf; }
    [data-testid="stButton"] button[kind="primary"], [data-testid="stDownloadButton"] button[kind="primary"] {
        background:var(--action); color:var(--action-ink); border:0;
    }
    .empty { background:var(--panel); border-color:var(--line); }
    .empty h2 { color:var(--ink); }
    .empty p { color:var(--muted); }
    table { color:var(--ink) !important; }
    svg { border-color:var(--line) !important; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    for channel in ("master", "red", "green", "blue"):
        for point, value in (("shadows", 64), ("midtones", 128), ("highlights", 192)):
            st.session_state.setdefault(f"curve_{channel}_{point}", value)
    st.markdown('<div class="brand">STILLROOM</div><div style="height:14px"></div><div class="sidebar-title">Make it yours.</div><div class="sidebar-note">Adjust your photo with simple controls.</div>', unsafe_allow_html=True)
    st.toggle("Dark mode", value=True, key="dark_mode", help="Switch between the dark and light editor themes.")
    upload = st.file_uploader("Drop a photo here", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "heic", "heif", "dng", "nef", "nrw", "cr2", "cr3", "arw", "srf", "sr2", "raf", "rw2", "orf", "pef", "ptx", "srw", "3fr", "ari", "bay", "cap", "iiq", "eip", "erf", "mef", "mos", "mrw", "obm", "rwl", "x3f"], help="Drag a photo here or choose a file. HEIC/HEIF iPhone photos convert automatically; RAW files use camera white balance.", label_visibility="visible")
    with st.popover("Editor guide", icon="❔", use_container_width=True):
        st.markdown("### How to use Stillroom")
        st.markdown("**Upload:** Choose a photo. HEIC and supported RAW files work too.\n\n**Canvas:** Preview edits. Export a full-size PNG from the sidebar.\n\n**Light & color:** Adjust brightness, contrast, and color intensity.\n\n**Color balance:** Shift tint or adjust red, green, and blue.\n\n**Detail:** Reduce haze or noise. Strong denoise can soften detail.\n\n**Tone curve:** Choose a channel; adjust dark, mid, or bright tones.\n\n**Orientation:** Rotate in 90° steps.\n\n**AI advice:** Ask for suggestions; confirm before applying.\n\n**Color table:** See common colors and HEX codes.")
    st.markdown("<div style='height:20px'></div><div class='eyebrow'>Adjustments</div>", unsafe_allow_html=True)
    with st.expander("Light & color", expanded=True):
        guide_popover("ⓘ", "Light & color", "Brightness lifts or lowers the whole image. Contrast makes light and dark areas farther apart or closer together. Saturation controls how vivid the colors look. The neutral value for each is 1.0.")
        brightness = st.slider("Brightness", 0.0, 2.0, 1.0, 0.01, key="brightness", help="1.0 is the original brightness.")
        contrast = st.slider("Contrast", 0.0, 2.0, 1.0, 0.01, key="contrast", help="1.0 is the original contrast.")
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.01, key="saturation", help="1.0 is the original color saturation.")
    with st.expander("Color balance"):
        guide_popover("ⓘ", "Color balance", "Tint shifts the image toward green or magenta. Reds, Greens, and Blues change the strength of each color channel; 1.0 leaves that channel unchanged.")
        tint = st.slider("Tint", -100, 100, 0, 1, key="tint", help="Shift the color balance toward green or magenta.")
        reds = st.slider("Reds", 0.0, 2.0, 1.0, 0.01, key="reds", help="Adjust the red channel. 1.0 is unchanged.")
        greens = st.slider("Greens", 0.0, 2.0, 1.0, 0.01, key="greens", help="Adjust the green channel. 1.0 is unchanged.")
        blues = st.slider("Blues", 0.0, 2.0, 1.0, 0.01, key="blues", help="Adjust the blue channel. 1.0 is unchanged.")
    with st.expander("Detail"):
        guide_popover("ⓘ", "Detail", "Dehaze restores contrast in hazy or flat scenes. Denoise reduces speckled noise, but stronger settings can smooth fine texture.")
        dehaze = st.slider("Dehaze", 0, 100, 0, 1, key="dehaze", help="Recover contrast in hazy or flat-looking photos.")
        denoise = st.slider("Denoise", 0, 100, 0, 1, key="denoise", help="Reduce image noise with a median filter.")
    with st.expander("Tone curve"):
        guide_popover("ⓘ", "Tone curve", "Choose Master for overall tone, or Red, Green, or Blue for one channel. Move Shadows, Midtones, and Highlights to change dark, middle, and bright values. The small graph previews the curve shape.")
        curve_channel = st.selectbox("Channel", ["Master", "Red", "Green", "Blue"], key="curve_channel")
        channel_key = curve_channel.lower()
        curve_color = {"Master": "var(--accent)", "Red": "var(--red)", "Green": "var(--green)", "Blue": "var(--blue)"}[curve_channel]
        shadows = st.slider("Shadows", 0, 255, key=f"curve_{channel_key}_shadows")
        midtones = st.slider("Midtones", 0, 255, key=f"curve_{channel_key}_midtones")
        highlights = st.slider("Highlights", 0, 255, key=f"curve_{channel_key}_highlights")
        current_points = [(0, 255), (64, shadows), (128, midtones), (192, highlights), (255, 0)]
        svg_points = " ".join(f"{round(x / 255 * 120 + 8, 1)},{round(128 - y / 255 * 120 - 4, 1)}" for x, y in current_points)
        st.markdown(
            f"<svg viewBox='0 0 136 136' style='--curve-color:{curve_color};width:100%;max-width:220px;background:var(--panel);border:1px solid var(--line);border-radius:8px'><path d='M8 8V128H128' fill='none' stroke='var(--muted)' stroke-width='1'/><polyline points='{svg_points}' fill='none' stroke='var(--curve-color)' stroke-width='2'/><circle cx='38.1' cy='{round(128 - shadows / 255 * 120 - 4, 1)}' r='3' fill='var(--curve-color)'/><circle cx='68.5' cy='{round(128 - midtones / 255 * 120 - 4, 1)}' r='3' fill='var(--curve-color)'/><circle cx='98.6' cy='{round(128 - highlights / 255 * 120 - 4, 1)}' r='3' fill='var(--curve-color)'/></svg>",
            unsafe_allow_html=True,
        )
    with st.expander("Orientation"):
        guide_popover("ⓘ", "Orientation", "Rotate the photo clockwise in 90-degree steps. This also rotates the final exported image.")
        rotation = st.selectbox("Rotation", [0, 90, 180, 270], key="rotation", format_func=lambda x: "Original" if x == 0 else f"{x}° clockwise")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.button("Reset adjustments", use_container_width=True, on_click=reset_adjustments)

export_slot = st.sidebar.empty()
palette_slot = st.sidebar.empty()

hero_hint = "Upload a photo to get started." if upload is None else "Adjust the photo, then export a PNG."
st.markdown(f'<div class="hero"><h1>Edit your photo.</h1><p>{hero_hint}</p></div>', unsafe_allow_html=True)

if upload is None:
    st.markdown('''<div class="empty"><svg class="empty-icon" viewBox="0 0 48 48" fill="none" aria-hidden="true"><rect x="5.5" y="8.5" width="37" height="31" rx="3.5" stroke="currentColor" stroke-width="1.8"/><circle cx="17" cy="19" r="3.2" stroke="currentColor" stroke-width="1.8"/><path d="m8 35 10.5-10 6.5 6 5-4 10 8" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg><h2>Start by adding a photo.</h2><p>Drop one into the uploader on the left. Then make edits, ask AI for advice if you like, and export a PNG.</p></div>''', unsafe_allow_html=True)
else:
    try:
        original_image_bytes = upload.getvalue()
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
        curve_values = [
            (st.session_state[f"curve_{channel}_shadows"], st.session_state[f"curve_{channel}_midtones"], st.session_state[f"curve_{channel}_highlights"])
            for channel in ("master", "red", "green", "blue")
        ]
        curves = [tone_curve_lut(*values) for values in curve_values]
        preview, colors = make_preview(image_bytes, is_raw, brightness, contrast, saturation, tint, reds, greens, blues, dehaze, denoise, curves, rotation)
        width, height = preview.size
        canvas_column, advice_column = st.columns([2.2, 1], gap="medium", vertical_alignment="top")
        with canvas_column:
            st.markdown(f'<div class="canvas-head"><span>Canvas / {upload.name}</span><span>Preview · {width} × {height} px</span></div>', unsafe_allow_html=True)
            guide_popover("ⓘ Canvas guide", "Photo canvas", "Your edits appear here. Use Export in the sidebar to prepare and download a full-resolution PNG.")
            st.image(preview, use_container_width=True)
        image_id = hashlib.sha256(image_bytes).hexdigest()
        if st.session_state.get("ai_chat_image_id") != image_id:
            st.session_state["ai_chat_image_id"] = image_id
            st.session_state["ai_chat_messages"] = []

        export_signature = hashlib.sha256(
            repr((image_id, brightness, contrast, saturation, tint, reds, greens, blues, dehaze, denoise, curve_values, rotation)).encode("utf-8")
        ).hexdigest()
        if st.session_state.get("prepared_export_signature") != export_signature:
            st.session_state.pop("prepared_export_bytes", None)
            st.session_state.pop("prepared_export_size", None)
            st.session_state.pop("prepared_export_signature", None)
        with export_slot.container():
            st.markdown("<div class='eyebrow'>Export</div>", unsafe_allow_html=True)
            if st.button("Prepare edited PNG", type="primary", key="prepare_edited_png"):
                with st.spinner("Preparing full-resolution image…"):
                    prepared_bytes, prepared_size = make_full_resolution_export(
                        image_bytes, is_raw, brightness, contrast, saturation, tint, reds, greens, blues, dehaze, denoise, curves, rotation
                    )
                st.session_state["prepared_export_bytes"] = prepared_bytes
                st.session_state["prepared_export_size"] = prepared_size
                st.session_state["prepared_export_signature"] = export_signature
            if st.session_state.get("prepared_export_signature") == export_signature:
                export_width, export_height = st.session_state["prepared_export_size"]
                st.download_button(
                    f"Download edited PNG · {export_width} × {export_height}",
                    data=st.session_state["prepared_export_bytes"],
                    file_name=f"{Path(upload.name).stem}-edited.png",
                    mime="image/png",
                    type="primary",
                    key="download_edited_png",
                )

        with advice_column:
            with st.container(key="ai_advice_panel"):
                st.markdown("<div class='ai-panel-title'>Photo advice</div>", unsafe_allow_html=True)
                st.markdown("<div class='ai-panel-note'>Ask about your photo, photography, or how the editing controls work.</div>", unsafe_allow_html=True)
                guide_popover("ⓘ AI guide", "AI photo advice", "The selected provider sees the current preview and settings, then can suggest slider values. Nothing changes until you choose Apply these changes. OpenAI API requests are billed to the owner of the key entered below.")
                with st.form("photo_advice_form", clear_on_submit=True):
                    question = st.text_input(
                        "Ask the photo coach",
                        placeholder="Ask about your photo or an editing setting…",
                        label_visibility="collapsed",
                    )
                    ask_button = st.form_submit_button("Ask", type="primary", use_container_width=True)
                ai_provider = st.selectbox(
                    "AI provider",
                    ["OpenRouter free", "OpenAI with your key"],
                    key="ai_provider",
                    help="Use OpenRouter's free image-capable model, or enter your own OpenAI API key.",
                    on_change=clear_ai_key_validation,
                )
                key_label = "OpenRouter API key" if ai_provider == "OpenRouter free" else "OpenAI API key"
                key_name = "openrouter_api_key" if ai_provider == "OpenRouter free" else "openai_user_api_key"
                typed_api_key = st.text_input(
                    key_label,
                    type="password",
                    key=key_name,
                    help="Used by the app server for your requests and kept only in this session, not app files.",
                    on_change=clear_ai_key_validation,
                )
                key_url = "https://openrouter.ai/keys" if ai_provider == "OpenRouter free" else "https://platform.openai.com/api-keys"
                st.markdown(f"[Get a {key_label}](<{key_url}>)")
                if ai_provider == "OpenRouter free":
                    st.caption("Free image-capable models; availability and rate limits apply. Your photo is sent to the model provider, so avoid sensitive images.")
                else:
                    st.caption("OpenAI API usage is billed to the owner of the key you enter.")
                api_key = typed_api_key.strip()
                key_fingerprint = hashlib.sha256(f"{ai_provider}\0{api_key}".encode("utf-8")).hexdigest() if api_key else ""
                if st.button("Check key", key="check_ai_provider_key", use_container_width=True):
                    if not api_key:
                        st.warning("Enter a key above first.")
                    else:
                        try:
                            with st.spinner("Checking key…"):
                                validate_ai_key(ai_provider, api_key)
                            st.session_state["ai_key_validation"] = key_fingerprint
                        except urllib.error.HTTPError as exc:
                            if exc.code in (401, 403):
                                st.error("The provider didn't accept this key. Check it and try again.")
                            else:
                                st.error(f"Key check failed (HTTP {exc.code}). Try again.")
                        except (urllib.error.URLError, TimeoutError):
                            st.error("Couldn't reach the provider to check this key. Try again.")
                if key_fingerprint and st.session_state.get("ai_key_validation") == key_fingerprint:
                    st.success("Key accepted by the selected provider.")
                if ask_button and not question.strip():
                    st.info("Type a question about the photo first.")
                elif ask_button and not api_key:
                    st.error("Enter a key for the selected provider below first.")
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
                                ai_provider,
                                api_key,
                                preview,
                                question.strip(),
                                st.session_state["ai_chat_messages"],
                                settings,
                            )
                        st.session_state["ai_key_validation"] = key_fingerprint
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
                        st.error(f"I couldn't get photo advice ({type(exc).__name__}). Check the selected key, provider limits, and connection, then try again.")

                for message in st.session_state["ai_chat_messages"]:
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

        canvas_column.markdown('<div class="meta"><span>Stillroom · preview</span><span>Full resolution export</span></div>', unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td><span style='display:inline-block;width:22px;height:16px;border-radius:4px;background:{hex_color};border:1px solid #ffffff30;vertical-align:middle'></span></td><td><code>{hex_color}</code></td><td>{share:.1f}%</td></tr>"
            for hex_color, share in colors
        )
        with palette_slot.container():
            with st.expander("Dominant colors", expanded=False):
                guide_popover("ⓘ", "Dominant colors", "This table lists the most common colors in the current edited preview, their HEX color codes, and the approximate percentage of the image represented by each swatch.")
                st.markdown(
                    f"<table class='palette-table'><thead><tr><th>Swatch</th><th>HEX</th><th>Image share</th></tr></thead><tbody>{rows}</tbody></table>",
                    unsafe_allow_html=True,
                )
    except Exception as exc:
        st.error(f"We couldn't open that image: {exc}")

