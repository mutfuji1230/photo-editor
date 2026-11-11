[README.md](https://github.com/user-attachments/files/34676079/README.md)
# Stillroom Photo Editor

A Streamlit photo editor with drag-and-drop upload, a large canvas, image adjustments, tone curves, a dominant-color table, RAW support, PNG export, and an optional AI photo-advice chat.

The **Editor guide** button in the sidebar explains the main areas. Small **ⓘ** buttons inside tool sections give focused help for that part of the editor. Use the **Dark mode** toggle in the sidebar to switch themes.

## Requirements

- Python 3.12

## Run

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Then open the local URL Streamlit prints in your browser. Drop a photo into the uploader in the left panel or select it with the file picker.

## AI photo advice

The AI chat sits beside the canvas. Choose **OpenRouter free** for free, image-capable model routing, or **OpenAI with your key** for advice through your own OpenAI API account. Free models have rate limits and availability can change. OpenRouter sends the edited preview to its selected model provider; provider data practices vary, so avoid sensitive photos. OpenAI API usage is billed to the owner of the key. The app does not read API keys from `.env.local`, environment variables, or deployment secrets; each visitor enters a provider key below the AI question and can check that the provider accepts it. Keys stay in that visitor's Streamlit session and are not written to app files. The app server sends the key and edited preview to the selected provider to get advice, so only enter keys on a deployment you trust. Suggestions follow the editor controls and are applied only after you select **Apply these changes**.

RAW support is provided by `rawpy` (LibRaw). Common formats such as DNG, NEF, CR2/CR3, ARW, RAF, RW2, ORF, PEF, and SRW are accepted when supported by the installed LibRaw build. RAW previews use a half-size camera-white-balanced render for responsiveness; PNG export develops the full-resolution image.

HEIC/HEIF iPhone photos can be dropped directly into the main upload box. The app converts them to JPEG in memory and opens the converted image on the editor canvas automatically.


