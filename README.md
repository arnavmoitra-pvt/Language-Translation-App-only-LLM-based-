# LLM Translation App

A simple Tkinter GUI app that uses Hugging Face Transformers to translate between English, Hindi and Bangla. The app prefers Helsinki-NLP Marian models (one-to-one language pairs) and falls back to chained Marian translation (via English) or a multilingual NLLB distilled model when a direct Marian model is not available.

## Contents

- `translation_app.py` — Main GUI application.
- `app.py` — (if present) additional runner or utilities.
- `requirements.txt` — Python dependencies.

## Prerequisites

- Python 3.10+ (your environment shows Python 3.12 — that's fine).
- Recommended: at least 8GB of RAM for small models; NLLB and larger models may require significantly more memory.

## Install dependencies

Open PowerShell in the project folder and run:

```powershell
pip install -r "c:\Users\Bappaditya Mandal\OneDrive\Desktop\Internship project\requirements.txt"
```

Notes:
- `tkinter` usually ships with Python on Windows. If you see errors about `tkinter` missing, install the appropriate Python/tk distribution or use your Python installer to add it.
- `torch` is included in `requirements.txt` as a baseline dependency. On some machines you may want a CPU-only or CUDA-specific wheel; visit https://pytorch.org for platform-specific install commands.

## Run the app

From the project folder run:

```powershell
python "c:\Users\Bappaditya Mandal\OneDrive\Desktop\Internship project\translation_app.py"
```

## How translation selection and fallbacks work

1. The app first tries to load a direct Marian model from Hugging Face using the pattern `Helsinki-NLP/opus-mt-{src}-{tgt}`.
2. If the direct model is not available, the app attempts a chained translation via English (src->en, then en->tgt) using Marian models, if both exist.
3. If chained Marian translation is not sensible or fails, the app attempts a multilingual fallback using `facebook/nllb-200-distilled-600M` and forces the target language token (language code mapping is included for `en`, `hi`, `bn`).

Caveat: chained translations may be slower and can compound translation errors. The NLLB fallback is powerful but large and uses significant RAM.

## Adding / Changing language pairs

Open `translation_app.py`, find `self.language_pairs` in the `TranslationApp.__init__` method and edit/add mappings. The code attempts to use Helsinki-NLP Marian models for mappings of the form `opus-mt-{src}-{tgt}`.

## Hugging Face private models / tokens

If you need to load private models, authenticate with the Hugging Face CLI on your machine before running the app:

```powershell
pip install huggingface-hub
huggingface-cli login
```

Or export an environment variable for the token (less interactive):

```powershell
setx HF_HOME "<your_token>"
```

(If you want, I can add explicit `use_auth_token=` calls in the code to use tokens passed via config.)

## Troubleshooting

- "Import 'transformers' could not be resolved" in VS Code: select the correct Python interpreter in VS Code (the one with `transformers` installed) and restart the window. This is an editor/Pylance warning, not a runtime error if `pip` shows `transformers` is installed.

- Out of memory / model load failures: try running smaller models or remove the NLLB fallback. I can add a checkbox to disable NLLB fallback by default.

- If a specific Helsinki-NLP model ID returns a 404 ("not a local folder and is not a valid model identifier"), the app will attempt chained and multilingual fallbacks before failing. You can also modify the `get_model_name` function or `language_pairs` mapping to point to any other HF model IDs.

## Development notes

- Models are loaded lazily on first use and cached in `self.models` / `self.tokenizers`.
- Error handling shows a dialog with details; see `load_model` for dependency checks and `perform_translation` for fallback logic.
- There's a TODO to optionally pre-check model availability on startup and to restrict the UI to only supported pairs to avoid runtime fallback costs.

## License & Attribution

The app uses models from Hugging Face. Check each model's license on its model card before distribution. The `facebook/nllb-200-distilled-600M` model is CC-BY-NC; Helsinki-NLP models generally use Apache-2.0 or similar — verify on Hugging Face.

---

If you'd like, I can:

- Add a small settings UI to enable/disable NLLB fallback.
- Add a startup availability check to show which pairs are available locally/online.
- Add example screenshots or a short demo script that runs a non-GUI translation (useful for CI or headless testing).

Which follow-up would you like next?
