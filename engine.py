# File: engine.py
# Core TTS model loading and speech generation logic.

import gc
import inspect
import logging
import os
import random
import re
import unicodedata
import numpy as np
import torch
from typing import Optional, Tuple
from pathlib import Path

from chatterbox.tts import ChatterboxTTS  # Main TTS engine class
from chatterbox.models.s3gen.const import (
    S3GEN_SR,
)  # Default sample rate from the engine

# Defensive Turbo import - Turbo may not be available in older package versions
try:
    from chatterbox.tts_turbo import ChatterboxTurboTTS

    TURBO_AVAILABLE = True
except ImportError:
    ChatterboxTurboTTS = None
    TURBO_AVAILABLE = False

# Defensive Multilingual import
try:
    from chatterbox import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

    MULTILINGUAL_AVAILABLE = True
except ImportError:
    ChatterboxMultilingualTTS = None
    SUPPORTED_LANGUAGES = {}
    MULTILINGUAL_AVAILABLE = False

# Import the singleton config_manager
from config import config_manager

logger = logging.getLogger(__name__)

# --- Compatibility shim for official chatterbox (resemble-ai 0.1.7+) ---
# Its bundled S3Tokenizer.log_mel_spectrogram does `torch.from_numpy(wav)` without
# casting, while `_mel_filters` is float32. When a reference clip arrives as float64
# (e.g. the Turbo voice-prep path), the `_mel_filters @ magnitudes` matmul raises
# "expected scalar type Double but found Float". Coerce audio to the filter dtype.
try:
    from chatterbox.models.s3tokenizer.s3tokenizer import S3Tokenizer as _S3Tok

    if not getattr(_S3Tok, "_dtype_shim_applied", False):
        _orig_log_mel = _S3Tok.log_mel_spectrogram

        def _log_mel_dtype_safe(self, audio, padding: int = 0):
            if not torch.is_tensor(audio):
                audio = torch.from_numpy(audio)
            audio = audio.to(self._mel_filters.dtype)
            return _orig_log_mel(self, audio, padding)

        _S3Tok.log_mel_spectrogram = _log_mel_dtype_safe
        _S3Tok._dtype_shim_applied = True
        logger.info("Applied S3Tokenizer log_mel_spectrogram dtype-safety shim.")
except Exception as _shim_err:  # pragma: no cover - defensive
    logger.warning(f"Could not apply S3Tokenizer dtype shim: {_shim_err}")

# Root-cause shim for the Turbo voice-prep dtype crash: ChatterboxTurboTTS.norm_loudness
# uses pyloudnorm, whose float64 gain promotes the waveform to float64. That float64 wav
# then flows into float32 model inputs (voice-encoder LSTM, s3 tokenizer) and raises
# "input must have the type torch.float32, got type torch.float64". Cast back to float32.
if TURBO_AVAILABLE:
    try:
        if not getattr(ChatterboxTurboTTS, "_norm_loudness_dtype_shim", False):
            _orig_norm_loudness = ChatterboxTurboTTS.norm_loudness

            def _norm_loudness_f32(self, wav, sr, target_lufs=-27):
                out = _orig_norm_loudness(self, wav, sr, target_lufs)
                try:
                    out = out.astype(np.float32)
                except Exception:
                    pass
                return out

            ChatterboxTurboTTS.norm_loudness = _norm_loudness_f32
            ChatterboxTurboTTS._norm_loudness_dtype_shim = True
            logger.info("Applied ChatterboxTurboTTS.norm_loudness float32 shim.")
    except Exception as _shim_err:  # pragma: no cover - defensive
        logger.warning(f"Could not apply norm_loudness dtype shim: {_shim_err}")

# Russian auto-stress: the multilingual model was trained on stress-marked Russian
# (combining acute U+0301 after the stressed vowel) and the tokenizer calls
# `add_russian_stress()` for language_id=='ru'. Upstream that needs `russian_text_stresser`
# (not installed / not on PyPI). We substitute `ruaccent`, converting its '+'-before-vowel
# markers to U+0301-after-vowel. The model is lazy-loaded into the persistent hf_cache
# volume on first Russian synthesis (~one-time download), then reused.
try:
    import chatterbox.models.tokenizers.tokenizer as _tk_mod

    _ruaccent_holder = {"obj": None, "failed": False}
    _RUACCENT_PLUS_RE = re.compile(r"\+(.)")
    _COMBINING_ACUTE = chr(0x0301)  # stress mark the model expects after the stressed vowel
    _RUACCENT_WORKDIR = os.environ.get("RUACCENT_WORKDIR", "/app/hf_cache/ruaccent")

    def _ruaccent_add_russian_stress(text: str) -> str:
        if _ruaccent_holder["failed"]:
            return text
        try:
            if _ruaccent_holder["obj"] is None:
                from ruaccent import RUAccent
                _acc = RUAccent()
                _acc.load(
                    omograph_model_size="turbo3.1",
                    use_dictionary=True,
                    workdir=_RUACCENT_WORKDIR,
                )
                _ruaccent_holder["obj"] = _acc
                logger.info("ruaccent Russian stresser loaded.")
            # Upstream preprocess_text NFKD-decomposes ё->е+U+0308 and й->и+U+0306.
            # ruaccent's normalize regex strips combining marks not in its allow-list,
            # so ё/й are silently destroyed before stressing. NFC-recompose first so
            # ruaccent sees real ё/й and preserves them in its output.
            text = unicodedata.normalize("NFC", text)
            marked = _ruaccent_holder["obj"].process_all(text)
            # ruaccent marks '+' BEFORE the stressed vowel; model wants U+0301 AFTER it.
            return _RUACCENT_PLUS_RE.sub(lambda m: m.group(1) + _COMBINING_ACUTE, marked)
        except Exception as _e:
            _ruaccent_holder["failed"] = True
            logger.warning(f"ruaccent stressing unavailable, using unstressed Russian: {_e}")
            return text

    _tk_mod.add_russian_stress = _ruaccent_add_russian_stress
    logger.info("Installed ruaccent-based Russian stress substitution.")
except Exception as _ru_err:  # pragma: no cover - defensive
    logger.warning(f"Could not install ruaccent Russian stresser: {_ru_err}")

# Log BF16 setting at module load so it's visible in startup logs
# (BF16_ENABLED is resolved after logger is set up — logged in initialize_tts_model)
if TURBO_AVAILABLE:
    logger.info("ChatterboxTurboTTS is available in the installed chatterbox package.")
else:
    logger.info("ChatterboxTurboTTS not available in installed chatterbox package.")

# Log Multilingual availability status at module load time
if MULTILINGUAL_AVAILABLE:
    logger.info("ChatterboxMultilingualTTS is available in the installed chatterbox package.")
    logger.info(f"Supported languages: {list(SUPPORTED_LANGUAGES.keys())}")
else:
    logger.info("ChatterboxMultilingualTTS not available in installed chatterbox package.")

# Model selector whitelist - maps config values to model types
MODEL_SELECTOR_MAP = {
    # Original model selectors
    "chatterbox": "original",
    "original": "original",
    "resembleai/chatterbox": "original",
    # Turbo model selectors
    "chatterbox-turbo": "turbo",
    "turbo": "turbo",
    "resembleai/chatterbox-turbo": "turbo",
    # Multilingual model selectors
    "chatterbox-multilingual": "multilingual",
    "multilingual": "multilingual",
}

# Paralinguistic tags supported by Turbo model
# Full set of dedicated tags recognized by the Chatterbox-Turbo tokenizer.
# These are real single tokens in the model's `added_tokens.json` (IDs 50257-50275),
# not free text — they must be written verbatim and lowercase, e.g. "[whispering]".
# Sounds = non-speech vocal events; emotions/styles steer delivery and replace the
# original model's exaggeration/cfg_weight knobs (which Turbo ignores).
TURBO_PARALINGUISTIC_TAGS = [
    # Non-speech sounds
    "laugh",
    "chuckle",
    "sigh",
    "gasp",
    "cough",
    "clear throat",
    "sniff",
    "groan",
    "shush",
    # Emotions
    "angry",
    "fear",
    "surprised",
    "crying",
    "happy",
    "sarcastic",
    # Speaking styles
    "whispering",
    "dramatic",
    "narration",
    "advertisement",
]

# --- BF16 optimization flag ---
# TTS_BF16: controls whether T3 is converted to bfloat16 and whether
# autocast is used during inference. Off by default so existing users
# see no behavior change on upgrade — opt in for the speedup.
#   off (default) — keep T3 in float32, no autocast
#   on / 1 / true  — force-enable (assumes hardware supports bf16)
#   auto           — enable only if torch.cuda.is_bf16_supported()
def _resolve_bf16_setting() -> bool:
    val = os.environ.get("TTS_BF16", "off").strip().lower()
    if val in ("on", "1", "true"):
        return True
    if val == "auto":
        if torch.cuda.is_available():
            return torch.cuda.is_bf16_supported()
        return False
    # off / 0 / false / anything else
    return False

BF16_ENABLED: bool = _resolve_bf16_setting()

# --- Global Module Variables ---
chatterbox_model: Optional[ChatterboxTTS] = None
MODEL_LOADED: bool = False
model_device: Optional[str] = (
    None  # Stores the resolved device string ('cuda' or 'cpu')
)

# Track which model type is loaded
loaded_model_type: Optional[str] = None  # "original" or "turbo"
loaded_model_class_name: Optional[str] = None  # "ChatterboxTTS" or "ChatterboxTurboTTS"
# For multilingual: the T3 version actually loaded ("v2"/"v3"), which may differ from
# the requested config value if the installed package can't honor the request.
effective_multilingual_version: Optional[str] = None

# Voice conditioning cache: avoids re-encoding the same voice file on every request.
# Key: (resolved_path, file_mtime, exaggeration) — mtime invalidates if file changes.
_conds_cache: dict = {}


def _conds_cache_key(path: str, exaggeration: float) -> tuple:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return (path, mtime, exaggeration)


def set_seed(seed_value: int):
    """
    Sets the seed for torch, random, and numpy for reproducibility.
    This is called if a non-zero seed is provided for generation.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    logger.info(f"Global seed set to: {seed_value}")


def _test_cuda_functionality() -> bool:
    """
    Tests if CUDA is actually functional, not just available.

    Returns:
        bool: True if CUDA works, False otherwise.
    """
    if not torch.cuda.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.cuda()
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"CUDA functionality test failed: {e}")
        return False


def _test_mps_functionality() -> bool:
    """
    Tests if MPS is actually functional, not just available.

    Returns:
        bool: True if MPS works, False otherwise.
    """
    if not torch.backends.mps.is_available():
        return False

    try:
        test_tensor = torch.tensor([1.0])
        test_tensor = test_tensor.to("mps")
        test_tensor = test_tensor.cpu()
        return True
    except Exception as e:
        logger.warning(f"MPS functionality test failed: {e}")
        return False


def _get_model_class(selector: str) -> tuple:
    """
    Determines which model class to use based on the config selector value.

    Args:
        selector: The value from config model.repo_id

    Returns:
        Tuple of (model_class, model_type_string)

    Raises:
        ImportError: If Turbo or Multilingual is selected but not available in the package
    """
    selector_normalized = selector.lower().strip()
    model_type = MODEL_SELECTOR_MAP.get(selector_normalized)

    if model_type == "turbo":
        if not TURBO_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxTurboTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Turbo model (ChatterboxTurboTTS)"
        )
        return ChatterboxTurboTTS, "turbo"

    if model_type == "multilingual":
        if not MULTILINGUAL_AVAILABLE:
            raise ImportError(
                f"Model selector '{selector}' requires ChatterboxMultilingualTTS, "
                f"but it is not available in the installed chatterbox package. "
                f"Please update the chatterbox-tts package to the latest version, "
                f"or use 'chatterbox' to select the original model."
            )
        logger.info(
            f"Model selector '{selector}' resolved to Multilingual model (ChatterboxMultilingualTTS)"
        )
        return ChatterboxMultilingualTTS, "multilingual"

    if model_type == "original":
        logger.info(
            f"Model selector '{selector}' resolved to Original model (ChatterboxTTS)"
        )
        return ChatterboxTTS, "original"

    # Unknown selector - default to original with warning
    logger.warning(
        f"Unknown model selector '{selector}'. "
        f"Valid values: chatterbox, chatterbox-turbo, chatterbox-multilingual, original, turbo, multilingual, "
        f"ResembleAI/chatterbox, ResembleAI/chatterbox-turbo. "
        f"Defaulting to original ChatterboxTTS model."
    )
    return ChatterboxTTS, "original"


def get_model_info() -> dict:
    """
    Returns information about the currently loaded model.
    Used by the API to expose model details to the UI.

    Returns:
        Dictionary containing model information
    """
    return {
        "loaded": MODEL_LOADED,
        "type": loaded_model_type,  # "original", "turbo", or "multilingual"
        "class_name": loaded_model_class_name,
        "device": model_device,
        "sample_rate": chatterbox_model.sr if chatterbox_model else None,
        "supports_paralinguistic_tags": loaded_model_type == "turbo",
        "available_paralinguistic_tags": (
            TURBO_PARALINGUISTIC_TAGS if loaded_model_type == "turbo" else []
        ),
        "turbo_available_in_package": TURBO_AVAILABLE,
        "multilingual_available_in_package": MULTILINGUAL_AVAILABLE,
        "supports_multilingual": loaded_model_type == "multilingual",
        "supported_languages": (
            SUPPORTED_LANGUAGES if loaded_model_type == "multilingual" else {"en": "English"}
        ),
        # Requested version (from config) and the version actually loaded.
        "multilingual_t3_version": config_manager.get_string(
            "model.multilingual_t3_version", "v3"
        ),
        "multilingual_t3_version_loaded": (
            effective_multilingual_version if loaded_model_type == "multilingual" else None
        ),
    }


def load_model() -> bool:
    """
    Loads the TTS model.
    This version directly attempts to load from the Hugging Face repository (or its cache)
    using `from_pretrained`, bypassing the local `paths.model_cache` directory.
    Updates global variables `chatterbox_model`, `MODEL_LOADED`, and `model_device`.

    Returns:
        bool: True if the model was loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device
    global loaded_model_type, loaded_model_class_name, effective_multilingual_version

    if MODEL_LOADED:
        logger.info("TTS model is already loaded.")
        return True

    try:
        # Determine processing device with robust CUDA detection and intelligent fallback
        device_setting = config_manager.get_string("tts_engine.device", "auto")

        if device_setting == "auto":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA functionality test passed. Using CUDA.")
            elif _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS functionality test passed. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.info("CUDA and MPS not functional or not available. Using CPU.")

        elif device_setting == "cuda":
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
                logger.info("CUDA requested and functional. Using CUDA.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "CUDA was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with CUDA support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "mps":
            if _test_mps_functionality():
                resolved_device_str = "mps"
                logger.info("MPS requested and functional. Using MPS.")
            else:
                resolved_device_str = "cpu"
                logger.warning(
                    "MPS was requested in config but functionality test failed. "
                    "PyTorch may not be compiled with MPS support. "
                    "Automatically falling back to CPU."
                )

        elif device_setting == "cpu":
            resolved_device_str = "cpu"
            logger.info("CPU device explicitly requested in config. Using CPU.")

        else:
            logger.warning(
                f"Invalid device setting '{device_setting}' in config. "
                f"Defaulting to auto-detection."
            )
            if _test_cuda_functionality():
                resolved_device_str = "cuda"
            elif _test_mps_functionality():
                resolved_device_str = "mps"
            else:
                resolved_device_str = "cpu"
            logger.info(f"Auto-detection resolved to: {resolved_device_str}")

        model_device = resolved_device_str
        logger.info(f"Final device selection: {model_device}")
        logger.info(
            f"BF16 optimization: {'enabled' if BF16_ENABLED else 'disabled'} "
            f"(TTS_BF16={os.environ.get('TTS_BF16', 'off')})"
        )

        # Get the model selector from config
        model_selector = config_manager.get_string("model.repo_id", "chatterbox-turbo")

        logger.info(f"Model selector from config: '{model_selector}'")

        try:
            # Determine which model class to use
            model_class, model_type = _get_model_class(model_selector)

            logger.info(
                f"Initializing {model_class.__name__} on device '{model_device}'..."
            )
            logger.info(f"Model type: {model_type}")
            if model_type == "turbo":
                logger.info(
                    f"Turbo model supports paralinguistic tags: {TURBO_PARALINGUISTIC_TAGS}"
                )

            # Load the model using from_pretrained - handles HuggingFace downloads automatically.
            # Multilingual supports selectable T3 checkpoint versions (v2/v3); turbo and
            # original take no version argument.
            if model_type == "multilingual":
                ml_version = config_manager.get_string(
                    "model.multilingual_t3_version", "v3"
                )
                # Only pass t3_model if the installed chatterbox package supports it.
                # Older releases (e.g. chatterbox-tts 0.1.6) have from_pretrained(device)
                # with no version argument and only ship the v2 checkpoint.
                supports_version = (
                    "t3_model" in inspect.signature(model_class.from_pretrained).parameters
                )
                if supports_version:
                    logger.info(f"Loading multilingual T3 weights version: '{ml_version}'")
                    chatterbox_model = model_class.from_pretrained(
                        device=model_device, t3_model=ml_version
                    )
                    effective_multilingual_version = ml_version
                else:
                    logger.warning(
                        "Installed chatterbox package does not support multilingual T3 "
                        f"version selection; ignoring requested '{ml_version}' and loading "
                        "the package default (v2). Upgrade the chatterbox package to use v3."
                    )
                    chatterbox_model = model_class.from_pretrained(device=model_device)
                    effective_multilingual_version = "v2 (package default)"
            else:
                chatterbox_model = model_class.from_pretrained(device=model_device)

            # Convert T3 to bfloat16 if enabled.
            # Token generation is memory-bandwidth bound; bf16 halves bytes read per
            # forward pass. S3Gen is intentionally kept in float32 — it runs only
            # 2 CFM timesteps and bf16 causes token/mask size mismatches.
            if BF16_ENABLED:
                if hasattr(chatterbox_model, "t3"):
                    chatterbox_model.t3 = chatterbox_model.t3.bfloat16()
                    logger.info("T3 model converted to bfloat16 for faster token generation.")
            else:
                logger.info("BF16 optimization disabled (TTS_BF16=off or hardware unsupported).")

            # Store model metadata
            loaded_model_type = model_type
            loaded_model_class_name = model_class.__name__

            logger.info(f"Successfully loaded {model_class.__name__} on {model_device}")
            logger.info(f"Model sample rate: {chatterbox_model.sr} Hz")
        except ImportError as e_import:
            logger.error(
                f"Failed to load model due to import error: {e_import}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False
        except Exception as e_hf:
            logger.error(
                f"Failed to load model using from_pretrained: {e_hf}",
                exc_info=True,
            )
            chatterbox_model = None
            MODEL_LOADED = False
            return False

        MODEL_LOADED = True
        if chatterbox_model:
            logger.info(
                f"TTS Model loaded successfully on {model_device}. Engine sample rate: {chatterbox_model.sr} Hz."
            )
        else:
            logger.error(
                "Model loading sequence completed, but chatterbox_model is None. This indicates an unexpected issue."
            )
            MODEL_LOADED = False
            return False

        return True

    except Exception as e:
        logger.error(
            f"An unexpected error occurred during model loading: {e}", exc_info=True
        )
        chatterbox_model = None
        MODEL_LOADED = False
        return False


def synthesize(
    text: str,
    audio_prompt_path: Optional[str] = None,
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    seed: int = 0,
    language: str = "en",
    top_p: float = 1.0,
    top_k: int = 1000,
    repetition_penalty: float = 1.2,
) -> Tuple[Optional[torch.Tensor], Optional[int]]:
    """
    Synthesizes audio from text using the loaded TTS model.

    Args:
        text: The text to synthesize.
        audio_prompt_path: Path to an audio file for voice cloning or predefined voice.
        temperature: Controls randomness in generation.
        exaggeration: Controls expressiveness. (Ignored by the Turbo model.)
        cfg_weight: Classifier-Free Guidance weight. (Ignored by the Turbo model.)
        seed: Random seed for generation. If 0, default randomness is used.
              If non-zero, a global seed is set for reproducibility.
        language: Language code for multilingual model (e.g., 'en', 'it', 'de').
        top_p: Nucleus sampling threshold. Lower = tighter/safer delivery,
               higher (->1.0) = more varied, expressive, riskier.
        top_k: Keeps only the k most-likely acoustic tokens each step.
               Turbo-only (the original/multilingual models do not accept it).
        repetition_penalty: Penalizes already-emitted acoustic tokens; guards
               against stutters/looping. ~1.2 sweet spot; too high distorts.

    Returns:
        A tuple containing the audio waveform (torch.Tensor) and the sample rate (int),
        or (None, None) if synthesis fails.
    """
    global chatterbox_model

    if not MODEL_LOADED or chatterbox_model is None:
        logger.error("TTS model is not loaded. Cannot synthesize audio.")
        return None, None

    try:
        # Set seed globally if a specific seed value is provided and is non-zero.
        if seed != 0:
            logger.info(f"Applying user-provided seed for generation: {seed}")
            set_seed(seed)
        else:
            logger.info(
                "Using default (potentially random) generation behavior as seed is 0."
            )

        logger.debug(
            f"Synthesizing with params: audio_prompt='{audio_prompt_path}', temp={temperature}, "
            f"exag={exaggeration}, cfg_weight={cfg_weight}, seed_applied_globally_if_nonzero={seed}, "
            f"language={language}"
        )

        # Voice conditioning cache: skip re-encoding the same voice file.
        # Turbo ignores exaggeration in conds; others include it in the key.
        effective_prompt = audio_prompt_path
        conds_key = None
        if audio_prompt_path and hasattr(chatterbox_model, "conds"):
            ex_for_key = 0.0 if loaded_model_type == "turbo" else exaggeration
            conds_key = _conds_cache_key(audio_prompt_path, ex_for_key)
            if conds_key in _conds_cache:
                chatterbox_model.conds = _conds_cache[conds_key]
                effective_prompt = None  # conds already set, skip prepare_conditionals
                logger.debug(f"Voice cache hit: {audio_prompt_path}")

        # Call the core model's generate method.
        # autocast promotes float32 inputs to bfloat16 to match T3/S3Gen weights,
        # keeping numerically sensitive ops (softmax, norms) in float32 automatically.
        # Common sampling kwargs accepted by all model variants.
        gen_kwargs = dict(
            audio_prompt_path=effective_prompt,
            temperature=temperature,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        # top_k is only accepted by the Turbo model's generate(); passing it to
        # the original/multilingual models raises a TypeError.
        if loaded_model_type == "turbo":
            gen_kwargs["top_k"] = top_k

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=BF16_ENABLED):
            if loaded_model_type == "multilingual":
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    language_id=language,
                    **gen_kwargs,
                )
            else:
                wav_tensor = chatterbox_model.generate(
                    text=text,
                    **gen_kwargs,
                )

        # Store conds in cache after first compute for this voice.
        if conds_key is not None and effective_prompt is not None:
            if chatterbox_model.conds is not None:
                _conds_cache[conds_key] = chatterbox_model.conds
                logger.debug(f"Cached voice conditionals for: {audio_prompt_path}")

        # The ChatterboxTTS.generate method already returns a CPU tensor.
        return wav_tensor, chatterbox_model.sr

    except Exception as e:
        logger.error(f"Error during TTS synthesis: {e}", exc_info=True)
        return None, None


def unload_model() -> bool:
    """
    Unloads the current model and releases all GPU memory.
    Does NOT reload the model - use reload_model() for that.

    Returns:
        bool: True if the model was unloaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name

    logger.info("Initiating model unload sequence...")

    # 1. Unload existing model
    if chatterbox_model is not None:
        logger.info("Unloading TTS model from memory...")
        del chatterbox_model
        chatterbox_model = None

    # 2. Reset state flags
    MODEL_LOADED = False
    model_device = None
    loaded_model_type = None
    loaded_model_class_name = None

    # 3. Force Python Garbage Collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # 4. Clear GPU Cache (CUDA)
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    # 5. Clear GPU Cache (MPS - Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache.")
        except AttributeError:
            logger.debug(
                "torch.mps.empty_cache() not available in this PyTorch version."
            )

    logger.info("Model unloaded and GPU memory released.")
    return True


def reload_model() -> bool:
    """
    Unloads the current model, clears GPU memory, and reloads the model
    based on the current configuration. Used for hot-swapping models
    without restarting the server process.

    Returns:
        bool: True if the new model loaded successfully, False otherwise.
    """
    global chatterbox_model, MODEL_LOADED, model_device, loaded_model_type, loaded_model_class_name, _conds_cache

    logger.info("Initiating model hot-swap/reload sequence...")

    # 1. Unload existing model
    if chatterbox_model is not None:
        logger.info("Unloading existing TTS model from memory...")
        del chatterbox_model
        chatterbox_model = None

    # 2. Reset state flags and clear voice cache (conds are model-specific)
    MODEL_LOADED = False
    loaded_model_type = None
    loaded_model_class_name = None
    _conds_cache.clear()
    logger.info("Voice conditioning cache cleared.")

    # 3. Force Python Garbage Collection
    gc.collect()
    logger.info("Python garbage collection completed.")

    # 4. Clear GPU Cache (CUDA)
    if torch.cuda.is_available():
        logger.info("Clearing CUDA cache...")
        torch.cuda.empty_cache()

    # 5. Clear GPU Cache (MPS - Apple Silicon)
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            logger.info("Cleared MPS cache.")
        except AttributeError:
            # Older PyTorch versions may not have mps.empty_cache()
            logger.debug(
                "torch.mps.empty_cache() not available in this PyTorch version."
            )

    # 6. Reload model from the (now updated) configuration
    logger.info("Memory cleared. Reloading model from updated config...")
    return load_model()


# --- End File: engine.py ---
