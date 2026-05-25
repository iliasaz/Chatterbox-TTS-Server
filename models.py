# File: models.py
# Pydantic models for API request and response validation.

from typing import Optional, Literal
from pydantic import BaseModel, Field


class GenerationParams(BaseModel):
    """Common parameters for TTS generation."""

    temperature: Optional[float] = Field(
        None,  # Defaulting to None means server will use config default if not provided
        ge=0.0,
        le=1.5,  # Based on Chatterbox Gradio app for temperature
        description="Controls randomness. Lower is more deterministic. (Range: 0.0-1.5)",
    )
    exaggeration: Optional[float] = Field(
        None,
        ge=0.25,  # Based on Chatterbox Gradio app
        le=2.0,  # Based on Chatterbox Gradio app
        description="Controls expressiveness/exaggeration. (Range: 0.25-2.0)",
    )
    cfg_weight: Optional[float] = Field(
        None,
        ge=0.2,  # Based on Chatterbox Gradio app
        le=1.0,  # Based on Chatterbox Gradio app
        description="Classifier-Free Guidance weight. Influences adherence to prompt/style and pacing. (Range: 0.2-1.0)",
    )
    seed: Optional[int] = Field(
        None,
        ge=0,  # Seed should be non-negative, 0 often implies random.
        description="Seed for generation. 0 may indicate random behavior based on engine.",
    )
    speed_factor: Optional[float] = Field(
        None,
        ge=0.25,
        le=4.0,
        description="DEPRECATED and ignored: post-processing time-stretch introduced audio "
        "artifacts, so it is no longer applied. Field kept for backward compatibility.",
    )
    language: Optional[str] = Field(
        None,
        description="Language of the text. (Primarily for UI, actual engine may infer)",
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold over acoustic tokens. Lower (~0.8) gives "
        "tighter, more stable, less expressive delivery; higher (->1.0) gives more varied "
        "and expressive prosody with greater risk of artifacts. (Range: 0.0-1.0)",
    )
    top_k: Optional[int] = Field(
        None,
        ge=0,
        le=2000,
        description="Keeps only the k most-likely acoustic tokens at each step. Lower "
        "(~50-100) is cleaner/more conservative; higher adds variety. Turbo model only; "
        "ignored by the original/multilingual models. (Range: 0-2000)",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        ge=1.0,
        le=2.0,
        description="Penalizes already-generated acoustic tokens to prevent stutters, "
        "looping and droning. ~1.2 is the sweet spot; values above ~1.5 can distort pitch "
        "and rush/clip delivery. 1.0 disables it. (Range: 1.0-2.0)",
    )


class CustomTTSRequest(BaseModel):
    """Request model for the custom /tts endpoint."""

    text: str = Field(..., min_length=1, description="Text to be synthesized.")

    voice_mode: Literal["predefined", "clone"] = Field(
        "predefined",  # Default voice mode
        description="Voice mode: 'predefined' for a built-in voice, 'clone' for voice cloning using a reference audio.",
    )
    predefined_voice_id: Optional[str] = Field(
        None,
        description="Filename of the predefined voice to use (e.g., 'default_sample.wav'). Required if voice_mode is 'predefined'.",
    )
    reference_audio_filename: Optional[str] = Field(
        None,
        description="Filename of a user-uploaded reference audio for voice cloning. Required if voice_mode is 'clone'.",
    )

    output_format: Optional[Literal["wav", "opus", "mp3"]] = Field(  # Added "mp3"
        "wav", description="Desired audio output format."  # Default output format
    )

    split_text: Optional[bool] = Field(
        True,  # Default to splitting enabled
        description="Whether to automatically split long text into chunks for processing.",
    )
    chunk_size: Optional[int] = Field(
        120,  # Default target chunk size from config
        ge=50,  # Minimum reasonable chunk size
        le=500,  # Maximum reasonable chunk size
        description="Approximate target character length for text chunks when splitting is enabled (50-500).",
    )

    # Embed generation parameters directly
    temperature: Optional[float] = Field(
        None, description="Overrides default temperature if provided."
    )
    exaggeration: Optional[float] = Field(
        None, description="Overrides default exaggeration if provided."
    )
    cfg_weight: Optional[float] = Field(
        None, description="Overrides default CFG weight if provided."
    )
    seed: Optional[int] = Field(None, description="Overrides default seed if provided.")
    speed_factor: Optional[float] = Field(
        None,
        description="DEPRECATED and ignored (caused audio artifacts). Accepted for backward "
        "compatibility but has no effect.",
    )
    language: Optional[str] = Field(
        None, description="Overrides default language if provided."
    )
    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Overrides default nucleus sampling (top_p) if provided. Lower = tighter/"
        "more stable; higher (->1.0) = more expressive but riskier.",
    )
    top_k: Optional[int] = Field(
        None,
        ge=0,
        le=2000,
        description="Overrides default top_k if provided. Lower = cleaner/more conservative; "
        "higher = more variety. Turbo model only.",
    )
    repetition_penalty: Optional[float] = Field(
        None,
        ge=1.0,
        le=2.0,
        description="Overrides default repetition penalty if provided. ~1.2 prevents stutters/"
        "loops; >1.5 can distort.",
    )

    stream: bool = Field(
        False,
        description="If true, returns a StreamingResponse with WAV audio yielded as each chunk is synthesized. output_format is ignored when streaming.",
    )


class SavePredefinedVoiceRequest(BaseModel):
    """Request model for promoting a reference clip into a named predefined voice."""

    reference_filename: str = Field(
        ...,
        min_length=1,
        description="Filename of an existing reference audio file (in the reference_audio "
        "directory) to copy into the persistent predefined voices set.",
    )
    voice_name: str = Field(
        ...,
        min_length=1,
        description="Display/base name for the saved predefined voice. The file is stored "
        "under this name (sanitized) with the source file's extension.",
    )


class ErrorResponse(BaseModel):
    """Standard error response model for API errors."""

    detail: str = Field(..., description="A human-readable explanation of the error.")


class UpdateStatusResponse(BaseModel):
    """Response model for status updates, e.g., after saving settings."""

    message: str = Field(
        ..., description="A message describing the result of the operation."
    )
    restart_needed: Optional[bool] = Field(
        False,
        description="Indicates if a server restart is recommended or required for changes to take full effect.",
    )
