"""Microphone audio capture using sounddevice.

Provides push-to-talk style recording: call ``start()`` to begin capturing
audio, then ``stop()`` to get the recorded audio as a NumPy array and WAV bytes.

The recorder queries the system's default input device for its native sample
rate and channel count, records at those settings, and then resamples /
down-mixes to 16 kHz mono on ``stop()`` — the format Whisper expects.

For live transcription, ``peek()`` returns a non-destructive snapshot of the
audio captured so far, and ``recent_rms()`` provides energy levels for
voice-activity detection.

The :class:`ContinuousVAD` class monitors the microphone continuously and
auto-detects speech start/end using energy-based VAD.  When a speech segment
is detected, it records it, and when the speech ends (silence timeout),
it emits a callback with the recorded clip.  This is used for hands-free
voice input and voice-interrupt of the agent.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import os
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Audio defaults — Whisper's expected input format.
WHISPER_SAMPLE_RATE = 16_000  # 16 kHz
DEFAULT_SAMPLE_RATE = WHISPER_SAMPLE_RATE
DEFAULT_CHANNELS = 1  # mono
DEFAULT_DTYPE = "int16"

# Device names to skip during auto-discovery.  These are audio-server virtual
# devices that either duplicate real hardware or don't produce real audio.
_SKIP_DEVICE_NAMES = {"default", "jack", "pipewire", "pulse"}
_SKIP_DEVICE_SUBSTRINGS = {"dummy", "speech-dispatcher"}


@contextlib.contextmanager
def _suppress_stderr():
    """Temporarily redirect fd 2 to /dev/null to suppress C-library noise.

    PortAudio / ALSA write error messages directly to stderr (bypassing
    Python's logging).  This context manager silences them during device
    probing so the user isn't flooded with ALSA diagnostics.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
    except OSError:
        yield  # can't redirect — just run unsilenced
        return
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


@dataclass(slots=True)
class AudioClip:
    """A recorded audio clip."""

    samples: np.ndarray
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return len(self.samples) / self.sample_rate

    def to_wav_bytes(self) -> bytes:
        """Encode the clip as a WAV file in memory."""
        import numpy as np

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.samples.astype(np.int16).tobytes())
        return buf.getvalue()


def _query_input_device(device: int | str | None = None) -> tuple[int, int]:
    """Query an input device for its native sample rate and channels.

    Returns ``(sample_rate, channels)``.
    """
    import sounddevice as sd

    info = sd.query_devices(device=device, kind="input")
    rate = int(info["default_samplerate"])  # type: ignore[index]
    channels = int(info["max_input_channels"])  # type: ignore[index]
    # Clamp channels to a sensible max (we'll downmix to mono anyway).
    channels = min(channels, 2)
    return rate, channels


def _resample(samples: "np.ndarray", orig_rate: int, target_rate: int) -> "np.ndarray":
    """Resample audio from *orig_rate* to *target_rate* using linear interpolation.

    This is a lightweight fallback.  For higher quality, scipy or soxr would
    be preferable, but we avoid the hard dependency.
    """
    import numpy as np

    if orig_rate == target_rate:
        return samples
    duration = len(samples) / orig_rate
    target_len = int(duration * target_rate)
    indices = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(
        indices, np.arange(len(samples)), samples.astype(np.float64)
    ).astype(samples.dtype)


def _downmix_to_mono(raw: "np.ndarray") -> "np.ndarray":
    """Down-mix multi-channel audio to mono, returning a 1-D array."""
    if raw.ndim == 2 and raw.shape[1] > 1:
        return raw.mean(axis=1).astype(raw.dtype)
    return raw.flatten()


class MicRecorder:
    """Records audio from the default microphone.

    On ``start()`` the recorder queries the system's default input device for
    its native sample rate and channel count, and opens a stream with those
    parameters (avoiding ALSA/JACK/PipeWire format mismatch errors).

    On ``stop()`` the captured audio is down-mixed to mono and resampled to
    16 kHz — the format expected by Whisper.

    Usage::

        recorder = MicRecorder()
        recorder.start()
        # ... user speaks ...
        clip = recorder.stop()
        wav_bytes = clip.to_wav_bytes()

    Parameters
    ----------
    device:
        Sounddevice input device index or name.  ``None`` = system default.
    """

    def __init__(self, device: int | str | None = None) -> None:
        from leuk.voice import require_voice

        require_voice()

        self.device = device
        # These are filled at start() from the device query.
        self._device_rate: int = 0
        self._device_channels: int = 0
        self._frames: list["np.ndarray"] = []
        self._stream: object | None = None
        self._recording = False

    # Expose constants for tests / callers that need them.
    sample_rate: int = WHISPER_SAMPLE_RATE
    channels: int = DEFAULT_CHANNELS

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Start capturing audio from the microphone.

        If the user specified a device, it is used directly.  Otherwise the
        system default is tried first; if that fails, every input device is
        attempted in order (skipping known virtual/meta devices) until one
        succeeds.

        Raises ``RuntimeError`` if no device can be opened.
        """
        import numpy as np
        import sounddevice as sd

        if self._recording:
            logger.warning("Already recording")
            return

        self._frames.clear()

        def _callback(
            indata: np.ndarray, frames: int, time_info: object, status: object
        ) -> None:
            if status:
                logger.debug("sounddevice status: %s", status)
            self._frames.append(indata.copy())

        # Build a list of devices to try.
        if self.device is not None:
            candidates = [self.device]
        else:
            # Default first, then every real input device.
            candidates: list[int | str | None] = [None]
            for idx, info in enumerate(sd.query_devices()):
                if info["max_input_channels"] <= 0:
                    continue
                name_lower = info.get("name", "").lower()
                if name_lower in _SKIP_DEVICE_NAMES:
                    continue
                if any(sub in name_lower for sub in _SKIP_DEVICE_SUBSTRINGS):
                    continue
                candidates.append(idx)

        last_err: Exception | None = None
        for dev in candidates:
            with _suppress_stderr():
                try:
                    rate, channels = _query_input_device(dev)
                except sd.PortAudioError as exc:
                    last_err = exc
                    continue

                try:
                    self._frames.clear()
                    self._stream = sd.InputStream(
                        device=dev,
                        samplerate=rate,
                        channels=channels,
                        dtype=DEFAULT_DTYPE,
                        callback=_callback,
                    )
                    self._stream.start()  # type: ignore[union-attr]
                except sd.PortAudioError as exc:
                    self._stream = None
                    last_err = exc
                    dev_name = dev if dev is not None else "default"
                    logger.debug("Device %s failed: %s", dev_name, exc)
                    continue

                # Verify the device produces non-zero audio.  Dead inputs
                # (e.g. unconnected ALSA hw: ports) open fine but emit only
                # zeros.  Wait up to 500 ms for at least one non-zero sample.
                deadline = time.monotonic() + 0.5
                while time.monotonic() < deadline:
                    if self._frames and any(np.any(f != 0) for f in self._frames):
                        break
                    time.sleep(0.05)
                else:
                    # No non-zero audio within the window → dead device.
                    self._stream.stop()  # type: ignore[union-attr]
                    self._stream.close()  # type: ignore[union-attr]
                    self._stream = None
                    self._frames.clear()
                    dev_name = dev if dev is not None else "default"
                    logger.debug("Device %s produced silence, skipping", dev_name)
                    continue

            # Success — the device is alive.
            self._device_rate = rate
            self._device_channels = channels
            self._recording = True
            dev_name = dev if dev is not None else "default"
            dev_info = sd.query_devices(dev, "input") if dev is not None else {}
            logger.info(
                "Recording on device %s (%s) rate=%d ch=%d",
                dev_name,
                dev_info.get("name", "default"),  # type: ignore[union-attr]
                rate,
                channels,
            )
            return

        raise RuntimeError(
            f"Cannot open any audio input device (last error: {last_err})\n"
            "Run the following to list devices:\n"
            '  python -c "import sounddevice; print(sounddevice.query_devices())"\n'
            "Then pass a working device index: MicRecorder(device=N)"
        )

    def peek(self) -> AudioClip:
        """Return a snapshot of audio captured so far *without* stopping.

        The returned clip is down-mixed and resampled to 16 kHz mono, just
        like ``stop()`` would produce.  Use this for live transcription.
        """
        import numpy as np

        if not self._recording:
            raise RuntimeError("Not currently recording")

        if not self._frames:
            return AudioClip(samples=np.array([], dtype=np.int16))

        raw = np.concatenate(self._frames, axis=0)
        mono = _downmix_to_mono(raw)
        samples = _resample(mono, self._device_rate, WHISPER_SAMPLE_RATE)
        return AudioClip(samples=samples, sample_rate=WHISPER_SAMPLE_RATE, channels=1)

    def recent_rms(self, seconds: float = 0.5) -> float:
        """Return the RMS energy of the most recent *seconds* of raw audio.

        This operates on raw device samples (before resampling) so it's
        cheap to call frequently for voice-activity detection.  Returns 0.0
        if no audio has been captured yet.
        """
        import numpy as np

        if not self._frames:
            return 0.0

        needed = int(seconds * self._device_rate)
        # Collect enough frames from the tail.
        collected: list[np.ndarray] = []
        total = 0
        for frame in reversed(self._frames):
            collected.append(frame)
            total += len(frame)
            if total >= needed:
                break
        collected.reverse()
        block = np.concatenate(collected, axis=0)[-needed:]

        # Down-mix to 1-D for RMS calculation.
        if block.ndim == 2:
            block = block.mean(axis=1)
        return float(math.sqrt(np.mean(block.astype(np.float64) ** 2)))

    def stop(self) -> AudioClip:
        """Stop recording and return the captured audio.

        The raw device audio is down-mixed to mono and resampled to 16 kHz.
        """
        import numpy as np

        if not self._recording or self._stream is None:
            raise RuntimeError("Not currently recording")

        self._stream.stop()  # type: ignore[union-attr]
        self._stream.close()  # type: ignore[union-attr]
        self._stream = None
        self._recording = False

        if not self._frames:
            samples = np.array([], dtype=np.int16)
        else:
            raw = np.concatenate(self._frames, axis=0)
            mono = _downmix_to_mono(raw)
            samples = _resample(mono, self._device_rate, WHISPER_SAMPLE_RATE)

        clip = AudioClip(
            samples=samples,
            sample_rate=WHISPER_SAMPLE_RATE,
            channels=1,
        )
        logger.debug("Recording stopped — %.1f seconds captured", clip.duration)
        return clip

    def cancel(self) -> None:
        """Cancel recording without returning audio."""
        if self._recording and self._stream is not None:
            self._stream.stop()  # type: ignore[union-attr]
            self._stream.close()  # type: ignore[union-attr]
            self._stream = None
            self._recording = False
            self._frames.clear()
            logger.debug("Recording cancelled")


# ── VAD tuning defaults ──────────────────────────────────────────

VAD_SILENCE_TIMEOUT = 1.0  # seconds of silence after speech → segment end
VAD_MIN_SPEECH_DURATION = 0.5  # ignore segments shorter than this
VAD_POLL_INTERVAL = 0.05  # seconds between energy polls

# Silero VAD processes audio in fixed-size chunks.
_SILERO_VAD_CHUNK_SAMPLES = 512  # 32 ms at 16 kHz


def _load_silero_vad() -> tuple[object, Callable]:
    """Load the Silero VAD model and ``get_speech_timestamps`` helper.

    Returns ``(model, get_speech_timestamps)``.
    """
    import torch

    _root_level = logging.getLogger().getEffectiveLevel()
    if _root_level > logging.DEBUG:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    logger.info("Loading Silero VAD model …")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]
    logger.info("Silero VAD ready")
    return model, get_speech_timestamps


def trim_silence(clip: AudioClip, vad_model: object, get_ts: Callable,
                 *, threshold: float = 0.4) -> AudioClip | None:
    """Remove non-speech portions from *clip* using Silero VAD.

    Returns a new :class:`AudioClip` containing only the speech segments
    concatenated together, or ``None`` if no speech was found.

    Parameters
    ----------
    clip:
        Audio clip to trim (16 kHz int16 mono).
    vad_model:
        Loaded Silero VAD model.
    get_ts:
        ``get_speech_timestamps`` function from Silero utils.
    threshold:
        Speech detection threshold (0–1).  Lower = more sensitive.
    """
    import numpy as np
    import torch

    audio_f32 = clip.samples.astype(np.float32) / 32768.0
    tensor = torch.from_numpy(audio_f32)

    timestamps = get_ts(
        tensor,
        vad_model,
        sampling_rate=WHISPER_SAMPLE_RATE,
        threshold=threshold,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
    )

    if not timestamps:
        logger.debug("trim_silence: no speech segments found")
        return None

    # Concatenate speech segments
    parts = []
    for ts in timestamps:
        parts.append(clip.samples[ts["start"]:ts["end"]])

    trimmed = np.concatenate(parts)

    # Reset VAD model state after full-clip analysis
    if hasattr(vad_model, "reset_states"):
        vad_model.reset_states()

    speech_dur = len(trimmed) / WHISPER_SAMPLE_RATE
    orig_dur = clip.duration
    logger.debug(
        "trim_silence: %.1fs → %.1fs (%d segments, %.0f%% kept)",
        orig_dur, speech_dur, len(timestamps),
        100 * speech_dur / orig_dur if orig_dur > 0 else 0,
    )
    return AudioClip(samples=trimmed, sample_rate=WHISPER_SAMPLE_RATE, channels=1)


class ContinuousVAD:
    """Continuous voice-activity detector using Silero VAD + :class:`MicRecorder`.

    Uses a neural Silero VAD model to detect speech, instead of simple
    RMS energy thresholding.  This eliminates Whisper hallucinations on
    silence, keyboard noise, breathing, etc.

    When a speech segment is detected and followed by enough silence,
    the captured clip is delivered via the ``on_speech`` callback.

    Parameters
    ----------
    recorder:
        A :class:`MicRecorder` (not yet started).
    on_speech:
        Async callback ``(clip: AudioClip) -> None`` called when a
        complete speech segment is captured.
    sensitivity:
        VAD sensitivity 0.0–1.0.  Higher values detect quieter speech
        but may trigger on non-speech sounds.  Default 0.5.
        Maps to Silero's ``threshold`` parameter (inverted: high
        sensitivity = low threshold).
    silence_timeout:
        Seconds of continuous silence after speech to trigger segment end.
    min_duration:
        Minimum speech duration (seconds) to emit.  Shorter segments are
        discarded as noise.

    Usage::

        async def handle(clip: AudioClip) -> None:
            text = await stt.transcribe(clip)
            ...

        vad = ContinuousVAD(MicRecorder(), on_speech=handle)
        vad.start()
        ...
        await vad.stop()
    """

    def __init__(
        self,
        recorder: MicRecorder,
        *,
        on_speech: Callable[["AudioClip"], object],
        sensitivity: float = 0.5,
        silence_timeout: float = VAD_SILENCE_TIMEOUT,
        min_duration: float = VAD_MIN_SPEECH_DURATION,
    ) -> None:
        self._recorder = recorder
        self._on_speech = on_speech
        # Sensitivity 0–1 maps to Silero threshold inversely:
        # sensitivity=1.0 → threshold=0.15 (very sensitive)
        # sensitivity=0.5 → threshold=0.40 (balanced)
        # sensitivity=0.0 → threshold=0.65 (very strict)
        self._vad_threshold = 0.65 - 0.5 * max(0.0, min(1.0, sensitivity))
        self._silence_timeout = silence_timeout
        self._min_duration = min_duration
        self._task: asyncio.Task[None] | None = None
        self._active = False
        self._paused = False
        self._vad_model: object | None = None
        self._get_speech_ts: Callable | None = None

    def _ensure_vad(self) -> object:
        """Lazy-load the Silero VAD model."""
        if self._vad_model is None:
            self._vad_model, self._get_speech_ts = _load_silero_vad()
        return self._vad_model

    @property
    def vad_model(self) -> object | None:
        """The loaded Silero VAD model (or None if not yet loaded)."""
        return self._vad_model

    @property
    def get_speech_timestamps(self) -> Callable | None:
        """The ``get_speech_timestamps`` utility (or None)."""
        return self._get_speech_ts

    @property
    def vad_threshold(self) -> float:
        """The current VAD detection threshold."""
        return self._vad_threshold

    def _is_speech(self, samples: "np.ndarray") -> bool:
        """Run Silero VAD on the most recent audio chunk.

        Takes 512-sample (32 ms) chunks of 16 kHz float32 audio and
        returns True if the model's speech probability exceeds the
        threshold.
        """
        import numpy as np
        import torch

        model = self._ensure_vad()

        if len(samples) < _SILERO_VAD_CHUNK_SAMPLES:
            return False

        # Take the last chunk
        chunk = samples[-_SILERO_VAD_CHUNK_SAMPLES:]
        audio_f32 = chunk.astype(np.float32) / 32768.0
        tensor = torch.from_numpy(audio_f32)

        prob = model(tensor, WHISPER_SAMPLE_RATE).item()  # type: ignore[operator]
        return prob >= self._vad_threshold

    @property
    def active(self) -> bool:
        """Whether the VAD monitor is running."""
        return self._active and self._task is not None and not self._task.done()

    def start(self) -> None:
        """Start continuous monitoring in a background task."""
        if self._task is not None and not self._task.done():
            return
        self._active = True
        self._task = asyncio.create_task(self._loop(), name="continuous-vad")

    async def stop(self) -> None:
        """Stop the background monitoring task."""
        self._active = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Ensure recorder is cleaned up
        if self._recorder.is_recording:
            self._recorder.cancel()

    def pause(self) -> None:
        """Temporarily pause speech detection.

        While paused the background loop keeps running but ignores all
        audio.  Use this during TTS playback to prevent the VAD from
        picking up the speaker output and creating a feedback loop.
        """
        if not self._paused:
            self._paused = True
            # Discard any audio accumulated so far
            if self._recorder.is_recording:
                self._recorder.cancel()
            # Reset VAD model state so it doesn't carry over
            if self._vad_model is not None and hasattr(self._vad_model, "reset_states"):
                self._vad_model.reset_states()  # type: ignore[union-attr]
            logger.debug("VAD paused")

    def resume(self) -> None:
        """Resume speech detection after :meth:`pause`.

        If the background loop task has died (e.g. due to an unhandled
        exception), it is automatically restarted.
        """
        if self._paused:
            self._paused = False
            logger.debug("VAD resumed")
        # If the loop task is gone, restart it so recording actually resumes.
        if self._active and (self._task is None or self._task.done()):
            logger.info("VAD loop task was dead, restarting")
            self._task = asyncio.create_task(self._loop(), name="continuous-vad")

    async def _loop(self) -> None:
        """Main monitoring loop.

        Uses Silero VAD to detect speech start/end instead of RMS energy.
        The Silero model runs on each 32 ms audio chunk and returns a
        speech probability — far more accurate than volume thresholding.
        """
        try:
            while self._active:
                # Wait while paused
                while self._paused and self._active:
                    await asyncio.sleep(VAD_POLL_INTERVAL)

                if not self._active:
                    break

                # Ensure VAD model is loaded before starting the mic
                self._ensure_vad()

                # Start recording to monitor audio
                try:
                    self._recorder.start()
                except RuntimeError as exc:
                    logger.error("ContinuousVAD: mic error: %s", exc)
                    await asyncio.sleep(2.0)
                    continue

                in_speech = False
                silent_since: float | None = None
                speech_start: float | None = None

                try:
                    while self._active:
                        await asyncio.sleep(VAD_POLL_INTERVAL)

                        # If paused mid-recording, discard and break out
                        if self._paused:
                            if self._recorder.is_recording:
                                self._recorder.cancel()
                            break

                        if not self._recorder.is_recording:
                            break

                        # Get recent samples for Silero VAD
                        clip = self._recorder.peek()
                        if len(clip.samples) < _SILERO_VAD_CHUNK_SAMPLES:
                            continue

                        is_speech = self._is_speech(clip.samples)

                        if is_speech:
                            # Speech detected
                            silent_since = None
                            if not in_speech:
                                in_speech = True
                                speech_start = time.monotonic()
                                logger.debug("VAD: speech started")
                        else:
                            # Silence
                            if in_speech:
                                if silent_since is None:
                                    silent_since = time.monotonic()
                                elif (
                                    time.monotonic() - silent_since
                                    >= self._silence_timeout
                                ):
                                    # Speech segment ended
                                    duration = time.monotonic() - (
                                        speech_start or time.monotonic()
                                    )
                                    if duration >= self._min_duration:
                                        final_clip = self._recorder.stop()
                                        if final_clip.duration >= self._min_duration:
                                            logger.debug(
                                                "VAD: speech ended (%.1fs)",
                                                final_clip.duration,
                                            )
                                            # Reset VAD state for next segment
                                            if hasattr(self._vad_model, "reset_states"):
                                                self._vad_model.reset_states()  # type: ignore[union-attr]
                                            try:
                                                result = self._on_speech(final_clip)
                                                if asyncio.iscoroutine(result):
                                                    await result
                                            except Exception:
                                                logger.exception(
                                                    "VAD: on_speech callback error"
                                                )
                                            break  # restart recording loop
                                    else:
                                        logger.debug(
                                            "VAD: speech too short, discarding"
                                        )
                                    in_speech = False
                                    silent_since = None
                                    speech_start = None

                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("ContinuousVAD: error in monitoring loop")
                finally:
                    if self._recorder.is_recording:
                        self._recorder.cancel()
                    # Reset VAD internal state between recording sessions
                    if self._vad_model is not None and hasattr(
                        self._vad_model, "reset_states"
                    ):
                        self._vad_model.reset_states()  # type: ignore[union-attr]

        except asyncio.CancelledError:
            logger.debug("ContinuousVAD stopped")
