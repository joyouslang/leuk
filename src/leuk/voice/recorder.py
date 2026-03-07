"""Microphone audio capture using sounddevice.

Provides push-to-talk style recording: call ``start()`` to begin capturing
audio, then ``stop()`` to get the recorded audio as a NumPy array and WAV bytes.

The recorder queries the system's default input device for its native sample
rate and channel count, records at those settings, and then resamples /
down-mixes to 16 kHz mono on ``stop()`` — the format Whisper expects.

For live transcription, ``peek()`` returns a non-destructive snapshot of the
audio captured so far, and ``recent_rms()`` provides energy levels for
voice-activity detection.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import time
import wave
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
    return np.interp(indices, np.arange(len(samples)), samples.astype(np.float64)).astype(
        samples.dtype
    )


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

        def _callback(indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
            if status:
                logger.warning("sounddevice status: %s", status)
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
