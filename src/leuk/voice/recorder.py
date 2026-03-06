"""Microphone audio capture using sounddevice.

Provides push-to-talk style recording: call ``start()`` to begin capturing
audio, then ``stop()`` to get the recorded audio as a NumPy array and WAV bytes.

The recorder queries the system's default input device for its native sample
rate and channel count, records at those settings, and then resamples /
down-mixes to 16 kHz mono on ``stop()`` — the format Whisper expects.
"""

from __future__ import annotations

import io
import logging
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
    """Query the default input device for its native sample rate and channels.

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
        system default is tried first; if that fails, every hardware input
        device is attempted in order until one succeeds.

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
            # Default first, then every ALSA hw device with input channels.
            candidates: list[int | str | None] = [None]
            for idx, info in enumerate(sd.query_devices()):
                if info["max_input_channels"] > 0 and "hw:" in info.get("name", ""):
                    candidates.append(idx)

        last_err: Exception | None = None
        for dev in candidates:
            try:
                rate, channels = _query_input_device(dev)
            except sd.PortAudioError as exc:
                last_err = exc
                continue

            try:
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

            # Success.
            self._device_rate = rate
            self._device_channels = channels
            self._recording = True
            dev_name = dev if dev is not None else "default"
            logger.debug(
                "Recording started on device %s (rate=%d, ch=%d)",
                dev_name,
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

            # Down-mix to mono if multi-channel.
            if raw.ndim == 2 and raw.shape[1] > 1:
                raw = raw.mean(axis=1).astype(raw.dtype)
            else:
                raw = raw.flatten()

            # Resample to Whisper's 16 kHz if the device recorded at a different rate.
            samples = _resample(raw, self._device_rate, WHISPER_SAMPLE_RATE)

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
