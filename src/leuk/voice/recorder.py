"""Microphone audio capture using sounddevice.

Provides push-to-talk style recording: call ``start()`` to begin capturing
audio, then ``stop()`` to get the recorded audio as a NumPy array and WAV bytes.
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

# Audio defaults
DEFAULT_SAMPLE_RATE = 16_000  # 16 kHz — Whisper's native rate
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


class MicRecorder:
    """Records audio from the default microphone.

    Usage::

        recorder = MicRecorder()
        recorder.start()
        # ... user speaks ...
        clip = recorder.stop()
        wav_bytes = clip.to_wav_bytes()

    Parameters
    ----------
    sample_rate:
        Audio sample rate in Hz.  Default 16000 (Whisper's native rate).
    channels:
        Number of audio channels.  Default 1 (mono).
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
    ) -> None:
        from leuk.voice import require_voice

        require_voice()

        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._stream: object | None = None
        self._recording = False

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        """Start capturing audio from the microphone."""
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

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=DEFAULT_DTYPE,
            callback=_callback,
        )
        self._stream.start()  # type: ignore[union-attr]
        self._recording = True
        logger.debug("Recording started (rate=%d, ch=%d)", self.sample_rate, self.channels)

    def stop(self) -> AudioClip:
        """Stop recording and return the captured audio."""
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
            samples = np.concatenate(self._frames, axis=0).flatten()

        clip = AudioClip(
            samples=samples,
            sample_rate=self.sample_rate,
            channels=self.channels,
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
