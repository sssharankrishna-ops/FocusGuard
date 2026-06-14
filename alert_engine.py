"""
alert_engine.py — Multi-level alert system for FocusGuard.
Handles audio beeps, screen flash signals, and debounce logic.
"""

import time
import threading
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

SOUNDS_DIR     = os.path.join(os.path.dirname(__file__), '..', 'sounds')
DEBOUNCE_SEC   = 5      # minimum seconds between same-level alerts
FLASH_DURATION = 0.4    # seconds for screen flash

# Alert level → sound file
SOUND_FILES = {
    1: 'beep_soft.wav',
    2: 'beep_loud.wav',
    3: 'alarm.wav',
}


def _generate_beep(filename: str, freq: int, duration: float,
                   volume: float = 0.5):
    """Generate a sine-wave WAV file using numpy + scipy."""
    try:
        from scipy.io import wavfile
        sr   = 44100
        t    = np.linspace(0, duration, int(sr * duration), False)
        wave = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
        wavfile.write(filename, sr, wave)
        logger.info(f"Generated sound: {filename}")
    except Exception as e:
        logger.warning(f"Could not generate sound file {filename}: {e}")


def ensure_sounds():
    """Create sound files if they don't exist."""
    os.makedirs(SOUNDS_DIR, exist_ok=True)
    specs = {
        'beep_soft.wav': (880,  0.3, 0.3),
        'beep_loud.wav': (1200, 0.5, 0.7),
        'alarm.wav':     (1500, 1.0, 0.9),
    }
    for fname, (freq, dur, vol) in specs.items():
        path = os.path.join(SOUNDS_DIR, fname)
        if not os.path.exists(path):
            _generate_beep(path, freq, dur, vol)


class AlertEngine:
    """
    Thread-safe multi-level alert system.
    Levels: 0=normal, 1=warning, 2=danger, 3=critical.
    Provides debounce, audio playback, and flash signals.
    """

    def __init__(self):
        self._lock            = threading.Lock()
        self._current_level   = 0
        self._last_trigger    = {}     # level → timestamp
        self._stop_event      = threading.Event()
        self._alarm_thread: threading.Thread | None = None
        self._flash_until     = 0.0   # epoch time until which screen should flash
        self._pygame_ok       = False
        self._sounds          = {}

        ensure_sounds()
        self._init_pygame()

    def _init_pygame(self):
        try:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            for level, fname in SOUND_FILES.items():
                path = os.path.join(SOUNDS_DIR, fname)
                if os.path.exists(path):
                    self._sounds[level] = pygame.mixer.Sound(path)
            self._pygame_ok = True
            self._pygame    = pygame
            logger.info("Alert audio initialised.")
        except Exception as e:
            logger.warning(f"pygame not available — audio alerts disabled: {e}")
            self._pygame_ok = False

    def trigger_alert(self, level: int, reason: str = "") -> bool:
        """
        Fire an alert at the given level.
        Respects debounce: returns False if suppressed.
        """
        if level == 0:
            self.clear_alert()
            return False

        with self._lock:
            now  = time.time()
            last = self._last_trigger.get(level, 0)
            if now - last < DEBOUNCE_SEC:
                return False   # debounced

            self._last_trigger[level] = now
            self._current_level       = level

        logger.info(f"ALERT L{level}: {reason}")
        self._play_sound(level)

        if level >= 2:
            self._flash_until = time.time() + FLASH_DURATION

        if level == 3:
            self._start_continuous_alarm()

        return True

    def _play_sound(self, level: int):
        if not self._pygame_ok:
            return
        try:
            snd = self._sounds.get(level)
            if snd:
                self._pygame.mixer.stop()
                snd.play()
        except Exception as e:
            logger.warning(f"Sound playback error: {e}")

    def _start_continuous_alarm(self):
        """Run alarm in a loop until clear_alert() is called."""
        self._stop_event.clear()
        if self._alarm_thread and self._alarm_thread.is_alive():
            return

        def _loop():
            while not self._stop_event.is_set():
                self._play_sound(3)
                time.sleep(1.2)

        self._alarm_thread = threading.Thread(target=_loop, daemon=True)
        self._alarm_thread.start()

    def clear_alert(self):
        """Stop all sounds and reset level."""
        self._stop_event.set()
        self._current_level = 0
        if self._pygame_ok:
            try:
                self._pygame.mixer.stop()
            except Exception:
                pass

    def should_flash(self) -> bool:
        """Return True if the dashboard should be flashing red right now."""
        return time.time() < self._flash_until

    @property
    def current_level(self) -> int:
        return self._current_level

    def get_alert_color_bgr(self) -> tuple:
        colors = {0: (0,200,0), 1: (0,200,255), 2: (0,100,255), 3: (0,0,255)}
        return colors.get(self._current_level, (0,200,0))

    def shutdown(self):
        self.clear_alert()
        if self._pygame_ok:
            try:
                self._pygame.mixer.quit()
            except Exception:
                pass
