import cv2
import time
import config

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


class AlertSystem:

    def __init__(self):
        print("[Alert] Alert system ready.")

        self.high_risk_counter = 0
        self.alert_active = False
        self.alert_start_time = None
        self.sound_ready = False

        if config.ENABLE_SOUND and PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
                self.sound_ready = True
                print("[Alert] Sound ready.")
            except Exception:
                print("[Alert] Sound failed. Continuing without sound.")

    def update(self, risk_level):
        if risk_level == config.RISK_HIGH:
            self.high_risk_counter += 1

            if self.high_risk_counter >= config.ALERT_FRAME_THRESHOLD:
                if not self.alert_active:
                    self.alert_active = True
                    self.alert_start_time = time.time()
                    self._play_sound()
                    print("[Alert] HIGH RISK ALERT ACTIVATED.")

        else:
            self.high_risk_counter = max(
                0,
                self.high_risk_counter - 1
            )

            if self.high_risk_counter == 0 and self.alert_active:
                print("[Alert] Risk reduced. Alert cleared.")
                self.alert_active = False
                self.alert_start_time = None

        return self.alert_active

    def display(self, frame):
        if not self.alert_active:
            return frame

        elapsed = time.time() - self.alert_start_time
        flash_cycle = elapsed % 1.0

        if flash_cycle > 0.6:
            return frame

        height, width = frame.shape[:2]

        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (width, 90),
            (0, 0, 150),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.75,
            frame,
            0.25,
            0,
            frame
        )

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = min(width / 800, 2.0)
        thickness = 3

        text = config.ALERT_TEXT

        (text_w, text_h), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        text_x = (width - text_w) // 2
        text_y = (90 + text_h) // 2

        cv2.putText(
            frame,
            text,
            (text_x + 2, text_y + 2),
            font,
            font_scale,
            (0, 0, 0),
            thickness + 2
        )

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            config.ALERT_COLOR,
            thickness
        )

        seconds = int(time.time() - self.alert_start_time)

        cv2.putText(
            frame,
            f"Alert active: {seconds}s",
            (15, height - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            config.ALERT_COLOR,
            2
        )

        return frame

    def _play_sound(self):
        if not self.sound_ready:
            return

        try:
            import numpy as np

            sr = 44100
            t = np.linspace(0, 1.0, sr)

            wave = (
                np.sin(2 * np.pi * 880 * t) * 32767
            ).astype(np.int16)

            wave = np.column_stack([wave, wave])

            pygame.sndarray.make_sound(wave).play()

        except Exception as e:
            print(f"[Alert] Sound error: {e}")

    def get_counter(self):
        return self.high_risk_counter