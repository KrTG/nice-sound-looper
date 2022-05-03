import player
import recorder
from gui_lib import StretchImage

import cv2
import librosa
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Rectangle
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty
from kivy.properties import ObjectProperty
from kivy.properties import StringProperty
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label

def to_texture(image):
    image = cv2.flip(image, 0)
    im_bytes = np.reshape(image, [-1])
    out_texture = Texture.create(size=(image.shape[1], image.shape[0]))
    out_texture.blit_buffer(im_bytes, colorfmt='bgr', bufferfmt='ubyte')
    #cv2.imshow("test", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return out_texture

class Track(BoxLayout):
    box = ObjectProperty(None)
    scale = NumericProperty(1)
    texture = ObjectProperty(None)
    record_button_text = StringProperty("Record")
    record_button_color = ObjectProperty((1, 1, 1, 1))
    play_button_text = StringProperty("Play")
    play_button_color = ObjectProperty((1, 1, 1, 1))
    number = NumericProperty(None)
    screen = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spectrogram = None
        self.new_texture = None
        self.track_length = 0
        self.scale = 0
        self.playing = False

    def on_press_record(self):
        try:
            self.screen.start_listening(self.number)
            self.record_button_text = "Waiting..."
            self.record_button_color = (1, 1, 0.4, 1)
        except recorder.RecorderException as e:
            print(str(e))

    def on_press_play(self):
        try:
            if self.playing:
                self.screen.stop_playing(self.number)
                self.on_playing_stop()
            else:
                self.screen.start_playing(self.number)
                self.on_playing_start()
        except player.PlayerException as e:
            print(str(e))

    def on_press_cut(self):
        try:
            self.screen.cut_track(self.number)
        except player.PlayerException as e:
            print(str(e))

    def on_recorder_start(self):
        self.record_button_text = "Recording..."
        self.record_button_color = (0.4, 1, 0.4, 1)

    def update_image(self, _):
        self.texture = to_texture(self.new_texture)

    def set_spectrogram(self, spectrogram):
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram += np.min(spectrogram)
        spectrogram /= np.max(spectrogram)
        spectrogram *= 255
        spectrogram = np.array(spectrogram, dtype=np.uint8)
        spectrogram = np.expand_dims(spectrogram, axis=2)
        spectrogram = np.repeat(spectrogram, 3, axis=2)
        self.new_texture = spectrogram
        self.track_length = self.new_texture.shape[1]
        Clock.schedule_once(self.update_image)

    def on_recorder_stop(self, spectrogram):
        self.set_spectrogram(spectrogram)
        self.record_button_text = "Record"
        self.record_button_color = (1, 1, 1, 1)

    def on_playing_start(self):
        self.play_button_text = "Playing..."
        self.play_button_color = (0.4, 1, 0.4, 1)
        self.playing = True

    def on_playing_stop(self):
        self.play_button_text = "Play"
        self.play_button_color = (1, 1, 1, 1)
        self.playing = False

    def set_scale(self, max_length):
        self.scale = self.track_length / max_length


class Screen(FloatLayout):
    track1 = ObjectProperty(None)
    track2 = ObjectProperty(None)
    track3 = ObjectProperty(None)
    track4 = ObjectProperty(None)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tracks = {1: self.track1, 2: self.track2, 3: self.track3, 4: self.track4}
        self.current_track = None
        self.keyboard = Window.request_keyboard(lambda: 0, self, "text")
        self.keyboard.bind(on_key_down=self.handle_keyboard)
        self.recorder = recorder.Recorder(
            start_callback=self.on_recorder_start,
            stop_callback=self.on_recorder_stop
        )

        self.player = player.Player()
        self.player.start()

    def start_listening(self, track_number):
        self.current_track = track_number
        if (len(self.player.tracks) == 0 or
            len(self.player.tracks) == 1 and self.player.tracks.get(track_number)):
            self.recorder.wait(self.player.reference_progress)
        else:
            self.recorder.wait(self.player.reference_progress, self.player.reference_frame)

    def start_playing(self, track_number):
        self.player.play(track_number)

    def stop_playing(self, track_number):
        self.player.pause(track_number)

    def cut_track(self, track_number):
        spectrogram = self.player.cut(track_number)
        self.tracks[track_number].set_spectrogram(spectrogram)
        self.rescale_tracks()

    def on_recorder_start(self):
        self.tracks[self.current_track].on_recorder_start()

    def rescale_tracks(self):
        max_len = max(t.track_length for t in self.tracks.values())
        for track in self.tracks.values():
            track.set_scale(max_len)

    def on_recorder_stop(self):
        track, spectrogram = self.recorder.postprocess()
        self.tracks[self.current_track].on_recorder_stop(spectrogram)
        self.player.add_track(self.current_track, track)
        self.player.play(self.current_track)
        self.tracks[self.current_track].on_playing_start()
        self.rescale_tracks()

    def handle_keyboard(self, keyboard, keycode, text, modifiers):
        if keycode[1] == "1":
            self.track1.on_press()
        elif keycode[1] == "2":
            self.track2.on_press()



class LooperApp(App):
    def build(self):
        return Screen()

if __name__ == '__main__':
    LooperApp().run()
