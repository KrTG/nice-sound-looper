import player
import recorder
from gui_lib import StretchImage

import cv2
import librosa
import numpy as np
import soundfile

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
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider

import json
import os
import zipfile

WHITE = (1, 1, 1, 1)
GREEN = (0.4, 1, 0.4, 1)
YELLOW = (1, 1, 0.4, 1)


def to_texture(image):
    image = cv2.flip(image, 0)
    im_bytes = np.reshape(image, [-1])
    out_texture = Texture.create(size=(image.shape[1], image.shape[0]))
    out_texture.blit_buffer(im_bytes, colorfmt='bgr', bufferfmt='ubyte')
    #cv2.imshow("test", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return out_texture

class CustomLabel(Label):
    pass

class VolumeSlider(BoxLayout):
    slider = ObjectProperty(None)
    def get(self):
        return 2 * self.slider.value / 100

class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    extension = ObjectProperty(None)

class SaveDialog(FloatLayout):
    save = ObjectProperty(None)
    cancel = ObjectProperty(None)
    extension = ObjectProperty(None)

class Track(BoxLayout):
    info = StringProperty("Info: ")
    box = ObjectProperty(None)
    scale = NumericProperty(0)
    texture = ObjectProperty(None)
    record_button_text = StringProperty("Record")
    record_button_color = ObjectProperty(WHITE)
    play_button_text = StringProperty("Play")
    play_button_color = ObjectProperty(WHITE)
    number = NumericProperty(None)
    screen = ObjectProperty(None)
    volume = ObjectProperty(None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reset()

    def reset(self):
        self.spectrogram = None
        self.new_texture = None
        self.track_length = 0
        self.scale = 0
        self.playing = False
        self.record_button_text = "Record"
        self.record_button_color = WHITE
        self.play_button_text = "Play"
        self.play_button_color = WHITE

    def on_press_record(self):
        try:
            self.screen.start_listening(self.number)
            self.record_button_text = "Waiting..."
            self.record_button_color = YELLOW
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
        self.record_button_color = GREEN

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

    def on_recorder_stop(self, spectrogram, track):
        length = "{:.2f}".format(len(track) / recorder.SR)
        s_length = "{}".format(spectrogram.shape[1])
        self.info = "Info: \nLength: " + length
        self.set_spectrogram(spectrogram)
        self.record_button_text = "Record"
        self.record_button_color = WHITE

    def on_playing_start(self):
        self.play_button_text = "Playing..."
        self.play_button_color = GREEN
        self.playing = True

    def on_playing_stop(self):
        self.play_button_text = "Play"
        self.play_button_color = WHITE
        self.playing = False

    def set_scale(self, max_length):
        self.scale = self.track_length / max_length

    def update_image(self, _):
        self.texture = to_texture(self.new_texture)


class Screen(FloatLayout):
    track1 = ObjectProperty(None)
    track2 = ObjectProperty(None)
    track3 = ObjectProperty(None)
    track4 = ObjectProperty(None)
    latency_adjustment = ObjectProperty(None)
    noise_threshold = ObjectProperty(None)
    noise_sample_button_color = ObjectProperty(WHITE)
    progress_bar = NumericProperty(0.0)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = None
        self.load_config()
        self.tracks = {1: self.track1, 2: self.track2, 3: self.track3, 4: self.track4}
        self.current_track = None
        self.sampling_noise = False
        self.recorder = recorder.Recorder(
            start_callback=self.on_recorder_start,
            stop_callback=self.on_recorder_stop
        )
        self.player = player.Player(volume_callback=self.get_volumes)
        self.player.start()
        Clock.schedule_interval(self.update_progress, 0.02)

        self.get_volumes()

    def get_latency_adjustment(self):
        try:
            return int(self.latency_adjustment.text)
        except ValueError:
            print("Latency adjustment not an integer")
            return 0

    def get_noise_threshold(self):
        try:
            return float(self.noise_threshold.text)
        except ValueError:
            print("Noise threshold is not a number")
            return 0

    def load_config(self):
        with open("config.json") as config:
            try:
                self.config = json.load(config)
            except:
                self.config = { "latency_adjustment": 0, "noise_threshold": 1.0 }

        self.latency_adjustment.text = str(self.config["latency_adjustment"])
        self.noise_threshold.text = str(self.config["noise_threshold"])

    def save_config(self):
        self.config["latency_adjustment"] = self.get_latency_adjustment()
        self.config["noise_threshold"] = self.get_noise_threshold()
        with open("config.json", "w") as config:
            json.dump(self.config, config)

    def reset(self):
        for t in self.tracks.values():
            t.reset()
        self.player.tracks = {}

    def start_listening(self, track_number):
        self.current_track = track_number
        if (len(self.player.tracks) == 0 or
            len(self.player.tracks) == 1 and self.player.tracks.get(track_number)):
            self.recorder.wait(self.player.get_reference_progress(exclude=track_number) - self.get_latency_adjustment())
        else:
            self.recorder.wait(
                self.player.get_reference_progress(exclude=track_number) - self.get_latency_adjustment(),
                self.player.get_reference_frame(exclude=track_number)
            )

    def start_playing(self, track_number):
        self.player.play(track_number)

    def stop_playing(self, track_number):
        self.player.pause(track_number)

    def cut_track(self, track_number):
        spectrogram = self.player.cut(track_number)
        self.tracks[track_number].set_spectrogram(spectrogram)
        self.rescale_tracks()

    def get_noise_sample(self):
        self.sampling_noise = True
        self.noise_sample_button_color = GREEN
        self.recorder.record(recorder.SR * 2.5)

    def on_recorder_start(self):
        self.tracks[self.current_track].on_recorder_start()

    def rescale_tracks(self):
        max_len = max(t.track_length for t in self.tracks.values())
        for track in self.tracks.values():
            track.set_scale(max_len)

    def add_track(self, number, track, spectrogram):
        self.tracks[number].on_recorder_stop(spectrogram, track)
        self.player.add_track(number, track)
        self.player.play(number)
        self.tracks[number].on_playing_start()
        self.rescale_tracks()

    def get_volumes(self):
        return {n: track.volume.get() for n, track in self.tracks.items()}

    def on_recorder_stop(self):
        if self.sampling_noise:
            track = self.recorder.raw()
            self.sampling_noise = False
            self.recorder.noise_sample = track.flatten()
            self.noise_sample_button_color = WHITE
            #self.recorder.playback()
        else:
            track, spectrogram = self.recorder.postprocess(self.get_noise_threshold())
            self.add_track(self.current_track, track, spectrogram)

    def dismiss_popup(self):
        self._popup.dismiss(animation=False)

    def show_save(self):
        content = SaveDialog(save=self.save, cancel=self.dismiss_popup, extension=".looper")
        self._popup = Popup(
            title="Save file", content=content, size_hint=(0.9, 0.8), pos_hint={'y': 0.2},
            auto_dismiss=False
        )
        self._popup.open()

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup, extension=".looper")
        self._popup = Popup(
            title="Load file", content=content, size_hint=(0.9, 0.8), pos_hint={'y': 0.2},
            auto_dismiss=False
        )
        self._popup.open()

    def show_export(self):
        content = SaveDialog(save=self.export, cancel=self.dismiss_popup, extension=".wav")
        self._popup = Popup(
            title="Load file", content=content, size_hint=(0.9, 0.8), pos_hint={'y': 0.2},
            auto_dismiss=False
        )
        self._popup.open()

    def save(self, path, filename):
        self.dismiss_popup()
        if not filename:
            return

        if not filename.endswith(".looper"):
            filename += ".looper"
        with zipfile.ZipFile(os.path.join(path, filename), "w") as zippy:
            if self.recorder.noise_sample is not None:
                name = "sv_noise"
                with open(name, "wb") as savefile:
                    np.save(savefile, self.recorder.noise_sample, allow_pickle=False, fix_imports=False)
                    zippy.write(name)
                os.remove(name)

            for number, track in self.player.tracks.items():
                name = "sv_{}".format(number)
                with open(name, "wb") as savefile:
                    np.save(savefile, track.track[:track.len], allow_pickle=False, fix_imports=False)
                    zippy.write(name)
                os.remove(name)

    def load(self, path, filename):
        self.dismiss_popup()
        if not filename:
            return

        with zipfile.ZipFile(filename[0], "r") as zippy:
            tracks = zippy.namelist()
            if "sv_noise" in tracks:
                with zippy.open("sv_noise", "r") as noise_file:
                    noise_sample = np.load(noise_file)
                    self.recorder.noise_sample = noise_sample
                tracks.remove("sv_noise")

            for track_save in tracks:
                with zippy.open(track_save, "r") as track_file:
                    track = np.load(track_file)
                    number = int(track_save.split("_")[1])
                    self.add_track(number, track, self.recorder.get_spectrogram(track))

    def export(self, path, filename):
        self.dismiss_popup()
        if not filename:
            return

        if not filename.endswith(".wav"):
            filename += ".wav"

        output = self.player.export()
        with soundfile.SoundFile(filename, "w", samplerate=player.SR, channels=1) as wav:
            wav.write(output)

    def update_progress(self, _):
        reference = self.player.get_max_frame()
        if reference is None:
            reference = 1
        self.progress_bar = self.player.get_max_progress() / reference


class LooperApp(App):
    def build(self):
        return Screen()

if __name__ == '__main__':
    LooperApp().run()
