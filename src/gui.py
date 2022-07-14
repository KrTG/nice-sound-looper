import glob
import json
import os
import traceback
import zipfile

import librosa
import numpy as np
import soundfile
from kivy.app import App
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.graphics.texture import Texture
from kivy.properties import NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from src import player, recorder
from src.const import *

Config.set("input", "mouse", "mouse,multitouch_on_demand")
Window.size = WINDOW_SIZE


def to_texture(image):
    image = np.flip(image, 0)
    im_bytes = np.reshape(image, [-1])
    out_texture = Texture.create(size=(image.shape[1], image.shape[0]))
    out_texture.blit_buffer(im_bytes, colorfmt="bgr", bufferfmt="ubyte")

    return out_texture


class CustomLabel(Label):
    pass


class VolumeSlider(BoxLayout):
    slider = ObjectProperty(None)

    def get(self):
        return 2 * self.slider.value / 100

    def set(self, value):
        self.slider.value = int(value * 100 / 2)


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)
    extension = ObjectProperty(None)


class SaveDialog(FloatLayout):
    cancel = ObjectProperty(None)
    extension = ObjectProperty(None)
    filename = ObjectProperty("")
    path = StringProperty("")
    save = ObjectProperty(None)


class Track(BoxLayout):
    info_text = StringProperty("Info: ")
    box = ObjectProperty(None)
    scale = NumericProperty(0)
    texture = ObjectProperty(None)
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
        self.np_texture = None
        self.track_length = 0
        self.scale = 0
        self.playing = False
        self.record_button_text = "Record"
        self.record_button_color = WHITE
        self.play_button_text = "Play"
        self.play_button_color = WHITE
        self.info_text = ""
        self.info = {"length": 0, "samples": 0, "repeats": 0}
        self.watch_file = None
        self.watch_file_last_changed = 0
        if self.volume:
            self.volume.set(1)
        Clock.schedule_interval(self.update_volume, 0.02)

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

    def on_press_export(self):
        self.screen.show_export_track(self.number)

    def on_press_reset(self):
        self.reset()
        self.screen.reset_track(self.number)

    def generate_info(self):
        template = """Length: {length}
Samples: {samples}
Repeats: {repeats}
"""
        self.info_text = template.format(**self.info)

    def set_track(self, track, spectrogram):
        length = "{:.2f}".format(len(track) / recorder.SR)
        self.info["samples"] = len(track)
        self.info["length"] = length
        self.generate_info()
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram += np.min(spectrogram)
        spectrogram /= np.max(spectrogram)
        spectrogram *= 255
        spectrogram = np.array(spectrogram, dtype=np.uint8)

        self.np_texture = spectrogram
        texture = self.np_texture[:]
        texture = np.expand_dims(self.np_texture, axis=2)
        texture = np.repeat(texture, 3, axis=2)
        self.track_length = texture.shape[1]
        self.texture = to_texture(texture)

    def on_playing_start(self):
        self.play_button_text = "Playing..."
        self.play_button_color = GREEN
        self.playing = True

    def on_playing_stop(self):
        self.play_button_text = "Play"
        self.play_button_color = WHITE
        self.playing = False

    def set_scale(self, max_length):
        repeats = round(
            max(0, max_length / self.track_length)
        )  # small adjustment in case this is not a perfect division

        repeated_texture = np.tile(self.np_texture, repeats)
        texture = np.expand_dims(repeated_texture, axis=2)
        texture = np.repeat(texture, 3, axis=2)
        self.texture = to_texture(texture)
        self.scale = 1
        self.info["repeats"] = repeats
        self.generate_info()

    def update_volume(self, _):
        try:
            self.screen.player.tracks[self.number].volume = self.volume.get()
        except KeyError:
            pass


class Screen(FloatLayout):
    trackLayout = ObjectProperty(None)
    export_length = ObjectProperty(None)
    noise_threshold = ObjectProperty(None)
    silence_threshold = ObjectProperty(None)
    silence_window = ObjectProperty(None)
    noise_sample_button_color = ObjectProperty(WHITE)
    progress_bar = NumericProperty(0.0)

    record_button_text = StringProperty("Record")
    record_button_color = ObjectProperty(WHITE)

    startpoint_button_text = StringProperty("Change\nstartpoint")
    startpoint_button_color = ObjectProperty(WHITE)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = None
        self.load_config()
        self.tracks = {}
        self.sampling_noise = False
        self.path = None
        self.filename = None
        self._popup = None
        self.changing_startpoint = False
        self.recorder = recorder.Recorder(
            start_callback=self.on_recorder_start, stop_callback=self.on_recorder_stop
        )
        self.player = player.Player()
        self.player.start()
        self.recorder.start()
        self.player.forward_to(0)
        self.recorder.forward_to(0)
        Clock.schedule_interval(self.update_progress, 0.02)
        Clock.schedule_interval(self.watch_for_changes, 1)
        Clock.schedule_interval(self.autosave, AUTOSAVE_FREQ)
        self.reset()

    def on_touch_down(self, touch):
        for track in self.tracks.values():
            if track.box.collide_point(*touch.pos):
                x = (touch.pos[0] - track.box.pos[0]) / track.box.width
                if (
                    self.changing_startpoint
                    and not self.startpoint_button.collide_point(*touch.pos)
                ):
                    self.change_startpoint(x)
                else:
                    self.forward_to(x)
                break
        else:
            if self.changing_startpoint:
                self.stop_changing_startpoint()

        super().on_touch_down(touch)

    @property
    def tracks_x(self):
        try:
            return list(self.tracks.values())[0].box.x
        except (IndexError, AttributeError):
            return 0

    @property
    def tracks_width(self):
        try:
            return list(self.tracks.values())[0].box.width
        except (IndexError, AttributeError):
            return 0

    def get_latency_adjustment(self):
        return int((self.player.stream.latency + self.recorder.stream.latency) * SR)

    def get_noise_threshold(self):
        try:
            return float(self.noise_threshold.text)
        except ValueError:
            print("Noise threshold is not a number")
            return DEFAULT_NOISE_THRESHOLD

    def get_silence_threshold(self):
        try:
            return float(self.silence_threshold.text)
        except ValueError:
            print("Noise threshold is not a number")
            return DEFAULT_SILENCE_THRESHOLD

    def get_silence_window(self):
        try:
            return float(self.silence_window.text)
        except ValueError:
            print("Noise threshold is not a number")
            return DEFAULT_SILENCE_WINDOW

    def load_config(self):
        try:
            with open("config.json") as config:
                self.config = json.load(config)
        except (ValueError, OSError):
            self.config = {}
        self.config.setdefault("noise_threshold", DEFAULT_NOISE_THRESHOLD)
        self.config.setdefault("silence_threshold", DEFAULT_SILENCE_THRESHOLD)
        self.config.setdefault("silence_window", DEFAULT_SILENCE_WINDOW)

        self.noise_threshold.text = str(self.config["noise_threshold"])
        self.silence_threshold.text = str(self.config["silence_threshold"])
        self.silence_window.text = str(self.config["silence_window"])

    def save_config(self):
        self.config["noise_threshold"] = self.get_noise_threshold()
        self.config["silence_threshold"] = self.get_silence_threshold()
        self.config["silence_window"] = self.get_silence_window()
        with open("config.json", "w") as config:
            json.dump(self.config, config)

    def reset_tracks(self):
        for track in self.tracks.values():
            self.trackLayout.remove_widget(track)

        self.tracks = {}

    def reset(self):
        self.reset_tracks()
        self.player.tracks = {}
        self.path = "."
        self.filename = ""
        self.changing_startpoint = False

    def on_press_record(self):
        try:
            self.start_listening(max(self.tracks, default=0) + 1)
            self.record_button_text = "Waiting..."
            self.record_button_color = YELLOW
        except recorder.RecorderException as e:
            print(str(e))

    def start_listening(self, track_number):
        if (
            len(self.player.tracks) == 0
            or len(self.player.tracks) == 1
            and self.player.tracks.get(track_number)
        ):
            self.recorder.wait_for_sound(
                self.get_silence_threshold(),
                self.get_silence_window(),
                self.get_latency_adjustment(),
            )
        else:
            self.recorder.wait_for_sound(
                self.get_silence_threshold(),
                self.get_silence_window(),
                self.get_latency_adjustment(),
                reference_frame=self.player.get_max_frame(exclude=track_number),
            )

    def start_playing(self, track_number):
        self.player.play(track_number)

    def stop_playing(self, track_number):
        self.player.pause(track_number)

    def reset_track(self, track_number):
        try:
            self.player.remove_track(track_number)
            self.trackLayout.remove_widget(self.tracks[track_number])
            del self.tracks[track_number]
            self.rescale_tracks()
        except player.PlayerException as e:
            print(e)

    def get_noise_sample(self):
        self.sampling_noise = True
        self.noise_sample_button_color = GREEN
        self.recorder.record(recorder.SR * 2.5)

    def on_recorder_start(self):
        self.record_button_text = "Recording..."
        self.record_button_color = GREEN

    def rescale_tracks(self):
        max_len = max((t.track_length for t in self.tracks.values()), default=0)
        for track in self.tracks.values():
            track.set_scale(max_len)

    def postprocess_and_add_track(self, _):
        try:
            track = self.recorder.get_postprocessed_data(self.get_noise_threshold())
            spectrogram = self.recorder.get_spectrogram(track)
            self.add_track(track, spectrogram)
        except Exception as e:
            print(traceback.format_exception(e))

    def add_track(self, track, spectrogram, number=None):
        if number is None:
            number = max(self.tracks, default=0) + 1
        self.player.add_track(number, track)
        self.player.play(number)

        gui_track = Track(screen=self, number=number)
        gui_track.set_track(track, spectrogram)
        gui_track.on_playing_start()
        self.trackLayout.add_widget(gui_track)
        self.tracks[number] = gui_track
        self.rescale_tracks()

    def on_recorder_stop(self):
        if self.sampling_noise:
            track = self.recorder.get_raw_data()
            self.sampling_noise = False
            self.recorder.noise_sample = track
            self.noise_sample_button_color = WHITE
        else:
            self.record_button_text = "Record"
            self.record_button_color = WHITE
            Clock.schedule_once(self.postprocess_and_add_track)

    def dismiss_popup(self):
        self._popup.dismiss(animation=False)

    def show_save(self):
        content = SaveDialog(
            save=self.save,
            cancel=self.dismiss_popup,
            extension=".looper",
            filename=self.filename,
            path=self.path,
        )
        self._popup = Popup(
            title="Save file",
            content=content,
            size_hint=(0.9, 0.8),
            pos_hint={"y": 0.2},
            auto_dismiss=False,
        )
        self._popup.open()

    def show_load(self):
        content = LoadDialog(
            load=self.load, cancel=self.dismiss_popup, extension=".looper"
        )
        self._popup = Popup(
            title="Load file",
            content=content,
            size_hint=(0.9, 0.8),
            pos_hint={"y": 0.2},
            auto_dismiss=False,
        )
        self._popup.open()

    def show_export(self):
        content = SaveDialog(
            save=self.export,
            cancel=self.dismiss_popup,
            extension=".wav",
            filename="",
            path=".",
        )
        self._popup = Popup(
            title="Export file",
            content=content,
            size_hint=(0.9, 0.8),
            pos_hint={"y": 0.2},
            auto_dismiss=False,
        )
        self._popup.open()

    def show_export_track(self, number):
        content = SaveDialog(
            save=lambda p, f: self.export_track(number, p, f),
            cancel=self.dismiss_popup,
            extension=".wav",
            filename="",
            path=".",
        )
        self._popup = Popup(
            title="Export single track as .wav",
            content=content,
            size_hint=(0.9, 0.8),
            pos_hint={"y": 0.2},
            auto_dismiss=False,
        )
        self._popup.open()

    def save(self, path, filename, autosave=False):
        if not autosave:
            self.dismiss_popup()

        if not filename:
            return
        if not filename.endswith(".looper"):
            filename += ".looper"

        if not autosave:
            self.path = path
            self.filename = filename

        with zipfile.ZipFile(os.path.join(path, filename), "w") as zippy:
            if self.recorder.noise_sample is not None:
                name = "sv_noise"
                with open(name, "wb") as savefile:
                    np.save(
                        savefile,
                        self.recorder.noise_sample,
                        allow_pickle=False,
                        fix_imports=False,
                    )
                    zippy.write(name)
                os.remove(name)

            for number, track in self.player.tracks.items():
                name = "sv_{}".format(number)
                with open(name, "wb") as savefile:
                    np.save(
                        savefile,
                        track.track[: track.len],
                        allow_pickle=False,
                        fix_imports=False,
                    )
                zippy.write(name)
                os.remove(name)

                playing_name = name + ".playing"
                with open(playing_name, "wb") as savefile:
                    savefile.write(track.playing.to_bytes(1, "big"))
                zippy.write(playing_name)
                os.remove(playing_name)

                volume_name = name + ".volume"
                with open(volume_name, "wb") as savefile:
                    savefile.write(int(track.volume * 100).to_bytes(8, "big"))
                zippy.write(volume_name)
                os.remove(volume_name)

    def load(self, path, filename):
        self.dismiss_popup()
        if not filename:
            return

        self.reset()
        self.path = path
        self.filename = filename[0]

        with zipfile.ZipFile(filename[0], "r") as zippy:
            tracks = zippy.namelist()
            if "sv_noise" in tracks:
                with zippy.open("sv_noise", "r") as noise_file:
                    noise_sample = np.load(noise_file)[:, :CHANNELS]
                    self.recorder.noise_sample = noise_sample
                tracks.remove("sv_noise")

            for track_save in tracks[:]:
                if track_save.endswith(".playing"):
                    continue
                if track_save.endswith(".volume"):
                    continue
                with zippy.open(track_save, "r") as track_file:
                    track = np.load(track_file)[:, :CHANNELS]
                    number = int(track_save.split("_")[1])
                    self.add_track(
                        track, self.recorder.get_spectrogram(track), number=number
                    )
                    tracks.remove(track_save)

            for track_playing in tracks[:]:
                if not track_playing.endswith(".playing"):
                    continue
                with zippy.open(track_playing, "r") as track_file:
                    content = track_file.read()
                    playing = bool.from_bytes(content, "big")
                    number = int(track_playing.split(".")[0].split("_")[1])
                    if playing:
                        self.start_playing(number)
                        self.tracks[number].on_playing_start()
                    else:
                        self.stop_playing(number)
                        self.tracks[number].on_playing_stop()
                    tracks.remove(track_playing)

            for track_volume in tracks[:]:
                if not track_volume.endswith(".volume"):
                    continue
                with zippy.open(track_volume, "r") as track_file:
                    content = track_file.read()
                    volume = int.from_bytes(content, "big") / 100
                    number = int(track_volume.split(".")[0].split("_")[1])
                    self.tracks[number].volume.set(volume)
                    tracks.remove(track_volume)

    def export(self, path, filename):
        if not filename:
            return
        self.dismiss_popup()

        if not filename.endswith(".wav"):
            filename += ".wav"

        try:
            length = int(self.export_length.text)
        except ValueError:
            print("Invalid value for 'Minimum length'")
        output = self.player.export(min_length=length)
        fullpath = os.path.join(path, filename)
        with soundfile.SoundFile(
            fullpath, "w", samplerate=player.SR, channels=CHANNELS
        ) as wav:
            wav.write(output)

    def export_track(self, number, path, filename):
        if not filename:
            return
        self.dismiss_popup()

        if not filename.endswith(".wav"):
            filename += ".wav"
        fullpath = os.path.join(path, filename)
        with soundfile.SoundFile(
            fullpath, "w", samplerate=player.SR, channels=CHANNELS
        ) as wav:
            wav.write(self.player.get_track(number))

        self.tracks[number].watch_file = fullpath
        self.tracks[number].watch_file_last_changed = os.stat(fullpath).st_mtime

    def forward_to(self, percent_progress):
        x = int(self.player.get_max_frame() * percent_progress)
        self.player.forward_to(x)
        self.recorder.forward_to(x)

    def start_changing_startpoint(self):
        if self.changing_startpoint:
            self.stop_changing_startpoint()
            return
        if len(self.player.tracks) == 0:
            return
        self.startpoint_button_color = GREEN
        self.startpoint_button_text = "Click\non the\ntimeline"
        self.changing_startpoint = True

    def stop_changing_startpoint(self):
        self.startpoint_button_color = WHITE
        self.startpoint_button_text = "Change\nstartpoint"
        self.changing_startpoint = False

    def change_startpoint(self, percent_change):
        self.startpoint_button_color = WHITE
        self.startpoint_button_text = "Change\nstartpoint"
        self.changing_startpoint = False

        # do stuff
        self.player.change_startpoint(percent_change)
        for number, track in self.player.tracks.items():
            track = track.get_track()
            spectrogram = self.recorder.get_spectrogram(track)
            self.tracks[number].set_track(track, spectrogram)
        self.rescale_tracks()

    def refresh_track(self, number):
        filename = self.tracks[number].watch_file
        if filename is None:
            return
        with soundfile.SoundFile(filename, "r") as wav:
            new_track = wav.read(always_2d=True)[:, :CHANNELS]
            if new_track.shape[0] == 0:
                raise RuntimeError("Read error.")
            padding = self.player.tracks[number].len - new_track.shape[0]
            if padding > 0:
                mean_value = (new_track[0] + new_track[-1]) / 2
                new_track = np.pad(
                    new_track, ((0, padding), (0, 0)), constant_values=mean_value
                )
            else:
                new_track = new_track[: self.player.tracks[number].len]

        new_spectrogram = self.recorder.get_spectrogram(new_track)
        self.player.add_track(number, new_track)
        self.player.play(number)
        self.tracks[number].set_track(new_track, new_spectrogram)
        self.tracks[number].on_playing_start()
        self.rescale_tracks()

    def update_progress(self, _):
        reference = self.player.get_max_frame()
        if reference is None:
            reference = 1
        self.progress_bar = self.player.get_max_progress() / reference

    def watch_for_changes(self, _):
        for track in self.tracks.values():
            if track.watch_file is not None:
                last_changed = os.stat(track.watch_file).st_mtime
                if last_changed > track.watch_file_last_changed:
                    try:
                        self.refresh_track(track.number)
                        track.watch_file_last_changed = last_changed
                    except RuntimeError:
                        pass

    def autosave(self, _):
        if not os.path.isdir("autosaves"):
            os.mkdir("autosaves")
        if len(self.player.tracks) > 0:
            autosaves = glob.glob("autosaves/autosave.[0-6].looper")
            autosaves.sort(key=lambda x: int(x.split(".")[1]), reverse=True)
            for f in autosaves:
                number = int(f.split(".")[1])
                next_number = number + 1
                os.replace(f, "autosaves/autosave.{}.looper".format(next_number))
            self.save("./autosaves", "autosave.0.looper", autosave=True)

    def cleanup(self):
        self.player.stop()
        self.recorder.stop()


class LooperApp(App):
    def __init__(self):
        super().__init__()
        self.screen = None

    def build(self):
        self.screen = Screen()
        return self.screen

    def cleanup(self):
        if self.screen is not None:
            self.screen.cleanup()
