import enum
import traceback

import librosa
import librosa.display
import noisereduce
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from const import *


sd.default.samplerate = SR
sd.default.channels = CHANNELS
sd.default.blocksize = BLOCKSIZE

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

class RecorderException(Exception):
    pass

class State(enum.Enum):
    STOPPED = 0
    WAITING = 1
    RECORDING = 2
    RECORDING_INTERVAL = 3
    PLAYING = 4

class Recorder():
    def __init__(self, start_callback=lambda: None, stop_callback=lambda: None):
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self._state = State.STOPPED
        self.stream = sd.InputStream(callback=self.callback)
        self.time_index = 0
        self.rec_index = 0
        self.start_time = None

        self.start_threshold = DEFAULT_START_THRESHOLD
        self.start_window = max(1, int(DEFAULT_START_WINDOW * SR) // BLOCKSIZE)
        self.silence_threshold = DEFAULT_SILENCE_THRESHOLD
        self.silence_window = max(1, int(DEFAULT_SILENCE_WINDOW * SR) // BLOCKSIZE)
        self.reference_frame = None
        self.latency_adjustment = 0

        self.time_left = None

        self.noise_sample = None

    def _reset_state(self):
        self.volumes = np.zeros(max(self.start_window, self.silence_window), dtype=np.float32)
        self.buffer = np.zeros([SR * 60, CHANNELS], dtype=np.float32)
        self.time_left = None
        self.rec_index = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        print ("State changed to: {}".format(value))

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def forward_to(self, x):
        self.time_index = x

    def wait_for_sound(self, silence_threshold, silence_window, latency_adjustment, reference_frame=None):
        if self.state != State.STOPPED:
            raise RecorderException("Cannot record in state: {}".format(self.state))

        self._reset_state()

        self.start_threshold = 0.02
        self.start_window = 10
        self.silence_threshold = silence_threshold
        self.silence_window = max(1, int(silence_window * SR) // BLOCKSIZE)
        self.reference_frame = reference_frame
        self.latency_adjustment = latency_adjustment

        self.state = State.WAITING

    def record(self, length):
        if self.state != State.STOPPED:
            raise RecorderException("Cannot record in state: {}".format(self.state))

        self._reset_state()

        self.time_left = length
        self.state = State.RECORDING_INTERVAL

    def stop_recording(self):
        self.state = State.STOPPED
        self.stop_callback()

    def playback(self):
        if self.state != State.STOPPED:
            raise RecorderException("Cannot playback in state: {}".format(self.state))
        self.state = State.PLAYING
        sd.play(np.tile(self.buffer[:self.rec_index], (4, 1)))

    def get_raw_data(self):
        if self.state != State.STOPPED:
            raise RecorderException("Cannot get recording in state: {}".format(self.state))
        track = self.buffer[:self.rec_index]
        return track

    def get_postprocessed_data(self, noise_threshold):
        if self.state != State.STOPPED:
            raise RecorderException("Cannot postprocess in state: {}".format(self.state))
        # pad recorder sound up to spectrogram hop_length
        if self.rec_index % HOP_LENGTH != 0:
            self.rec_index += (HOP_LENGTH - self.rec_index % HOP_LENGTH)
        track = self._noise_reduce(self.buffer[:self.rec_index], noise_threshold)
        track, spect = self._even_out(track)
        if self.reference_frame:
            track, spect = self._quantize(track, spect)
            track, spect = self._adjust_to_start_time(track, spect)

        return track, spect

    def get_spectrogram(self, track):
        track = librosa.feature.melspectrogram(y=track[:, 0], sr=SR, n_mels=128, fmax=10000, hop_length=HOP_LENGTH)
        return track

    def _noise_reduce(self, track, noise_threshold):
        track = noisereduce.reduce_noise(
            np.swapaxes(track, 0, 1),
            sr=SR,
            stationary=True,
            n_std_thresh_stationary=noise_threshold,
            y_noise=None if self.noise_sample is None else np.swapaxes(self.noise_sample, 0, 1)
        )
        track = np.swapaxes(track, 0, 1)
        return track

    def _even_out(self, track):
        spect = self.get_spectrogram(track)
        values = []
        adjusted_values = []
        _range = np.array(range(40 if spect.shape[1] % 2 == 0 else 41, min(400, spect.shape[1]), 2))
        for i in _range:
            cut = spect[:, :-i]
            if cut.shape[1] < 8:
                break
            split1, split2 = np.array_split(cut, 2, axis=1)
            diff = np.abs(split2 - split1)
            avg = np.average(diff)
            values.append(avg)
            #adjusted_values.append(avg / np.log(cut.shape[1]))
            adjustment = spect.shape[1] / cut.shape[1]
            adjusted_values.append(avg * adjustment)
        if len(adjusted_values) == 0:
            return track, spect
        adjusted_values = np.array(adjusted_values)
        argmin = np.argmin(adjusted_values)
        cut_amount = _range[argmin]
        cut_pcent = cut_amount / spect.shape[1]
        cut_samples = int(track.shape[0] * cut_pcent)

        times = _range * HOP_LENGTH / SR

        fig, ax = plt.subplots()
        plt.plot(times[:len(values)], values / max(values), label="values")
        plt.plot(times[:len(adjusted_values)], adjusted_values / max(adjusted_values), label="adjusted values")
        plt.legend()
        plt.savefig("prep.png")

        if cut_amount >= spect.shape[1]:
            return track, spect
        else:
            return track[:-cut_samples], spect[:, :-cut_amount]

    def _quantize(self, track, spect):
        quant = self.reference_frame
        if track.shape[0] > self.reference_frame:
            while track.shape[0] > quant + self.reference_frame * 0.5:
                quant += self.reference_frame
        else:
            while track.shape[0] <= 1.5 * (quant // 2):
                quant //= 2
        padding = quant - track.shape[0]
        spect_padding = int((padding / track.shape[0]) * spect.shape[1])
        if (padding >= 0):
            print("Track got padded with {x:.2}s of silence to fit the other tracks".format(
                padding / SR
            ))
            track = np.pad(track, ((0, padding), (0, 0)))
            spect = np.pad(spect, ((0, 0), (0, spect_padding)))
        else:
            print("Track got cut by {x:.2}s to fit the other tracks".format(
                -padding / SR
            ))
            track = track[:padding]
            track = self._attenuate_transition(track)
            if spect_padding > 0:
                spect = spect[:, :spect_padding]

        assert(track.shape[0] == quant)

        return track, spect

    def _adjust_to_start_time(self, track, spect):
        adjustment = (self.start_time - self.latency_adjustment) % track.shape[0]
        spect_adjustment = int(adjustment * spect.shape[1] / track.shape[0])
        track = np.concatenate((track[-adjustment:], track[:-adjustment]))
        spect = np.concatenate((spect[:, -spect_adjustment:], spect[:, :-spect_adjustment]), axis=1)
        return track, spect

    def _attenuate_transition(self, track):
        length = BLOCKSIZE * 2
        half = length // 2
        change = 1 / half

        f = 1.0
        for n in range(len(track) - half, len(track)):
            f -= change
            track[n] = f * track[n] + (1 - f) * track[n - 1]
        f = 0.0
        for n in range(0, half):
            f += change
            track[n] = f * track[n] + (1 - f) * track[n - 1]

        return track

    def callback(self, indata, frames, time, status):
        try:
            if self.state != State.STOPPED:
                volume_current = np.average(np.abs(indata))
                start_window_average = np.average(self.volumes[:self.start_window])
                silence_window_max = np.max(self.volumes[:self.silence_window])
                if self.state == State.WAITING:
                    if volume_current - start_window_average > self.start_threshold:
                        self.state = State.RECORDING
                        self.start_time = self.time_index
                        self.start_callback()
                self.volumes = np.roll(self.volumes, 1)
                self.volumes[0] = volume_current
                if self.state == State.RECORDING:
                    if (
                        (silence_window_max < self.silence_threshold and volume_current < self.silence_threshold)
                        or (self.rec_index + frames > len(self.buffer))
                    ):
                        self.stop_recording()
                    else:
                        self.buffer[self.rec_index:self.rec_index+frames] = indata
                        self.rec_index += frames
                elif self.state == State.RECORDING_INTERVAL:
                    if self.time_left is not None and self.time_left <= 0:
                        self.stop_recording()
                    else:
                        self.buffer[self.rec_index:self.rec_index+frames] = indata
                        self.rec_index += frames
                        self.time_left -= frames
            self.time_index += frames
        except Exception as e:
            print (traceback.format_exception(e))
