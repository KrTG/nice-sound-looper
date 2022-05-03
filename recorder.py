import enum
import librosa
import librosa.display
import noisereduce
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd


SR = 48000
FFT_WIDTH = 2048
HOP_LENGTH = 512
CHANNELS = 1

sd.default.samplerate = SR
sd.default.channels = CHANNELS

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

class RecorderException(Exception):
    pass

class State(enum.Enum):
    STOPPED = 0
    WAITING = 1
    RECORDING = 2
    PLAYING = 3


class Recorder():
    def __init__(self, start_callback=lambda: None, stop_callback=lambda: None):
        self.start_callback = start_callback
        self.stop_callback = stop_callback

        self.volume_start_threshold = 0.02
        self.silence_threshold = 0.012
        self._state = State.STOPPED
        self.stream = sd.InputStream(callback=self.callback)

        self.start_time = 0
        self.reference_frame = None
        self.time_sig = None

    def _reset(self):
        self.volumes = np.zeros(80, dtype=np.float32)
        self.buffer = np.zeros([SR * 60, CHANNELS], dtype=np.float32)
        self.rec_index = 0

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        print ("State changed to: {}".format(value))

    def wait(self, start_time, reference_frame=None, time_sig=4):
        print(self.stream.active)
        if self.state != State.STOPPED:
            raise RecorderException("Cannot record in state: {}".format(self.state))
        self.start_time = start_time
        self.reference_frame = reference_frame
        self.time_sig = time_sig
        self.state = State.WAITING
        self._reset()
        print("STARTING STREAM")
        if self.stream.stopped:
            self.stream.start()

    def stop(self):
        self.state = State.STOPPED
        self.stop_callback()

    def playback(self):
        if self.state != State.STOPPED:
            raise RecorderException()("Cannot playback in state: {}".format(self.state))
        self.state = State.PLAYING
        sd.play(np.tile(self.buffer[:self.rec_index], (4, 1)))

    def postprocess(self):
        if self.state != State.STOPPED:
            raise RecorderException()("Cannot postprocess in state: {}".format(self.state))
        # pad recorder sound up to spectrogram hop_length
        if self.rec_index % HOP_LENGTH != 0:
            self.rec_index += (HOP_LENGTH - self.rec_index % HOP_LENGTH)
        track = noisereduce.reduce_noise(
            self.buffer[:self.rec_index].flatten(),
            sr=SR,
            stationary=True,
            n_std_thresh_stationary=1
        )
        track = np.expand_dims(track, 1)
        #track, spect = self._even_out(self.buffer[:self.rec_index])
        track, spect = self._even_out(track)
        if self.reference_frame:
            track, spect = self._quantize(track, spect)
            track, spect = self._adjust_to_start_time(track, spect)
        else:
            return track, spect

        return track, spect

    def _even_out(self, track):
        spect = librosa.feature.melspectrogram(y=track.flatten(), sr=SR, n_mels=128, fmax=10000, hop_length=HOP_LENGTH)
        #spect = librosa.decompose.nn_filter(spect, aggregate=np.median)
        values = []
        _range = np.array(range(40 if spect.shape[1] % 2 == 0 else 41, min(400, spect.shape[1] - 40), 2))
        for i in _range:
            cut = spect[:, :-i]
            split1, split2 = np.array_split(cut, 2, axis=1)
            diff = np.abs(split2 - split1)
            avg = np.average(diff)
            values.append(avg)
        values = np.array(values)
        argmin = np.argmin(values)
        cut_amount = _range[argmin]
        cut_pcent = cut_amount / spect.shape[1]
        cut_samples = int(track.shape[0] * cut_pcent)

        times = _range * 512 / SR

        fig, ax = plt.subplots()
        plt.plot(times, values)
        plt.savefig("prep.png")

        if cut_amount >= spect.shape[1]:
            return track, spect
        else:
            return track[:-cut_samples], spect[:, :-cut_amount]

    def _quantize(self, track, spect):
        quant = self.reference_frame
        if track.shape[0] > self.reference_frame:
            while track.shape[0] > quant:
                quant += self.reference_frame
        else:
            while track.shape[0] <= quant // 2:
                quant //= 2
        padding = quant - track.shape[0]
        track = np.pad(track, ((0, padding), (0, 0)))
        spect = np.pad(spect, ((0, 0), (0, int(padding * spect.shape[1] / track.shape[0]))))

        assert(track.shape[0] == quant)

        return track, spect


    def _adjust_to_start_time(self, track, spect):
        adjustment = self.start_time % track.shape[0]
        spect_adjustment = int(adjustment * spect.shape[1] / track.shape[0])
        print(spect_adjustment)
        print(spect.shape[1])
        track = np.concatenate((track[-adjustment:], track[:-adjustment]))
        spect = np.concatenate((spect[:, -spect_adjustment:], spect[:, :-spect_adjustment]), axis=1)
        print(spect)
        return track, spect


    def callback(self, indata, frames, time, status):
        try:
            if self.state == State.STOPPED:
                return
            volume_current = np.average(np.abs(indata))
            volume_10_average = np.average(self.volumes[:10])
            volume_80_max = np.max(self.volumes)
            if self.state == State.WAITING:
                if volume_current - volume_10_average > self.volume_start_threshold:
                    self.state = State.RECORDING
                    self.start_callback()
                else:
                    self.start_time += frames
            self.volumes = np.roll(self.volumes, 1)
            self.volumes[0] = volume_current
            if self.state == State.RECORDING:
                if (volume_80_max < self.silence_threshold and volume_current < self.silence_threshold) or self.rec_index + frames > len(self.buffer):
                    self.stop()
                else:
                    self.buffer[self.rec_index:self.rec_index+frames] = indata
                    self.rec_index += frames
        except Exception as e:
            print (e)
