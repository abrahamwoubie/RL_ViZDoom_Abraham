import scipy
import scipy.io
from scipy import signal
import scipy.io.wavfile
import numpy as np
import aubio
from scipy import signal # audio processing
from scipy.fftpack import dct
from pydub import AudioSegment
import librosa # library for audio processing

target_position_x = 100
target_position_y = 100
import scipy.io.wavfile

class Extract_Features:

    def Extract_Spectrogram(self,player_pos_x,player_pos_y):

        player=[player_pos_x,player_pos_y]
        target=[target_position_x,target_position_y]

        distance=scipy.spatial.distance.euclidean(player, target)

        self.sample_rate, self.data = scipy.io.wavfile.read('Hello.wav')
        # Spectrogram of .wav file
        self.sample_freq, self.segment_time, self.spec_data = signal.spectrogram(self.data, self.sample_rate)

        if(distance==0):
            factor=1
        else:
            factor=1/distance
        return self.spec_data * factor

    def Extract_MFCC(self,player_pos_x,player_pos_y):
        import sys
        from aubio import source, pvoc, mfcc
        from numpy import vstack, zeros, diff
        player=[player_pos_x,player_pos_y]
        target=[target_position_x,target_position_y]
        distance=scipy.spatial.distance.euclidean(player, target)

        n_filters = 40  # must be 40 for mfcc
        n_coeffs = 13

        source_filename = 'Hello.wav'
        samplerate = 44100
        win_s = 512
        hop_s = 128

        s = source(source_filename, samplerate, hop_s)
        samplerate = s.samplerate
        p = pvoc(win_s, hop_s)
        m = mfcc(win_s, n_filters, n_coeffs, samplerate)

        mfccs = zeros([n_coeffs, ])
        frames_read = 0
        while True:
            samples, read = s()
            spec = p(samples)
            mfcc_out = m(spec)
            mfccs = vstack((mfccs, mfcc_out))
            frames_read += read
            if read < hop_s: break
        if(distance==0):
            factor=1
        else:
            factor=1/distance
        return mfccs*factor

    def Extract_Samples(self,player_pos_x,player_pos_y):

        from pydub import AudioSegment
        player = [player_pos_x, player_pos_y]
        target = [target_position_x, target_position_y]
        distance = scipy.spatial.distance.euclidean(player, target)
        if (distance == 0):
            factor = 1
        else:
            factor = 1 / distance
        sound = AudioSegment.from_mp3("Hello.wav")

        # get raw audio data as a bytestring
        raw_data = sound.raw_data
        # get the frame rate
        sample_rate = sound.frame_rate
        # get amount of bytes contained in one sample
        sample_size = sound.sample_width
        # get channels
        channels = sound.channels
        rawdata = []
        for i in range(100):
                rawdata.append(raw_data[i] * factor)
        return np.array(rawdata)
