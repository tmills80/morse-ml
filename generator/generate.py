import re
import uuid
import random
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import math

import yaml
from functools import reduce
import sys
import os


class Config():
    def __init__(self, file_name):
        with open(file_name) as f:
            self.config = yaml.safe_load(f.read())

    def value(self, key):
        return reduce(lambda c, k: c[k], key.split('.'), self.config)

    def __repr__(self):
        return str(self.config)


class Morse():
    """Generates morse audio files from text. Can add noise to desired
       SNR level"""
    code = {
        '!': '-.-.--',
        '$': '...-..-',
        "'": '.----.',
        '(': '-.--.',  # <KN>
        ')': '-.--.-',
        ',': '--..--',
        '-': '-....-',
        '.': '.-.-.-',
        '/': '-..-.',
        '0': '-----',
        '1': '.----',
        '2': '..---',
        '3': '...--',
        '4': '....-',
        '5': '.....',
        '6': '-....',
        '7': '--...',
        '8': '---..',
        '9': '----.',
        ':': '---...',
        ';': '-.-.-.',
        '>': '.-.-.',   # <AR>
        '<': '.-...',   # <AS>
        '{': '....--',  # <HM>
        '&': '..-.-',   # <INT>
        '%': '...-.-',  # <SK>
        '}': '...-.',   # <VE>
        '=': '-...-',   # <BT>
        '?': '..--..',
        '@': '.--.-.',
        'A': '.-',
        'B': '-...',
        'C': '-.-.',
        'D': '-..',
        'E': '.',
        'F': '..-.',
        'G': '--.',
        'H': '....',
        'I': '..',
        'J': '.---',
        'K': '-.-',
        'L': '.-..',
        'M': '--',
        'N': '-.',
        'O': '---',
        'P': '.--.',
        'Q': '--.-',
        'R': '.-.',
        'S': '...',
        'T': '-',
        'U': '..-',
        'V': '...-',
        'W': '.--',
        'X': '-..-',
        'Y': '-.--',
        'Z': '--..',
        '\\': '.-..-.',
        '_': '..--.-',
        '~': '.-.-',
        ' ': '_',
        '\n': '_'
    }

    def __init__(self, text, file_name=None, SNR_dB=20, f_code=600, Fs=8000,
                 code_speed=20, length_seconds=4, total_seconds=8,
                 play_sound=True, variation=None):
        self.text = text.upper()              # text to be converted in here
        self.file_name = file_name            # file name of generated WAV file
        self.SNR_dB = SNR_dB                  # target SNR in dB
        self.f_code = f_code                  # CW tone frequency
        self.Fs = Fs                          # Sampling frequency
        self.code_speed = code_speed          # code speed in WPM
        # caps the CW generation to this length in seconds
        self.length_seconds = length_seconds
        self.total_seconds = total_seconds    # pads to the total length
        self.play_sound = play_sound          # If true, play the audio
        # max variation of component length
        self.variation = variation

        self.len = self.len_str_in_dits(self.text)
        self.morsecode = []  # store audio representation here
        self.t = np.linspace(0., 1.2/self.code_speed, num=int(self.Fs *
                             1.2/self.code_speed), endpoint=True,
                             retstep=False)
        print(self.t.shape)
        self.Dit = np.sin(2*np.pi*self.f_code*self.t)
        self.ssp = np.zeros(len(self.Dit))
        # one Dah of time is 3 times  dit time
        self.t2 = np.linspace(0., 3*1.2/self.code_speed, num=3 *
                              int(self.Fs*1.2/self.code_speed), endpoint=True,
                              retstep=False)
        # Dah = np.concatenate((Dit,Dit,Dit))
        self.Dah = np.sin(2*np.pi*self.f_code*self.t2)
        self.lsp = np.zeros(len(self.Dah))

    def len_dits(self, cws):
        """Return the length of cw_string in dit units, including spaces. """
        val = 0
        for ch in cws:
            if ch == '.':  # dit len
                val += 1
            if ch == '-':  # dah len
                val += 3
            if ch == '_':  # word space
                val += 4
            val += 1  # el space is one dit
        val += 2     # char space = 3  (el space + 2)
        return val

    def len_chr_in_dits(self, ch):
        s = Morse.code[ch]
        return self.len_dits(s)

    def len_str_in_dits(self, s):
        """Return length of string in dit units"""
        if len(s) == 0:
            return 0
        val = 0
        for ch in s:
            val += self.len_chr_in_dits(ch)
        return val-3  # remove last char space at end of string

    def len_str_in_secs(self, s):
        dit = 1.2/self.code_speed
        len_in_dits = self.len_str_in_dits(s)
        return dit*len_in_dits

    def add_variation(self, element, repeats=1):
        var = random.uniform(-self.variation, self.variation)
        diff = math.floor(math.fabs(len(self.Dit) * repeats * var))
        if var < 0:
            self.morsecode = self.morsecode[:-diff]
        else:
            self.morsecode = np.concatenate(
                (self.morsecode, element[:diff]))

    def generate_audio(self):
        for ch in self.text:
            s = Morse.code[ch]
            for el in s:
                if el == '.':
                    self.morsecode = np.concatenate((self.morsecode, self.Dit))

                    if self.variation:
                        self.add_variation(self.Dit)
                elif el == '-':
                    self.morsecode = np.concatenate((self.morsecode, self.Dah))

                    if self.variation:
                        self.add_variation(self.Dah)
                elif el == '_':
                    self.morsecode = np.concatenate(
                        (self.morsecode, self.ssp, self.ssp, self.ssp))

                    if self.variation:
                        self.add_variation(self.ssp, 3)

                self.morsecode = np.concatenate((self.morsecode, self.ssp))
                if self.variation:
                    self.add_variation(self.ssp)

            self.morsecode = np.concatenate(
                (self.morsecode, self.ssp, self.ssp))

            if self.variation:
                self.add_variation(self.ssp, 2)

    def SNR(self):
        if self.SNR_dB is not None:
            SNR_linear = 10.0 ** (self.SNR_dB/10.0)
            power = self.morsecode.var()
            noise_power = power/SNR_linear
            noise = np.sqrt(noise_power) * \
                np.random.normal(0, 1, len(self.morsecode))
            self.morsecode = noise + self.morsecode

    def pad_start(self):
        dit = 1.2/self.code_speed  # dit duration in seconds
        txt_dits = self.len  # calculate the length of text in dit units
        tot_len = txt_dits * dit  # calculate total text length in seconds
        if (self.length_seconds - tot_len < 0):
            raise ValueError(f"text length {tot_len:.2f} exceeds audio length"
                             f"{self.length_seconds:.2f}")
        # calculate how many dits will fit in with the text
        pad_dits = int((self.length_seconds - tot_len)/dit)
        # pad with random space to fit proper length
        pad = random.randint(0, pad_dits)
        for i in range(pad):
            self.morsecode = np.concatenate((self.morsecode, self.ssp))

    def pad_end(self):
        if self.total_seconds:
            append_length = self.Fs*self.total_seconds - len(self.morsecode)
            if (append_length > 0):
                self.morsecode = np.concatenate(
                    (self.morsecode, np.zeros(append_length)))

    def normalize(self):
        self.morsecode = self.morsecode/max(self.morsecode)

    def audio(self):
        """Generate audio file using other functions"""
        self.morsecode = []
        self.pad_start()
        self.generate_audio()
        self.pad_end()
        self.SNR()
        self.normalize()
        if self.play_sound:
            sd.play(self.morsecode, self.Fs)
        if self.file_name:
            write(self.file_name, self.Fs, self.morsecode)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            print(f"in __exit__:{exc_type} {exc_value} {traceback}")

    def generate_fragments(self):
        """ Yield string fragments shorter than self.length_seconds until end
        of self.text"""
        mybuf = ''
        for nextchar in self.text:
            mybuf += nextchar
            len_str_in_secs = self.len_str_in_secs(mybuf)
            if len_str_in_secs < self.length_seconds:
                continue
            elif len_str_in_secs >= self.length_seconds:
                yield mybuf[:-1], self.len_str_in_secs(mybuf[:-1])
                mybuf = nextchar
            elif len_str_in_secs < 0.:
                raise ValueError('ERROR: parse_string should '
                                 'never have negative length strings')

        yield mybuf[:], self.len_str_in_secs(mybuf[:])


# 24487 words in alphabetical order
# https://svnweb.freebsd.org/csrg/share/dict/words?view=co&content-type=text/plain
#


def generate_dataset(config):
    "generate audio dataset from a corpus of words"
    directory = config.value('model.directory')
    corpus_file = config.value('generator.corpus')
    fnTrain = directory + "/" + config.value('model.fnTrain')
    fnAudio = directory + "/" + config.value('model.fnAudio')
    code_speed = config.value('generator.code_speed')
    length_seconds = config.value('generator.length_seconds')
    error_counter = 0

    try:
        os.makedirs(fnAudio, exist_ok=True)
    except OSError:
        print("Error: cannot create ", directory)

    wordcount = 0
    with open(corpus_file) as corpus:
        # words = corpus.read().split("\n")
        text = corpus.read()
        # generate training material in all WPM speeds in the list
        for speed in code_speed:
            wordcount = 0
            with (Morse(text, code_speed=speed, length_seconds=length_seconds,
                        total_seconds=length_seconds+4) as m1,
                  open(fnTrain, 'w') as mf):
                for line, duration in m1.generate_fragments():
                    # remove extra characters
                    phrase = re.sub(r'[\'&/\n]', '', line)
                    if len(phrase) <= 1:
                        continue
                    print(
                        f"speed:{speed} of {len(code_speed)} phrase:{phrase} "
                        f"dur:{duration}")
                    audio_file = "{}WPM{}-{}.wav".format(
                        fnAudio, speed, uuid.uuid4().hex)
                    try:
                        m = Morse(
                            phrase, audio_file, None, 600, 8000, speed,
                            length_seconds, length_seconds + 4, False)
                        m.audio()
                        mf.write(audio_file+',"'+phrase+'"\n')
                        wordcount += 1
                    except Exception as err:
                        print(f"ERROR: {audio_file} {err}")
                        error_counter += 1
                        continue
                print(f"completed {wordcount} files for speed:{speed}, "
                      f"with {error_counter} errors")


def main(argv):
    if len(argv) < 2:
        print("usage: python generate.py <model-config.yaml>")
        exit(1)
    print(argv)
    configs = Config(argv[1])
    generate_dataset(configs)


if __name__ == "__main__":
    main(sys.argv)
