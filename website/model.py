import sys
import os
import torch
from unidecode import unidecode
import numpy as np
from pathlib import Path
from IPython.display import Audio

print(os.path.abspath(os.path.join('..', 'WaveRNN/models')))
sys.path.append(os.path.abspath(os.path.join('..', 'WaveRNN')))

from models.fatchord_version import WaveRNN
from models.tacotron import Tacotron
from utils.text.symbols import symbols
from utils.text import text_to_sequence
from utils import hparams as hp

if not hp.is_configured():
    hp.configure(os.path.abspath(os.path.join('..', 'WaveRNN/hparams.py')))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

vocoder = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                    fc_dims=hp.voc_fc_dims,
                    bits=hp.bits,
                    pad=hp.voc_pad,
                    upsample_factors=hp.voc_upsample_factors,
                    feat_dims=hp.num_mels,
                    compute_dims=hp.voc_compute_dims,
                    res_out_dims=hp.voc_res_out_dims,
                    res_blocks=hp.voc_res_blocks,
                    hop_length=hp.hop_length,
                    sample_rate=hp.sample_rate,
                    mode=hp.voc_mode
                ).to(device)

# vocoder.load(os.path.abspath(os.path.join('..', 'WaveRNN/checkpoints/ljspeech_mol.wavernn/latest_weights.pyt')))
vocoder.load(os.path.abspath(os.path.join('..', 'WaveRNN/checkpoints/ljspeech_mol.wavernn/wave_step300k_weights.pyt')))

# Instantiate Tacotron Model
tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                        num_chars=len(symbols),
                        encoder_dims=hp.tts_encoder_dims,
                        decoder_dims=hp.tts_decoder_dims,
                        n_mels=hp.num_mels,
                        fft_bins=hp.num_mels,
                        postnet_dims=hp.tts_postnet_dims,
                        encoder_K=hp.tts_encoder_K,
                        lstm_dims=hp.tts_lstm_dims,
                        postnet_K=hp.tts_postnet_K,
                        num_highways=hp.tts_num_highways,
                        dropout=hp.tts_dropout,
                        stop_threshold=hp.tts_stop_threshold).to(device)

tts_model.load(os.path.abspath(os.path.join('..', 'WaveRNN/checkpoints/ljspeech_lsa_smooth_attention.tacotron/latest_weights.pyt')))

def synthesize_speech(input_text):
    encoded_text = unidecode(input_text)

    inputs = [text_to_sequence(encoded_text.strip(), hp.tts_cleaner_names)]
    print(f"Encoded inputs: {inputs}")
    
    for i, x in enumerate(inputs, 1):

        print(f'\n| Generating {i}/{len(inputs)}')
        _, m, attention = tts_model.generate(x)
        # Fix mel spectrogram scaling to be from 0 to 1
        m = (m + 4) / 8
        np.clip(m, 0, 1, out=m)

        m = torch.tensor(m).unsqueeze(0)

        # Generate waveform from mel spectrogram using WaveRNN
        output = vocoder.generate(m, hp.voc_gen_batched, hp.voc_target, hp.voc_overlap, hp.mu_law)

        # return Audio(output.astype(np.float32), rate=hp.sample_rate)
        return output.astype(np.float32)

    print('\n\nDone.\n')