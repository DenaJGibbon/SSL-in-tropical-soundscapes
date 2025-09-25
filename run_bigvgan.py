import torch
import bigvgan
import librosa
from meldataset import get_mel_spectrogram

# --- device selection ---
if torch.backends.mps.is_available():
    device = "cpu"   # Apple Metal acceleration
else:
    device = "cpu"

# --- load pretrained BigVGAN ---
model = bigvgan.BigVGAN.from_pretrained(
    "nvidia/bigvgan_v2_24khz_100band_256x",
    use_cuda_kernel=False
)
model.remove_weight_norm()
model = model.eval().to(device)

# --- load wav file and compute mel spectrogram ---
#wav_path = "/Users/denaclink/Desktop/RStudioProjects/gibbonNetR_notgithub/data/trainingclips/female.gibbon/female.gibbon_S10_20180620_080003_286.782649889322_295.327563126734_.wav"
wav_path ="/Users/denaclink/Downloads/CR_01_001.01.wav"
wav_path ="/Users/denaclink/Downloads/denatest.wav"
wav, sr = librosa.load(wav_path, sr=model.h.sampling_rate, mono=True)
wav = torch.FloatTensor(wav).unsqueeze(0)

mel = get_mel_spectrogram(wav, model.h).to(device)

# --- generate waveform from mel ---
with torch.inference_mode():
    wav_gen = model(mel)
wav_gen_float = wav_gen.squeeze(0).cpu()

# --- convert to 16-bit PCM and save ---
import soundfile as sf
sf.write("bigvgan_output.wav", wav_gen_float.numpy().T, model.h.sampling_rate)
print("Saved synthesized audio to bigvgan_output.wav")
