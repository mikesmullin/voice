"""StyleTTS2 engine wrapper for zero-shot voice cloning."""

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torchaudio
import librosa
import yaml
from munch import Munch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def recursive_munch(d):
    """Convert dict to Munch recursively."""
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


class StyleTTS2Engine:
    """StyleTTS2 TTS engine with zero-shot voice cloning."""
    
    def __init__(self, force_cpu: bool = False):
        """
        Initialize StyleTTS2 engine.
        
        Args:
            force_cpu: Force CPU usage instead of GPU
        """
        self.device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
        self.force_cpu = force_cpu
        self.model = None
        self.sampler = None
        self.to_mel = None
        self.global_phonemizer = None
        self.textclenaer = None
        self.model_params = None
        self.config = None
        self.style_cache = {}  # Cache for computed reference audio styles
        
        # Path to StyleTTS2 repository
        self.style_root = Path(__file__).parent.parent / "tmp" / "StyleTTS2"
        
        # Download NLTK data if needed
        try:
            import nltk
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            import nltk
            nltk.download('punkt_tab', quiet=True)
        if not self.style_root.exists():
            raise FileNotFoundError(
                f"StyleTTS2 repository not found at {self.style_root}. "
                "Please clone it to tmp/StyleTTS2"
            )
        
        # Add StyleTTS2 to path
        sys.path.insert(0, str(self.style_root))
        
    def _initialize_model(self):
        """Initialize the StyleTTS2 model (lazy loading)."""
        if self.model is not None:
            return
        
        from .timing import log
        log(f"[StyleTTS2] Initializing model on device: {self.device}")
        
        # Import StyleTTS2 modules
        from models import build_model
        from Utils.PLBERT.util import load_plbert
        from text_utils import TextCleaner
        from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
        
        # Set seeds for reproducibility
        torch.manual_seed(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(0)
        
        # Initialize mel spectrogram converter
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        
        # Initialize Gruut phonemizer
        import gruut
        log(f"[StyleTTS2] Initializing Gruut phonemizer...")
        # Gruut doesn't need explicit initialization, we'll use it per-text
        
        # Initialize text cleaner
        self.textclenaer = TextCleaner()
        
        # Load config
        config_path = self.style_root / "Models" / "LibriTTS" / "config.yml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"StyleTTS2 config not found at {config_path}. "
                "Please download the LibriTTS model from "
                "https://huggingface.co/yl4579/StyleTTS2-LibriTTS"
            )
        
        self.config = yaml.safe_load(open(config_path))
        
        # Load pretrained ASR model
        log(f"[StyleTTS2] Loading ASR model...")
        ASR_config = self.config.get('ASR_config', False)
        ASR_path = self.config.get('ASR_path', False)
        text_aligner = self._load_ASR_model(ASR_path, ASR_config)
        
        # Load pretrained F0 model
        log(f"[StyleTTS2] Loading F0 model...")
        F0_path = self.config.get('F0_path', False)
        pitch_extractor = self._load_F0_model(F0_path)
        
        # Load BERT model
        log(f"[StyleTTS2] Loading PLBERT model...")
        BERT_path = self.config.get('PLBERT_dir', False)
        
        # Change to StyleTTS2 directory for load_plbert (it uses relative paths)
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(str(self.style_root))
            plbert = load_plbert(BERT_path)
        finally:
            os.chdir(old_cwd)
        
        # Build model
        log(f"[StyleTTS2] Building model...")
        self.model_params = recursive_munch(self.config['model_params'])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]
        
        # Load checkpoint
        log(f"[StyleTTS2] Loading checkpoint...")
        checkpoint_path = self.style_root / "Models" / "LibriTTS" / "epochs_2nd_00020.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"StyleTTS2 checkpoint not found at {checkpoint_path}. "
                "Please download epochs_2nd_00020.pth from "
                "https://huggingface.co/yl4579/StyleTTS2-LibriTTS"
            )
        
        params_whole = torch.load(str(checkpoint_path), map_location='cpu', weights_only=False)
        params = params_whole['net']
        
        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        
        _ = [self.model[key].eval() for key in self.model]
        
        # Initialize diffusion sampler
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )
        
        log(f"[StyleTTS2] Model initialized successfully")
    
    def _load_ASR_model(self, ASR_path, ASR_config):
        """Load ASR model for text alignment."""
        from Utils.ASR.models import ASRCNN
        
        ASR_config = self.style_root / ASR_config
        ASR_path = self.style_root / ASR_path
        
        # Load config
        with open(ASR_config, 'r') as f:
            asr_config = yaml.safe_load(f)
        
        # Build model
        text_aligner = ASRCNN(**asr_config['model_params'])
        
        # Load checkpoint (weights_only=False needed for StyleTTS2 checkpoints)
        checkpoint = torch.load(str(ASR_path), map_location='cpu', weights_only=False)
        
        # Extract model state from training checkpoint if needed
        if 'model' in checkpoint:
            params = checkpoint['model']
        else:
            params = checkpoint
            
        text_aligner.load_state_dict(params)
        text_aligner = text_aligner.to(self.device)
        text_aligner.eval()
        
        return text_aligner
    
    def _load_F0_model(self, F0_path):
        """Load F0 (pitch) extraction model."""
        from Utils.JDC.model import JDCNet
        
        F0_path = self.style_root / F0_path
        
        # Build model (JDC model has fixed parameters)
        pitch_extractor = JDCNet(num_class=1, seq_len=192)
        
        # Load checkpoint (weights_only=False needed for StyleTTS2 checkpoints)
        checkpoint = torch.load(str(F0_path), map_location='cpu', weights_only=False)
        
        # Extract model state from training checkpoint if needed
        if 'net' in checkpoint:
            params = checkpoint['net']
        elif 'model' in checkpoint:
            params = checkpoint['model']
        else:
            params = checkpoint
            
        pitch_extractor.load_state_dict(params)
        pitch_extractor = pitch_extractor.to(self.device)
        pitch_extractor.eval()
        
        return pitch_extractor
    
    def _length_to_mask(self, lengths):
        """Convert lengths to mask."""
        mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
        mask = torch.gt(mask + 1, lengths.unsqueeze(1))
        return mask
    
    def _preprocess(self, wave):
        """Preprocess audio waveform to mel spectrogram."""
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mean, std = -4, 4
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
        return mel_tensor
    
    def compute_style(self, audio_path: str):
        """
        Compute style embedding from reference audio.
        Uses caching to speed up repeated calls with the same reference.
        
        Args:
            audio_path: Path to reference audio file
            
        Returns:
            Style embedding tensor
        """
        # Check cache first
        cache_key = str(Path(audio_path).resolve())
        if cache_key in self.style_cache:
            return self.style_cache[cache_key]
        
        # Compute style embedding
        wave, sr = librosa.load(audio_path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        
        mel_tensor = self._preprocess(audio).to(self.device)
        
        with torch.no_grad():
            ref_s = self.model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self.model.predictor_encoder(mel_tensor.unsqueeze(1))
        
        ref_style = torch.cat([ref_s, ref_p], dim=1)
        
        # Cache the result
        self.style_cache[cache_key] = ref_style
        
        return ref_style
    
    def synthesize(
        self, 
        text: str, 
        ref_audio_path: str,
        alpha: float = 0.3,
        beta: float = 0.7,
        diffusion_steps: int = 5,
        embedding_scale: float = 1.0
    ) -> np.ndarray:
        """
        Synthesize speech from text using reference audio for zero-shot cloning.
        
        Args:
            text: Text to synthesize
            ref_audio_path: Path to reference audio for voice cloning
            alpha: Timbre blending factor (0-1). Higher = more predicted, lower = more reference
            beta: Prosody blending factor (0-1). Higher = more emotional/text-driven
            diffusion_steps: Number of diffusion steps (higher = better quality, slower)
            embedding_scale: Classifier-free guidance scale
            
        Returns:
            Audio data as numpy array (24kHz, float32)
        """
        from .timing import log, get_elapsed
        from nltk.tokenize import word_tokenize
        
        self._initialize_model()
        
        # Compute reference style (with caching)
        cache_key = str(Path(ref_audio_path).resolve())
        using_cache = cache_key in self.style_cache
        if using_cache:
            log(f"[StyleTTS2] Using cached style from: {ref_audio_path}")
        else:
            log(f"[StyleTTS2] Computing style from reference: {ref_audio_path}")
        ref_s = self.compute_style(ref_audio_path)
        
        # Prepare text
        text = text.strip()
        log(f"[StyleTTS2] Phonemizing text with Gruut...")
        
        # Use Gruut to get IPA phonemes (same format as espeak-ng IPA)
        import gruut
        phoneme_parts = []
        for sent in gruut.sentences(text, lang='en-us'):
            for word in sent:
                if word.phonemes:
                    # Join phonemes within a word (no spaces)
                    word_phonemes = ''.join(word.phonemes)
                    phoneme_parts.append(word_phonemes)
            # Add sentence break at end of sentence
            if phoneme_parts and not phoneme_parts[-1].endswith('‖'):
                phoneme_parts.append('‖')
        
        # Join word phonemes with spaces between words
        ps = ' '.join(phoneme_parts)
        
        # Convert to token indices
        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        log(f"[StyleTTS2] Generating speech (alpha={alpha}, beta={beta}, steps={diffusion_steps})...")
        gen_start = get_elapsed()
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = self._length_to_mask(input_lengths).to(self.device)
            
            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2)
            
            # Style diffusion
            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,
                num_steps=diffusion_steps
            ).squeeze(1)
            
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            
            # Blend with reference style
            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]
            
            # Predict duration
            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            
            # Create alignment
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            
            # Encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, 0:-1]
                en = asr_new
            
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            
            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, 0:-1]
                asr = asr_new
            
            out = self.model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
        
        log(f"[StyleTTS2] Generation took {get_elapsed() - gen_start:.2f}s")
        
        # Convert to numpy and remove weird pulse at the end
        audio_data = out.squeeze().cpu().numpy()[..., :-50]
        return audio_data
