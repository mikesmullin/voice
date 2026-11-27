#!/usr/bin/env python3
"""GPU synthesis for Audio8-TTS-Preview-0.1b (PyTorch checkpoint, bf16/CUDA).

Counterpart to the CPU path (`arktts_runtime.cli`, INT8 ONNX): this runs the
full-precision model on CUDA. Reference voice comes from pre-registered
codec codes (e.g. vendor/audio8-runtime/voices/<name>/codes.npy) plus the
reference transcript -- the same pair `register_default_voice.py` /
`scripts/audio8-register-voice.sh` produce -- or from a fresh reference
recording (`--ref-audio`), which the model encodes on the fly at full
precision.

Kept as a standalone script (not a module) so the Zig side can invoke it
with an absolute path and no package/cwd assumptions:
  <gpu-venv>/bin/python /path/to/audio8_gpu_synth.py --model-dir ... --codes ... --ref-text ... --text ... --output ...

Setup: ./scripts/setup-audio8-gpu.sh
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="PyTorch checkpoint dir (config.json, model.safetensors, codec.pth, ...)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--codes", type=Path,
                       help="registered reference codes (.npy, [10, T]) for the voice")
    group.add_argument("--ref-audio", type=Path,
                       help="reference recording (wav/flac/ogg, any rate); encoded fresh on GPU")
    parser.add_argument("--ref-text", required=True,
                        help="exact transcript of the reference (required for cloning)")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from transformers import AutoModel, AutoProcessor

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_dir, trust_remote_code=True, dtype=dtype).eval().to(device)
    print(f"[audio8-gpu] model loaded on {device} in {time.time()-t0:.1f}s", flush=True)

    torch.manual_seed(int(args.seed))
    if device == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    ref_kwargs = {"reference_text": args.ref_text}
    if args.codes is not None:
        ref_kwargs["reference_codes"] = [str(args.codes)]
    else:
        ref_kwargs["reference_audio"] = [str(args.ref_audio)]

    t1 = time.time()
    inputs = processor(text=[args.text], return_tensors="pt", **ref_kwargs)
    inputs = {n: v.to(device) for n, v in inputs.items()}
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            do_sample=True,
            return_dict_in_generate=True,
        )
    print(f"[audio8-gpu] generated {tuple(output.codes.shape)} in {time.time()-t1:.1f}s", flush=True)

    waveforms, lengths = model.decode_audio(output.codes)
    audio = waveforms[0, : int(lengths[0])].float().cpu().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, model.config.codec_sample_rate)
    print(f"[audio8-gpu] wrote {args.output} ({len(audio)/model.config.codec_sample_rate:.2f}s audio, "
          f"{time.time()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
