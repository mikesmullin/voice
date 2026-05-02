# Voice CLI Operator Guide

This file is the fast operator manual for agents using the `voice` CLI.

## Default Operating Guidance

- Prefer Piper unless the user explicitly wants a Kokoro voice.
- Do not reach for `serve` or `hot` by default. Piper is usually fast enough without them.
- Use Kokoro when you specifically want one of its voices, and expect higher latency unless the server is already warm.
- Kokoro voices download automatically on first use.

## Engine Selection

### Use Piper When

- You need fast local playback from the CLI.
- You want offline CPU-friendly synthesis.
- You are scripting or calling the tool repeatedly from automation.
- You do not want to manage a background server.

### Favorite Piper Voices

- `lessac`: safest default, configured fallback voice, American female Piper pick
- `norman`: American male Piper pick with a faster tuned delivery at `speed: 1.7`
- `cori`: preferred British female Piper voice with a faster tuned delivery at `speed: 1.7`
- `alan`: preferred British male Piper voice with a moderately faster tuned delivery at `speed: 1.4`
- `aru`: preferred British male Piper alternative at the default pacing
- `vctk`: preferred British male Piper alternative at the default pacing

### Use Kokoro When

- You need a specific Kokoro voice, and you know why.
- You are willing to trade startup time, or a medium size block of memory reserved, for that voice.
- You can keep a warm process running with `voice serve` and issue requests with `voice hot`.

### Favorite Kokoro Voices

- `bella`: main American female Kokoro pick when you want a Kokoro voice rather than a Piper default
- `heart`: alternate American female Kokoro favorite
- `michael`: clear authoritative male radio voice
- `daniel`: best male broadcaster voice

## Core Commands

### Direct Synthesis

```bash
voice [options] <preset> <text>
```

Examples:

```bash
voice lessac "Piper is the default operator choice."
voice bella "Use Kokoro when that voice is specifically wanted."
```

### Save to File

```bash
voice [options] <preset> <text>
```

Examples:

```bash
voice -o norman.wav norman "Write this to disk with Piper."
voice -o heart.wav heart "Write this to disk with Kokoro."
```

### Adjust Volume

Use `--gain=<value>` to scale the synthesized voice volume before playback or saving.

```bash
voice --gain=0.6 lessac "Quieter playback."
voice --gain=1.4 -o daniel.wav daniel "Louder output file."
```

### Read from STDIN

Use `-` as the text argument when stdin should supply the spoken text.

```bash
printf '%s' "Piper stdin example." | voice cori -
printf '%s' "Kokoro stdin example." | voice --config ./src/config.yaml daniel -
```

### List Presets

```bash
voice --list
```

### Inspect a Preset

```bash
voice --info <preset>
```

Examples:

```bash
voice --info alan
voice --info michael
```

### Custom Config

```bash
voice [options] <preset> <text>
```

Example:

```bash
voice --config ./src/config.yaml aru "Use an explicit config path."
voice --config ./src/config.yaml bella "Use Kokoro from an explicit config path."
```

## Low-Latency Kokoro Workflow

Use this only when Kokoro latency matters.

### Start the Server

```bash
voice serve
```

### Send Hot Requests

```bash
voice hot bella "Warm Kokoro request."
voice hot daniel "Second warm Kokoro request."
```

### Important Behavior

- `voice hot` is mainly for a running server workflow.
- If the server is unavailable, the CLI falls back to the configured fallback voice rather than transparently synthesizing the requested preset.
- In the current config that fallback voice is `lessac`.
- This is another reason to prefer direct Piper commands unless you intentionally want Kokoro hot mode.

## Stingers

Stingers are optional WAV sound effects played before speech.

### Use a Stinger

```bash
voice --stinger alert ada "Alert with stinger"
voice hot ada "Hot mode alert" --stinger error
```

### Stinger Notes

- Stingers only matter for playback. They are not used when saving with `-o`.
- If the stinger name is missing from config, it is ignored.
- `ada` is the clearest preset to use when you need a stinger example.

## CPU and GPU Guidance

- Piper is already CPU-oriented; you usually do not need extra flags.
- `--cpu` is most relevant when you want to force Kokoro off GPU.

Examples:

```bash
voice lessac "CPU-friendly default path."
voice --cpu bella "Force Kokoro onto CPU."
```

## Practical Operator Rules

- Start with a Piper preset unless the request names a Kokoro voice.
- If the user asks for a recommended voice, suggest `lessac`, `norman`, `cori`, `alan`, `aru`, or `vctk` first.
- If the user asks for highest convenience, do not start server mode.
- If the user asks for repeated low-latency Kokoro output, then use `voice serve` and `voice hot`.
- When in doubt, use `voice --info <preset>` before synthesizing.
- Use `voice --list` if you need to discover available presets from the installed config.