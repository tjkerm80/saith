# saith

A local, privacy-first speech-to-text dictation app for Linux.

Saith captures audio from your microphone, transcribes it locally using OpenAI's Whisper model, and types the result via a virtual keyboard. No audio ever leaves your machine.

## Features

- Push-to-talk or toggle-to-talk interaction modes
- Configurable global hotkey
- Voice activity detection with silence trimming
- Custom dictionary with initial prompt biasing and text replacement rules
- Multiple Whisper model sizes (base.en, small.en, large-v3-turbo)
- Two transcription backends: candle (default) and whisper-rs
- Floating status indicator with live waveform visualization during recording and animated traveling wave during processing

## Prerequisites

- Linux with evdev support
- Rust 1.85+ (edition 2024)

### whisper-rs backend additional requirements

The whisper-rs backend compiles whisper.cpp from source and requires:

- A C/C++ compiler (gcc or clang)
- CMake

## Building

### Default (candle backend)

```sh
cargo build --release
```

### Without status indicator

```sh
cargo build --release --no-default-features --features whisper-rs-backend
```

### whisper-rs backend only

```sh
cargo build --release --features whisper-rs-backend --no-default-features
```

### Both backends

```sh
cargo build --release --features "candle-backend whisper-rs-backend"
```

## Configuration

Saith reads its configuration from `~/.config/saith/config.json`. A default configuration file is created on first run.

### Backend selection

Set the `backend` field to choose which transcription backend to use at runtime:

```json
{
  "backend": "candle"
}
```

Valid values are `"candle"` (default) and `"whisper_rs"`. If the configured backend was not compiled in, saith falls back to whichever backend is available.

### Model size

```json
{
  "model_size": "base_english"
}
```

Valid values: `"base_english"`, `"small_english"`, `"large_version3_turbo"`.

### Interaction mode

```json
{
  "interaction_mode": "push_to_talk"
}
```

Valid values: `"push_to_talk"` (hold key to record), `"toggle_to_talk"` (press to start, press again to stop).

### Hotkey

```json
{
  "hotkey": {
    "key_code": "KEY_RIGHTMETA"
  }
}
```

Uses evdev key code names (e.g. `KEY_F12`, `KEY_RIGHTMETA`).

### Dictionary

```json
{
  "dictionary": {
    "initial_prompt": "Claude, Anthropic, Rust",
    "replacement_rules": [
      { "pattern": "chat gpt", "replacement": "ChatGPT" }
    ]
  }
}
```

The `initial_prompt` biases Whisper toward expected vocabulary. Replacement rules are applied as post-processing on the transcription output.

### Status indicator

```json
{
  "indicator": {
    "show": true,
    "position": "bottom_center"
  }
}
```

The floating status indicator shows a colored pulsing dot and a mirrored waveform visualization. During recording, the waveform displays real audio levels scrolling left; during processing and transcribing, it transitions to a traveling sine wave. Set `show` to `false` to disable it.

Valid positions: `"top_left"`, `"top_center"`, `"top_right"`, `"bottom_left"`, `"bottom_center"`, `"bottom_right"`.

#### Wayland limitations

On native Wayland (GNOME/Mutter), the always-on-top window hint may not be respected, so the indicator pill may go behind other windows. For guaranteed always-on-top behavior, run with XWayland:

```sh
WINIT_UNIX_BACKEND=x11 cargo run --release
```

## Installation

```sh
cargo install --path .
```

## Running

```sh
cargo run --release
```

Press the configured hotkey to start dictating. The transcribed text is typed via virtual keyboard input.

Saith handles `SIGTERM` and `SIGINT` for graceful shutdown — any active recording is stopped and all input devices are released cleanly before exit.

### Running as a systemd user service

A systemd unit file is provided in `dist/saith.service` for running saith as a background daemon that starts with your graphical session.

1. Ensure your user is in the `input` group (log out and back in after):
   ```sh
   sudo usermod -aG input $USER
   ```

2. Copy the service file into your systemd user directory:
   ```sh
   cp dist/saith.service ~/.config/systemd/user/
   ```

3. Enable and start the service:
   ```sh
   systemctl --user enable --now saith
   ```

4. Check status:
   ```sh
   systemctl --user status saith
   ```

5. View logs:
   ```sh
   journalctl --user -u saith
   ```

6. Stop the service:
   ```sh
   systemctl --user stop saith
   ```
