use anyhow::{Context, Result};
use evdev::uinput::VirtualDevice;
use evdev::{AttributeSet, KeyCode, KeyEvent};
use std::collections::HashMap;

/// A virtual keyboard that types text via uinput.
///
/// Created once at startup and reused for all transcriptions. The user must be
/// in the `input` group or have a udev rule granting write access to `/dev/uinput`.
pub struct VirtualKeyboard {
    device: VirtualDevice,
    keymap: HashMap<char, KeyMapping>,
}

#[derive(Clone, Copy)]
struct KeyMapping {
    key_code: KeyCode,
    shift: bool,
}

impl super::KeyEventSink for VirtualKeyboard {
    fn type_text(&mut self, text: &str) -> anyhow::Result<()> {
        self.type_text_impl(text)
    }
}

impl VirtualKeyboard {
    /// Create a new virtual keyboard device registered with uinput.
    pub fn new() -> Result<Self> {
        let keymap = build_keymap();

        let mut key_capabilities = AttributeSet::<KeyCode>::new();
        key_capabilities.insert(KeyCode::KEY_LEFTSHIFT);
        for mapping in keymap.values() {
            key_capabilities.insert(mapping.key_code);
        }

        let device = VirtualDevice::builder()
            .context("Failed to open /dev/uinput (are you in the 'input' group?)")?
            .name("saith-keyboard")
            .with_keys(&key_capabilities)
            .context("Failed to register key capabilities")?
            .build()
            .context("Failed to create virtual keyboard device")?;

        // Give the system time to register the new device
        std::thread::sleep(std::time::Duration::from_millis(50));

        Ok(Self { device, keymap })
    }

    fn type_text_impl(&mut self, text: &str) -> Result<()> {
        for character in text.chars() {
            self.type_character(character)?;
        }
        Ok(())
    }

    fn type_character(&mut self, character: char) -> Result<()> {
        let mapping = self.keymap.get(&character);

        let mapping = match mapping {
            Some(mapping) => *mapping,
            None => {
                // Skip characters we can't type (emoji, unicode, etc.)
                return Ok(());
            }
        };

        if mapping.shift {
            let shift_down = KeyEvent::new_now(KeyCode::KEY_LEFTSHIFT, 1);
            self.device.emit(&[shift_down.into()])?;
        }

        let key_down = KeyEvent::new_now(mapping.key_code, 1);
        let key_up = KeyEvent::new_now(mapping.key_code, 0);
        self.device.emit(&[key_down.into(), key_up.into()])?;

        if mapping.shift {
            let shift_up = KeyEvent::new_now(KeyCode::KEY_LEFTSHIFT, 0);
            self.device.emit(&[shift_up.into()])?;
        }

        // Small delay between keystrokes for reliability
        std::thread::sleep(std::time::Duration::from_micros(500));

        Ok(())
    }
}

/// Build the US keyboard layout character-to-keycode mapping.
fn build_keymap() -> HashMap<char, KeyMapping> {
    let mut keymap = HashMap::new();

    let plain = |key_code| KeyMapping {
        key_code,
        shift: false,
    };
    let shifted = |key_code| KeyMapping {
        key_code,
        shift: true,
    };

    // Letters (lowercase = plain, uppercase = shifted)
    let letters = [
        ('a', KeyCode::KEY_A),
        ('b', KeyCode::KEY_B),
        ('c', KeyCode::KEY_C),
        ('d', KeyCode::KEY_D),
        ('e', KeyCode::KEY_E),
        ('f', KeyCode::KEY_F),
        ('g', KeyCode::KEY_G),
        ('h', KeyCode::KEY_H),
        ('i', KeyCode::KEY_I),
        ('j', KeyCode::KEY_J),
        ('k', KeyCode::KEY_K),
        ('l', KeyCode::KEY_L),
        ('m', KeyCode::KEY_M),
        ('n', KeyCode::KEY_N),
        ('o', KeyCode::KEY_O),
        ('p', KeyCode::KEY_P),
        ('q', KeyCode::KEY_Q),
        ('r', KeyCode::KEY_R),
        ('s', KeyCode::KEY_S),
        ('t', KeyCode::KEY_T),
        ('u', KeyCode::KEY_U),
        ('v', KeyCode::KEY_V),
        ('w', KeyCode::KEY_W),
        ('x', KeyCode::KEY_X),
        ('y', KeyCode::KEY_Y),
        ('z', KeyCode::KEY_Z),
    ];

    for (character, key_code) in letters {
        keymap.insert(character, plain(key_code));
        keymap.insert(character.to_ascii_uppercase(), shifted(key_code));
    }

    // Numbers and their shifted symbols
    let numbers = [
        ('1', '!', KeyCode::KEY_1),
        ('2', '@', KeyCode::KEY_2),
        ('3', '#', KeyCode::KEY_3),
        ('4', '$', KeyCode::KEY_4),
        ('5', '%', KeyCode::KEY_5),
        ('6', '^', KeyCode::KEY_6),
        ('7', '&', KeyCode::KEY_7),
        ('8', '*', KeyCode::KEY_8),
        ('9', '(', KeyCode::KEY_9),
        ('0', ')', KeyCode::KEY_0),
    ];

    for (number, symbol, key_code) in numbers {
        keymap.insert(number, plain(key_code));
        keymap.insert(symbol, shifted(key_code));
    }

    // Punctuation and symbols
    let punctuation = [
        (' ', KeyCode::KEY_SPACE, false),
        ('\n', KeyCode::KEY_ENTER, false),
        ('\t', KeyCode::KEY_TAB, false),
        ('-', KeyCode::KEY_MINUS, false),
        ('_', KeyCode::KEY_MINUS, true),
        ('=', KeyCode::KEY_EQUAL, false),
        ('+', KeyCode::KEY_EQUAL, true),
        ('[', KeyCode::KEY_LEFTBRACE, false),
        ('{', KeyCode::KEY_LEFTBRACE, true),
        (']', KeyCode::KEY_RIGHTBRACE, false),
        ('}', KeyCode::KEY_RIGHTBRACE, true),
        ('\\', KeyCode::KEY_BACKSLASH, false),
        ('|', KeyCode::KEY_BACKSLASH, true),
        (';', KeyCode::KEY_SEMICOLON, false),
        (':', KeyCode::KEY_SEMICOLON, true),
        ('\'', KeyCode::KEY_APOSTROPHE, false),
        ('"', KeyCode::KEY_APOSTROPHE, true),
        ('`', KeyCode::KEY_GRAVE, false),
        ('~', KeyCode::KEY_GRAVE, true),
        (',', KeyCode::KEY_COMMA, false),
        ('<', KeyCode::KEY_COMMA, true),
        ('.', KeyCode::KEY_DOT, false),
        ('>', KeyCode::KEY_DOT, true),
        ('/', KeyCode::KEY_SLASH, false),
        ('?', KeyCode::KEY_SLASH, true),
    ];

    for (character, key_code, is_shifted) in punctuation {
        keymap.insert(
            character,
            if is_shifted {
                shifted(key_code)
            } else {
                plain(key_code)
            },
        );
    }

    keymap
}
