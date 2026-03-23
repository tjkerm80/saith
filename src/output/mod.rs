mod typing;

use anyhow::Result;

use crate::configuration::{DictionaryConfiguration, apply_replacement_rules};
use typing::VirtualKeyboard;

/// Abstraction over key event emission, allowing test substitution.
pub trait KeyEventSink {
    fn type_text(&mut self, text: &str) -> Result<()>;
}

/// Owns dictionary replacement rules and a key event sink, providing a single
/// `output()` method that applies replacements then types the result.
pub struct TranscriptionOutput {
    dictionary: DictionaryConfiguration,
    event_sink: Box<dyn KeyEventSink>,
}

impl TranscriptionOutput {
    /// Production constructor — creates a real uinput virtual keyboard.
    pub fn new(dictionary: DictionaryConfiguration) -> Result<Self> {
        let virtual_keyboard = VirtualKeyboard::new()?;
        Ok(Self {
            dictionary,
            event_sink: Box::new(virtual_keyboard),
        })
    }

    /// Constructor that accepts any KeyEventSink implementation, enabling
    /// test substitution without requiring real uinput access.
    #[cfg(test)]
    pub fn with_event_sink(
        dictionary: DictionaryConfiguration,
        event_sink: Box<dyn KeyEventSink>,
    ) -> Self {
        Self {
            dictionary,
            event_sink,
        }
    }

    /// Apply replacement rules, skip empty text, then type the result.
    pub fn output(&mut self, text: &str) -> Result<()> {
        let processed_text = apply_replacement_rules(&self.dictionary, text);

        if processed_text.is_empty() {
            return Ok(());
        }

        self.event_sink.type_text(&processed_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::ReplacementRule;
    use std::sync::{Arc, Mutex};

    /// Test double that records all text passed to `type_text`.
    struct RecordingEventSink {
        typed_texts: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingEventSink {
        fn new(typed_texts: Arc<Mutex<Vec<String>>>) -> Self {
            Self { typed_texts }
        }
    }

    impl KeyEventSink for RecordingEventSink {
        fn type_text(&mut self, text: &str) -> Result<()> {
            self.typed_texts.lock().unwrap().push(text.to_string());
            Ok(())
        }
    }

    fn make_output(
        replacement_rules: Vec<ReplacementRule>,
    ) -> (TranscriptionOutput, Arc<Mutex<Vec<String>>>) {
        let typed_texts = Arc::new(Mutex::new(Vec::new()));
        let sink = RecordingEventSink::new(Arc::clone(&typed_texts));
        let dictionary = DictionaryConfiguration {
            initial_prompt: None,
            replacement_rules,
        };
        let output = TranscriptionOutput::with_event_sink(dictionary, Box::new(sink));
        (output, typed_texts)
    }

    #[test]
    fn applies_replacement_rules_before_typing() {
        let rules = vec![ReplacementRule {
            pattern: vec!["hello".to_string()],
            replacement: "world".to_string(),
        }];
        let (mut output, typed_texts) = make_output(rules);

        output.output("hello there").unwrap();

        let texts = typed_texts.lock().unwrap();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "world there");
    }

    #[test]
    fn empty_text_after_replacement_skips_typing() {
        let rules = vec![ReplacementRule {
            pattern: vec!["remove me".to_string()],
            replacement: String::new(),
        }];
        let (mut output, typed_texts) = make_output(rules);

        output.output("remove me").unwrap();

        let texts = typed_texts.lock().unwrap();
        assert!(texts.is_empty());
    }

    #[test]
    fn empty_rules_passes_text_through() {
        let (mut output, typed_texts) = make_output(vec![]);

        output.output("unchanged text").unwrap();

        let texts = typed_texts.lock().unwrap();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "unchanged text");
    }

    #[test]
    fn multiple_rules_applied_in_order() {
        let rules = vec![
            ReplacementRule {
                pattern: vec!["aaa".to_string()],
                replacement: "bbb".to_string(),
            },
            ReplacementRule {
                pattern: vec!["bbb".to_string()],
                replacement: "ccc".to_string(),
            },
        ];
        let (mut output, typed_texts) = make_output(rules);

        output.output("aaa").unwrap();

        let texts = typed_texts.lock().unwrap();
        assert_eq!(texts.len(), 1);
        // First rule: aaa -> bbb, second rule: bbb -> ccc
        assert_eq!(texts[0], "ccc");
    }

    #[test]
    fn multiple_patterns_in_single_rule() {
        let rules = vec![ReplacementRule {
            pattern: vec!["foo".to_string(), "bar".to_string()],
            replacement: "baz".to_string(),
        }];
        let (mut output, typed_texts) = make_output(rules);

        output.output("foo and bar").unwrap();

        let texts = typed_texts.lock().unwrap();
        assert_eq!(texts.len(), 1);
        assert_eq!(texts[0], "baz and baz");
    }
}
