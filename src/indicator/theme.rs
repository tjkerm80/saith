use iced::Color;

use crate::indicator_state::LinearColor;

pub use crate::indicator_state::WAVEFORM_BAR_COUNT;

/// Fully opaque black pill background.
pub const PILL_BACKGROUND: Color = Color::BLACK;

pub const PILL_WIDTH: f32 = 260.0;
pub const PILL_HEIGHT: f32 = 76.0;
pub const PILL_BORDER_RADIUS: f32 = 38.0;

/// Animation tick interval in milliseconds (~20fps).
pub const ANIMATION_INTERVAL_MILLISECONDS: u64 = 50;

/// Pixels from screen edge for indicator positioning.
pub const EDGE_OFFSET: f32 = 100.0;

/// Waveform visualization constants.
pub const WAVEFORM_BAR_WIDTH: f32 = 4.0;
pub const WAVEFORM_BAR_GAP: f32 = 2.0;
pub const WAVEFORM_TOTAL_WIDTH: f32 =
    (WAVEFORM_BAR_COUNT as f32) * (WAVEFORM_BAR_WIDTH + WAVEFORM_BAR_GAP) - WAVEFORM_BAR_GAP;

/// Converts a GUI-agnostic `LinearColor` to an iced `Color`.
pub fn to_iced_color(color: &LinearColor) -> Color {
    Color::from_rgba(color.red, color.green, color.blue, color.alpha)
}
