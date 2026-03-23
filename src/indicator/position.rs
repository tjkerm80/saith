use crate::configuration::IndicatorPosition;
use iced::Point;

use super::theme::{EDGE_OFFSET, PILL_HEIGHT, PILL_WIDTH};

/// Calculate the window position for a given preset.
///
/// Uses a hardcoded 1920x1080 fallback screen size since iced does not expose
/// monitor dimensions before window creation. Window managers handle
/// out-of-bounds positioning gracefully by clamping to the visible area.
pub fn calculate_position(preset: IndicatorPosition) -> Point {
    let screen_width: f32 = 1920.0;
    let screen_height: f32 = 1080.0;

    let horizontal = match preset {
        IndicatorPosition::TopLeft | IndicatorPosition::BottomLeft => EDGE_OFFSET,
        IndicatorPosition::TopCenter | IndicatorPosition::BottomCenter => {
            (screen_width - PILL_WIDTH) / 2.0
        }
        IndicatorPosition::TopRight | IndicatorPosition::BottomRight => {
            screen_width - PILL_WIDTH - EDGE_OFFSET
        }
    };

    let vertical = match preset {
        IndicatorPosition::TopLeft | IndicatorPosition::TopCenter | IndicatorPosition::TopRight => {
            EDGE_OFFSET
        }
        IndicatorPosition::BottomLeft
        | IndicatorPosition::BottomCenter
        | IndicatorPosition::BottomRight => screen_height - PILL_HEIGHT - EDGE_OFFSET,
    };

    Point::new(horizontal, vertical)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bottom_center_is_horizontally_centered() {
        let position = calculate_position(IndicatorPosition::BottomCenter);
        let expected_horizontal = (1920.0 - PILL_WIDTH) / 2.0;
        assert!((position.x - expected_horizontal).abs() < f32::EPSILON);
    }

    #[test]
    fn top_left_uses_edge_offset() {
        let position = calculate_position(IndicatorPosition::TopLeft);
        assert!((position.x - EDGE_OFFSET).abs() < f32::EPSILON);
        assert!((position.y - EDGE_OFFSET).abs() < f32::EPSILON);
    }

    #[test]
    fn bottom_right_offsets_from_far_edges() {
        let position = calculate_position(IndicatorPosition::BottomRight);
        let expected_horizontal = 1920.0 - PILL_WIDTH - EDGE_OFFSET;
        let expected_vertical = 1080.0 - PILL_HEIGHT - EDGE_OFFSET;
        assert!((position.x - expected_horizontal).abs() < f32::EPSILON);
        assert!((position.y - expected_vertical).abs() < f32::EPSILON);
    }

    #[test]
    fn all_presets_produce_non_negative_coordinates() {
        let presets = [
            IndicatorPosition::TopLeft,
            IndicatorPosition::TopCenter,
            IndicatorPosition::TopRight,
            IndicatorPosition::BottomLeft,
            IndicatorPosition::BottomCenter,
            IndicatorPosition::BottomRight,
        ];
        for preset in presets {
            let position = calculate_position(preset);
            assert!(position.x >= 0.0, "Negative x for {preset:?}");
            assert!(position.y >= 0.0, "Negative y for {preset:?}");
        }
    }
}
