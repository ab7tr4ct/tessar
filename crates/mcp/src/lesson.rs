//! Lessons: ordinary statement facts of a fixed shape that record an
//! agent's own mistakes. No fact kind of their own; the shape is the
//! contract between the `lesson` tool and whatever reads lessons back.

/// The key of the entity every lesson mentions.
pub const ENTITY_KEY: &str = "lessons";

/// The kind of that entity.
pub const ENTITY_KIND: &str = "topic";

/// What a lesson's text starts with, up to the date.
pub const PREFIX: &str = "Lesson (";

/// The text of a lesson written on `date` (`YYYY-MM-DD`): `Lesson (date):
/// text`. A `text` that already is a lesson ([`is_lesson`]) is kept as it
/// is. A dated header with nothing after it gives an empty text, which the
/// caller refuses like any empty one.
pub fn text(date: &str, text: &str) -> String {
    let text = text.trim();
    if is_lesson(text) {
        text.to_owned()
    } else if text.starts_with(PREFIX)
        && header_len(text).is_some_and(|n| text[n..].trim().is_empty())
    {
        String::new()
    } else {
        format!("{PREFIX}{date}): {text}")
    }
}

/// Whether a fact's text is a lesson: `Lesson (YYYY-MM-DD): ` and then some
/// text.
pub fn is_lesson(text: &str) -> bool {
    header_len(text).is_some_and(|n| {
        let rest = &text[n..];
        rest.starts_with(' ') && !rest.trim().is_empty()
    })
}

/// The length of a leading `Lesson (YYYY-MM-DD):`, if there is one.
fn header_len(text: &str) -> Option<usize> {
    let rest = text.strip_prefix(PREFIX)?;
    let date = rest.get(..10)?;
    let shaped = date.bytes().enumerate().all(|(i, b)| match i {
        4 | 7 => b == b'-',
        _ => b.is_ascii_digit(),
    });
    // A real calendar day: `2026-13-45` has the shape but is not a date.
    let real = shaped && chrono::NaiveDate::parse_from_str(date, "%Y-%m-%d").is_ok();
    (real && rest[10..].starts_with("):")).then_some(PREFIX.len() + 12)
}

/// The text of a lesson without its `Lesson (date):` header; `text` as it is
/// when it is not a lesson.
pub fn body(text: &str) -> &str {
    match header_len(text) {
        Some(n) if is_lesson(text) => text[n..].trim_start(),
        _ => text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: &str = "2026-11-12";

    #[test]
    fn prefix_is_added_once() {
        assert_eq!(
            text(D, "  Before X, read Y.  "),
            "Lesson (2026-11-12): Before X, read Y."
        );
        assert_eq!(
            text(D, "Lesson (2026-11-01): kept"),
            "Lesson (2026-11-01): kept"
        );
    }

    #[test]
    fn only_the_exact_shape_is_a_lesson() {
        assert!(is_lesson("Lesson (2026-11-12): x"));
        assert!(!is_lesson("A lesson (2026-11-12): x"));
        assert!(!is_lesson("Lesson ("));
        assert!(!is_lesson("Lesson (from review): x"));
        assert!(!is_lesson("Lesson (2026-01-03):"));
        assert!(!is_lesson("Lesson (2026-01-03):   "));
        assert!(!is_lesson("Lesson (2026-01-03):x"));
        assert!(!is_lesson("Lesson (2026-1-03): x"));
        assert!(!is_lesson("Lesson (2026-13-45): x"));
        assert!(!is_lesson("Lesson (2026-02-30): x"));
        assert!(is_lesson("Lesson (2024-02-29): x"));
        assert!(!is_lesson("Lesson (2026-01-03 ): x"));
        assert!(!is_lesson("Lesson (2026-01-0\u{e9}): x"));
    }

    #[test]
    fn the_body_drops_the_header() {
        assert_eq!(body("Lesson (2026-11-09): read the log"), "read the log");
        assert_eq!(body("Lesson (2026-13-45): x"), "Lesson (2026-13-45): x");
        assert_eq!(body("plain"), "plain");
    }

    #[test]
    fn anything_else_gets_the_dated_prefix() {
        assert_eq!(text(D, "Lesson ("), "Lesson (2026-11-12): Lesson (");
        assert_eq!(
            text(D, "Lesson (from review): x"),
            "Lesson (2026-11-12): Lesson (from review): x"
        );
        assert_eq!(
            text(D, "Lesson (2026-01-03):x"),
            "Lesson (2026-11-12): Lesson (2026-01-03):x"
        );
    }

    #[test]
    fn a_header_without_text_is_empty() {
        assert_eq!(text(D, "Lesson (2026-01-03):"), "");
        assert_eq!(text(D, " Lesson (2026-01-03):  "), "");
    }
}
