//! The clock port: where the store gets the current time.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::model::Timestamp;

/// A source of the current time.
pub trait Clock: Send + Sync {
    /// The current time.
    fn now(&self) -> Timestamp;
}

/// The system clock.
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        let ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(i64::MAX));
        Timestamp::from_millis(ms)
    }
}

/// A clock that moves only when told to. For tests.
#[derive(Debug)]
pub struct ManualClock(AtomicI64);

impl ManualClock {
    /// A clock showing `start`.
    pub fn new(start: Timestamp) -> Self {
        Self(AtomicI64::new(start.as_millis()))
    }

    /// Sets the time.
    pub fn set(&self, t: Timestamp) {
        self.0.store(t.as_millis(), Ordering::SeqCst);
    }

    /// Moves the time forward by `ms` milliseconds.
    pub fn advance(&self, ms: i64) {
        self.0.fetch_add(ms, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Timestamp {
        Timestamp::from_millis(self.0.load(Ordering::SeqCst))
    }
}
