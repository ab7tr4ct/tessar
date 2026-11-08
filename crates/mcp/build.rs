//! Stamps the binary with when and from which commit it was built, so a
//! running daemon can tell it is older than the binary that talks to it.

use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn main() {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=TESSAR_BUILD_ID={secs}");
    println!("cargo:rustc-env=TESSAR_GIT_HASH={hash}");
    // Without a rerun-if-changed line cargo reruns this script only when
    // this crate's own files change, so a rebuild after editing the core
    // would keep the old stamp. Watch the sibling crates and the commit.
    println!("cargo:rerun-if-changed=..");
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
