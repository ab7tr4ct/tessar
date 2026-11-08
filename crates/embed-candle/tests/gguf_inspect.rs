//! Prints the metadata and tensor table of the Q8_0 GGUF. Ignored.

use candle_core::quantized::gguf_file::Content;

#[test]
#[ignore = "inspection"]
fn inspect() {
    let p = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".tessar/models/voyage-4-nano-Q8_0.gguf");
    let mut f = std::fs::File::open(p).unwrap();
    let c = Content::read(&mut f).unwrap();
    for (k, v) in &c.metadata {
        let s = format!("{v:?}");
        println!("META {k} = {}", s.chars().take(100).collect::<String>());
    }
    let mut names: Vec<_> = c.tensor_infos.iter().collect();
    names.sort_by_key(|(n, _)| n.to_string());
    for (n, i) in names {
        if n.starts_with("blk.") && !n.starts_with("blk.0.") {
            continue;
        }
        println!("T {n} {:?} {:?}", i.ggml_dtype, i.shape);
    }
}
