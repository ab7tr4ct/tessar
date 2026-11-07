//! `HttpEmbedder` against a tiny local server that speaks just enough
//! HTTP/1.1 to answer `/v1/embeddings`.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};
use tessar_core::embed::Embedder;
use tessar_embed::{HttpEmbedder, Prompts};

/// What the server saw: request bodies and authorization headers.
#[derive(Default)]
struct Seen {
    bodies: Vec<Value>,
    auth: Vec<Option<String>>,
}

/// Serves requests until the process ends. `answer` turns a request body
/// into (status, response body).
fn serve(answer: fn(&Value) -> (u16, String)) -> (String, Arc<Mutex<Seen>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Seen::default()));
    let log = seen.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let (mut length, mut auth) = (0usize, None);
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                let line = line.trim_end();
                if line.is_empty() {
                    break;
                }
                let lower = line.to_ascii_lowercase();
                if let Some(v) = lower.strip_prefix("content-length:") {
                    length = v.trim().parse().unwrap();
                }
                if lower.starts_with("authorization:") {
                    auth = Some(line["authorization:".len()..].trim().to_owned());
                }
            }
            let mut body = vec![0u8; length];
            reader.read_exact(&mut body).unwrap();
            let body: Value = serde_json::from_slice(&body).unwrap();
            let (status, reply) = answer(&body);
            {
                let mut seen = log.lock().unwrap();
                seen.bodies.push(body);
                seen.auth.push(auth);
            }
            write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",
                reply.len()
            )
            .unwrap();
        }
    });
    (url, seen)
}

/// Each vector is [length of the input, position in the batch], and the
/// items come back in reverse order — clients must sort by `index`.
fn by_length(body: &Value) -> (u16, String) {
    let input = body["input"].as_array().unwrap();
    let data: Vec<Value> = input
        .iter()
        .enumerate()
        .rev()
        .map(|(i, t)| json!({ "index": i, "embedding": [t.as_str().unwrap().len() as f32, i as f32] }))
        .collect();
    (200, json!({ "data": data }).to_string())
}

#[test]
fn documents_come_back_in_order_with_prompts_and_batches() {
    let (url, seen) = serve(by_length);
    let embedder = HttpEmbedder::new(&url, "m")
        .with_prompts(Prompts {
            query: "q: ".into(),
            document: "d: ".into(),
        })
        .with_batch(2)
        .with_api_key("secret");
    let texts: Vec<String> = ["a", "bbb", "cc"].map(String::from).to_vec();

    let vectors = embedder.embed_documents(&texts).unwrap();
    assert_eq!(vectors, [vec![4.0, 0.0], vec![6.0, 1.0], vec![5.0, 0.0]]);

    let query = embedder.embed_query("xyz").unwrap();
    assert_eq!(query, [6.0, 0.0]);

    let seen = seen.lock().unwrap();
    let inputs: Vec<&Value> = seen.bodies.iter().map(|b| &b["input"]).collect();
    assert_eq!(
        inputs,
        [
            &json!(["d: a", "d: bbb"]),
            &json!(["d: cc"]),
            &json!(["q: xyz"])
        ]
    );
    assert!(seen.bodies.iter().all(|b| b["model"] == "m"));
    assert!(
        seen.auth
            .iter()
            .all(|a| a.as_deref() == Some("Bearer secret"))
    );
}

#[test]
fn server_errors_carry_status_and_body() {
    let (url, _) = serve(|_| (503, r#"{"error":"loading model"}"#.into()));
    let err = HttpEmbedder::new(&url, "m").embed_query("x").unwrap_err();
    assert!(err.0.contains("503"), "{err}");
    assert!(err.0.contains("loading model"), "{err}");
}

#[test]
fn a_missing_vector_is_an_error() {
    let (url, _) = serve(|_| {
        (
            200,
            json!({ "data": [{ "index": 0, "embedding": [1.0] }] }).to_string(),
        )
    });
    let texts: Vec<String> = ["a", "b"].map(String::from).to_vec();
    let err = HttpEmbedder::new(&url, "m")
        .embed_documents(&texts)
        .unwrap_err();
    assert!(err.0.contains("no vector for input 1"), "{err}");
}

#[test]
fn an_unreachable_server_is_an_error() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    assert!(HttpEmbedder::new(&url, "m").embed_query("x").is_err());
}
