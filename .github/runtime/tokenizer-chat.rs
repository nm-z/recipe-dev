//! Public API regression checks using real GGUF vocabulary metadata, without loading weights.
//! Run with `recipe run .github/runtime/tokenizer-chat.rs -- mistral.gguf gemma4.gguf llama-spm.gguf`.
//! Mistral source: bartowski/Mistral-7B-Instruct-v0.3-GGUF, revision 61fd4167fff3ab01ee1cfe0da183fa27a944db48, Q4_K_M.
//! Vocabulary sources: ggml-org/llama.cpp, revision 23b0202a189c44a54625aadcb37a946dd1d6278d, models/ggml-vocab-{gemma-4,llama-spm}.gguf.
use recipe::{Gguf, GgufValue};
use std::path::Path;

fn main() {
	let paths: Vec<_> = std::env::args().skip(1).collect();
	assert_eq!(paths.len(), 3, "pass Mistral-7B-Instruct-v0.3, Gemma 4, and template-free Llama SentencePiece GGUF paths");
	let mistral = Gguf::open(Path::new(&paths[0])).unwrap();
	assert_eq!(mistral.value("tokenizer.ggml.model").and_then(GgufValue::text), Some("llama"));
	let coder = mistral.tokenizer();
	for generation in [false, true] {
		assert_eq!(coder.chat(&[("user", "Hello")], generation), "<s>[INST] Hello [/INST]");
		assert_eq!(coder.chat(&[("user", " Hello "), ("assistant", "Hi"), ("user", "Again")], generation), "<s>[INST]  Hello  [/INST]Hi</s>[INST] Again [/INST]");
		println!("PASS mistral template generation={generation}: user and conversation, BOS/EOS, preserved whitespace");
	}
	assert!(coder.prompt(&[("assistant", "Hi")], true).is_err(), "Mistral's template must reject a conversation starting with an assistant");
	println!("PASS mistral template role validation");
	let gemma = Gguf::open(Path::new(&paths[1])).unwrap();
	assert_eq!(gemma.value("tokenizer.ggml.model").and_then(GgufValue::text), Some("gemma4"));
	let coder = gemma.tokenizer();
	let turns = "<|turn>system\nBe concise.<turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\nHi<turn|>\n";
	for generation in [false, true] {
		let expected = format!("{turns}{}", if generation { "<|turn>model\n<|channel>thought\n<channel|>" } else { "" });
		assert_eq!(coder.chat(&[("developer", " Be concise. "), ("user", " Hello "), ("assistant", " Hi ")], generation), expected);
		println!("PASS gemma4 turns generation={generation}: role mapping, trimming, generation suffix");
	}
	assert_eq!(coder.chat(&[], false), "");
	assert_eq!(coder.chat(&[], true), "<|turn>model\n<|channel>thought\n<channel|>");
	println!("PASS gemma4 empty conversation");
	let llama = Gguf::open(Path::new(&paths[2])).unwrap();
	assert_eq!(llama.value("tokenizer.ggml.model").and_then(GgufValue::text), Some("llama"));
	assert!(llama.value("tokenizer.chat_template").is_none(), "use a template-free Llama vocabulary");
	let error = llama.tokenizer().prompt(&[("user", "Hello")], true).unwrap_err();
	assert_eq!(error.to_string(), "tokenizer.chat_template is absent");
	println!("PASS llama without template reports missing metadata");
	println!("TOKENIZER CHAT PASS");
}
