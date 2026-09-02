//! The byte-level BPE tokenizer through the public `recipe.gguf(..).tokenizer()`
//! path. Each fixture is a metadata-only GGUF with a 321-token vocabulary
//! trained on `corpus.txt`; the `.ids` next to it is what
//! `llama-tokenize -m <fixture> -f corpus.txt --ids` printed for the same
//! text, so the three pre-tokenizer families are compared with llama.cpp.

use recipe::*;

const CORPUS: &str = "data/gguf/tokenizer/corpus.txt";

fn oracle(family: &str) -> Vec<u32> {
	let text = std::fs::read_to_string(format!("data/gguf/tokenizer/{family}.ids")).unwrap_or_else(|error| panic!("cannot read the {family} oracle: {error}"));
	text.trim().trim_start_matches('[').trim_end_matches(']').split(',').map(|id| id.trim().parse().unwrap_or_else(|error| panic!("{family} oracle id {id:?}: {error}"))).collect()
}

#[test]
fn every_family_encodes_like_llama_tokenize() {
	let corpus = std::fs::read_to_string(CORPUS).unwrap();
	for family in ["gpt-2", "llama-bpe", "qwen2"] {
		let tokenizer = recipe.gguf(format!("data/gguf/tokenizer/{family}.gguf")).tokenizer();
		assert_eq!(tokenizer.vocabulary(), 321, "{family} vocabulary size");
		assert_eq!((tokenizer.bos(), tokenizer.eos(), tokenizer.pad()), (Some(318), Some(319), Some(319)), "{family} special ids");
		let ids = tokenizer.encode(&corpus);
		assert_eq!(ids, oracle(family), "{family} ids differ from llama-tokenize");
		// Only the llama-bpe fixture asks for a leading beginning-of-sequence id.
		let body = if family == "llama-bpe" { &ids[1..] } else { &ids[..] };
		assert_eq!(ids[0] == 318, family == "llama-bpe", "{family} add_bos");
		assert_eq!(tokenizer.decode(body), corpus, "{family} round trip");
		assert_eq!(tokenizer.token(318), "<|bos|>");
	}
}

#[test]
fn special_tokens_and_split_utf8_decode() {
	let tokenizer = recipe.gguf("data/gguf/tokenizer/qwen2.gguf").tokenizer();
	// A user-defined special written in the text is one id, longest match first.
	let ids = tokenizer.encode("<|user|>hi<|eos|>");
	assert_eq!(ids[0], 320, "user-defined special");
	assert_eq!(*ids.last().unwrap(), 319, "control special");
	assert_eq!(tokenizer.decode(&ids), "<|user|>hi<|eos|>");
	// Characters the corpus never merged tokenize to their bytes and rejoin on
	// decode; a cut inside a character decodes to the replacement character.
	let ids = tokenizer.encode("ы𝄞");
	assert_eq!(ids.len(), 6, "two bytes and four bytes");
	assert_eq!(tokenizer.decode(&ids), "ы𝄞");
	assert_eq!(tokenizer.decode(&ids[..3]), "ы\u{FFFD}", "a lone lead byte is not text yet");
}

#[test]
fn an_unknown_family_is_refused_by_name() {
	let mut file = std::fs::read("data/gguf/tokenizer/qwen2.gguf").unwrap();
	let at = file.windows(5).position(|window| window == b"qwen2").expect("the fixture names its family");
	file[at..at + 5].copy_from_slice(b"qwen9");
	let path = std::env::temp_dir().join(format!("recipe-tokenizer-{}.gguf", std::process::id()));
	std::fs::write(&path, file).unwrap();
	let result = std::panic::catch_unwind(|| recipe.gguf(&path).tokenizer());
	let _ = std::fs::remove_file(&path);
	let message = result.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
	assert_eq!(message, "pre-tokenizer family \"qwen9\" is not supported");
}
