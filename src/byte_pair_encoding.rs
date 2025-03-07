//https://github.com/Jmkernes/SentencePiece-from-scratch/blob/main/byte_pair_encoder.py

use regex::Regex;
use std::collections::{HashMap, HashSet};

pub struct BPE {
    pub vocab: HashMap<String, usize>,
    pub tokens: HashMap<String, usize>,
    pub merges: Vec<(String, String)>,
    pub characters: HashSet<char>,
}

impl BPE {
    pub fn new() -> Self {
        BPE {
            vocab: HashMap::new(),
            tokens: HashMap::new(),
            merges: Vec::new(),
            characters: HashSet::new(),
        }
    }

    fn format_word(text: &str, space_token: char) -> String {
        text.chars()
            .map(|c| c.to_string())
            .collect::<Vec<String>>()
            .join(" ") + " " + &space_token.to_string()
    }

    fn initialize_vocab(text: &str) -> (HashMap<String, usize>, HashMap<String, usize>) {
        let re = Regex::new(r"\s+").unwrap();
        let text = re.replace_all(text, " ").to_string();
        let all_words: Vec<&str> = text.split(" ").collect();
        let mut vocab = HashMap::<String, usize>::new();

        for word in all_words {
            let p_word = Self::format_word(word, '_');
            let count = vocab.entry(p_word.clone()).or_insert(0);
            *count += 1;
        }

        let mut tokens = HashMap::<String, usize>::new();
        for c in text.chars() {
            let count = tokens.entry(c.to_string()).or_insert(0);
            *count += 1;
        }

        (vocab, tokens)
    }

    fn get_bigram_counts(vocab: &HashMap<String, usize>) -> HashMap<(String, String), usize> {
        let mut pairs = HashMap::<(String, String), usize>::new();
        for (word, count) in vocab.iter() {
            let symbols: Vec<&str> = word.split(" ").collect();
            for i in 0..(symbols.len() - 1) {
                let pair = (symbols[i].to_string(), symbols[i + 1].to_string());
                *pairs.entry(pair).or_insert(0) += count;
            }
        }
        pairs
    }

    fn merge_vocab(pair: &(String, String), vocab_in: &HashMap<String, usize>) -> (HashMap<String, usize>, (String, String)) {
        let mut vocab_out = HashMap::new();
        let bigram = format!("{} {}", pair.0, pair.1);  // Example: "w e"
        let bytepair = format!("{}{}", pair.0, pair.1); // Example: "we"
    
        for (word, &count) in vocab_in.iter() {
            let tokens: Vec<&str> = word.split_whitespace().collect(); // Split word into tokens
            let mut new_tokens: Vec<String> = Vec::new();
    
            let mut i = 0;
            while i < tokens.len() {
                if i < tokens.len() - 1 && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                    new_tokens.push(bytepair.clone()); // Merge "w e" → "we"
                    i += 2; // Skip the merged token
                } else {
                    new_tokens.push(tokens[i].to_string()); // Keep token unchanged
                    i += 1;
                }
            }
    
            let new_word = new_tokens.join(" "); // Convert Vec<String> to a single string
            vocab_out.insert(new_word, count);
        }
    
        (vocab_out, (bigram, bytepair))
    }

    fn find_merges(vocab: &mut HashMap<String, usize>, tokens: &mut HashMap<String, usize>, num_merges: usize) -> Vec<(String, String)> {
        let mut merges = Vec::new();

        for _ in 0..num_merges {
            let pairs = Self::get_bigram_counts(&vocab);
            if pairs.is_empty() {
                break;
            }
            let best_pair = pairs.iter().max_by_key(|entry| entry.1).map(|(pair, _)| pair.clone()).unwrap();
            let best_count = pairs[&best_pair];
            let (new_vocab, (bigram, bytepair)) = Self::merge_vocab(&best_pair, &vocab);
            *tokens.entry(bytepair.clone()).or_insert(0) += best_count;

            *vocab = new_vocab;
            merges.push((bigram, bytepair));
        }

        merges
    }

    pub fn fit(&mut self, text: &str, num_merges: usize) {
        let (mut vocab, mut tokens) = Self::initialize_vocab(text);
        self.characters = tokens.keys()
        .flat_map(|s| s.chars()) // Extract characters from each string
        .collect();
            self.merges = Self::find_merges(&mut vocab, &mut tokens, num_merges);
        self.vocab = vocab;
        self.tokens = tokens;
    }
    
    #[allow(dead_code)]
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut word = Self::format_word(text, '_');
        for (bigram, bytepair) in &self.merges {
            word = word.replace(bigram, bytepair);
        }
        word.split_whitespace().map(|s| s.to_string()).collect()
    }

    #[allow(dead_code)]
    pub fn detokenize(&self, tokens: Vec<String>) -> String {
        let mut text = tokens.join("");
        text = text.replace("_", " ");
        text
    }

}