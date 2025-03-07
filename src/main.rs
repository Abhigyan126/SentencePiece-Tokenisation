/* 
mod byte_pair_encoding;
use byte_pair_encoding::BPE;

fn main() {
    let example_text = "low lower newest wide widowed";
    let mut bpe = BPE::new();

    bpe.fit(example_text, 10);

    let tokenized = bpe.tokenize("lower");
    println!("Tokenized: {:?}", tokenized);

    let detokenized = bpe.detokenize(tokenized);
    println!("Detokenized: {:?}", detokenized);
}

*/
/*
mod trie;
use trie::Trie;


fn main() {
    let mut trie = Trie::new();

    trie.add("hello", 5);
    trie.add("hell", 3);
    trie.add("world", 10);

    println!("Value of 'hello': {}", trie.get_value("hello")); // 5
    println!("Value of 'hell': {}", trie.get_value("hell")); // 3
    println!("Value of 'world': {}", trie.get_value("world")); // 10
    println!("Value of 'hi': {}", trie.get_value("hi")); // 0 (not found)

    match trie.set_value("hello", 8) {
        Ok(_) => println!("Updated 'hello' to 8"),
        Err(err) => println!("Error: {}", err),
    }

    println!("Updated value of 'hello': {}", trie.get_value("hello")); // 8

    match trie.set_value("hi", 7) {
        Ok(_) => println!("Updated 'hi' to 7"),
        Err(err) => println!("Error: {}", err), // Error: Word not found in trie
    }
}
*/

mod trie;
use std::collections::{HashMap, HashSet};
use std::f64::NEG_INFINITY;
use regex::Regex;
use trie::Trie;
use std::cmp::max;
use std::f64::consts::PI;
use rand::{rng, Rng};
use std::fs;
mod byte_pair_encoding;
use byte_pair_encoding::BPE;




fn digamma(mut x: f64, terms: usize) -> f64 {
    if x <= 0.0 {
        return digamma(1.0 - x, terms) - PI / (PI * x).tan();
    }
    let mut result = 0.0;
    // Recurrence relation to shift x to >= 10
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    // Asymptotic expansion
    result += x.ln() - 1.0 / (2.0 * x);
    let coeffs = [
        1.0 / 6.0, -1.0 / 30.0, 1.0 / 42.0, -1.0 / 30.0, 5.0 / 66.0, 
        -691.0 / 2730.0, 7.0 / 6.0, -3617.0 / 510.0, 43867.0 / 798.0, -174611.0 / 330.0
    ];
        let x2 = x * x;
    let mut power = x2;
    for (i, c) in coeffs.iter().enumerate().take(terms) {
        result -= c / (power * (2.0 * (i as f64 + 1.0)));
        power *= x2;
    }
    result
}

struct SPT {
    trie: Trie,
    max_len: usize,
    vocab_size: usize,
}
impl SPT {
    fn new() -> Self {
        SPT {
            trie: Trie::new(),
            max_len: 0,
            vocab_size: 0,
        }
    }

    fn init_trie(&mut self, tokens: &HashMap<String, usize>) {
        let norm: usize = tokens.values().sum();
        let logsum = digamma(norm as f64, 10);
        self.max_len = 0;
        
        for (tok, val) in tokens {
            self.trie.add(tok, digamma(*val as f64, 10)-logsum);
            self.max_len = max(self.max_len, tok.len());
        }
    }

    fn forward_step(&mut self, text: &str) -> Result<(f64, Vec<Option<usize>>), String> {
        let n = text.chars().count();  // Count characters, not bytes
        let mut d = vec![f64::NEG_INFINITY; n + 1];
        let mut p = vec![None; n + 1];
        d[0] = 0.0;
    
        let mut char_indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        char_indices.push(text.len());  // Ensure `end` is always valid
    
        for i in 1..=n {
            for j in (i.saturating_sub(self.max_len))..i {
                if let (Some(&start), Some(&end)) = (char_indices.get(j), char_indices.get(i)) {
                    let final_token = &text[start..end];
    
                    if let Some(final_value) = self.trie.get_value(final_token) {
                        if d[j] + final_value > d[i] {
                            d[i] = d[j] + final_value;
                            p[i] = Some(i - j);
                        }
                    }
                }
            }
            if p[i].is_none() {
                return Err(format!(
                    "Encountered unknown token '{}'.",
                    text.chars().nth(i - 1).unwrap_or('?')
                ));
            }
        }
    
        Ok((d[n], p))
    }
    
    
    fn backward_step(&self, text: &str, p: &[Option<usize>]) -> Vec<String> {
        let mut idx = p.len().saturating_sub(1);
        let mut tokenization = Vec::new();
    
        let char_indices: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    
        while idx > 1 {
            if let Some(step) = p[idx - 1] {
                let next_idx = idx.saturating_sub(step);
                if next_idx == 0 {
                    break;
                }
    
                // Get the character indices for slicing
                let start = char_indices[next_idx - 1];
                let end = char_indices[idx - 1];
    
                let tok = text[start..end].to_string();
                tokenization.push(tok);
    
                idx = next_idx;
            } else {
                break; // Stop if we encounter None
            }
        }
    
        tokenization.reverse();
        tokenization
    }
    
    
    fn e_step(&mut self, tokenisation: Vec<String>) {
        //counter function
        fn count_tokens(tokenization: Vec<String>) -> HashMap<String, f64> {
            let mut counts = HashMap::<String, f64>::new();
            for token in tokenization {
                *counts.entry(token).or_insert(0.0) += 1.0;
            }
            counts
        }
        // function ends
        let mut counts = count_tokens(tokenisation);
        let norm:f64 = counts.values().sum();
        let logsum = digamma(norm, 10);
        for (_, v) in counts.iter_mut() {
            *v = digamma(*v, 10) - logsum;
        }
        for (k, v) in counts {
            if let Err(err) = self.trie.set_value(&k, v) {
                eprintln!("Error setting value for {}: {}", k, err);
            }
        }
    }

    fn m_step(&mut self, text: &str) -> (Vec<String>, f64) {
        let (loss, p) = self.forward_step(text).unwrap();
        let tokenisation = self.backward_step(text, &p);
        (tokenisation, loss)
    }

    fn em_step(&mut self,text: &str, tokenisation: Vec<String>) -> (f64, Vec<String>) {
        self.e_step(tokenisation);
        let (tokenisation, loss) = self.m_step(text);
        (loss, tokenisation)
    }

    fn em_round(&mut self, text: &str, delta: f64, max_iter: i32) {
        print!("EM round");
        let (tokenisation, old_loss) = self.m_step(text);
        let mut old_loss = old_loss;
        let mut current_tokenisation = tokenisation;
        for step in 0..max_iter {
            println!("EM iter {step}");
            let (loss, new_tokenisation) =  self.em_step(text, current_tokenisation.clone());
            println!("Loss= {loss}");
            if (old_loss-loss).abs() < delta {
                break;
            }
            old_loss = loss;
            current_tokenisation = new_tokenisation;
        }
    }

    fn prune_tokens(&mut self,tokens: &mut HashMap<String, usize>,characters: &HashSet<char>,vocab_size: usize,trim_frac: f64,) -> Result<bool, String> {
        let mut sorted_tokens: Vec<(String, usize)> = tokens.iter().map(|(k, v)| (k.clone(), *v)).collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1));
    
        let mut n = sorted_tokens.len();
        let mut n_trim = (trim_frac * n as f64) as usize;
    
        let mut i = n - 1;
        while i > 0 {
            if n <= vocab_size {
                return Ok(false); // Vocab size reached, no need for another round
            }
            if n_trim == 0 {
                return Ok(true); // More rounds needed
            }
    
            let tok = &sorted_tokens[i].0;
            if !characters.contains(&tok.chars().next().unwrap()) {
                if let Err(err) = self.trie.set_value(tok, 0.0) {
                    eprintln!("Error setting value for {}: {}", tok, err);
                }
                tokens.remove(tok); // Remove from tokens dynamically
                n_trim -= 1;
                n -= 1;
            }
    
            if i == 0 {
                break;
            }
            i -= 1;
        }
    
        if n_trim > 0 {
            return Err("Could not reduce tokens further. Please increase vocab size".to_string());
        }
        Ok(false)
    }
    

    fn fit(&mut self, text: &str, tokens: &mut HashMap<String, usize>, characters: &HashSet<char>, vocab_size: usize, delta: f64, max_iter: i32, max_round: i32) -> Result<(), String>{
        let text = text.replace(" ", "_");
        if vocab_size > tokens.len() {
            eprintln!("Vocab size is larger than the available number of tokens {}.",tokens.len());
        }
        self.init_trie(tokens);
        for i in 1..max_round {
            println!("-> Round {}, vocab size: {} <--", i, tokens.len());
            self.em_round(&text, delta, max_iter);
            match self.prune_tokens(tokens, characters, vocab_size, 0.2) {
                Ok(false) => break,
                Ok(true) => {}
                Err(err) => return Err(err),
            }
        }
        self.vocab_size = tokens.len();
        Ok(())
    }

    fn generalized_forward_step(&mut self, text: &str, nbest_size: usize) -> Vec<Option<Vec<usize>>> {
        let n = text.len();
        let mut d = vec![NEG_INFINITY; n + 1];
        let mut p: Vec<Option<Vec<usize>>> = vec![None; n + 1];
        d[0] = 0.0;
    
        for i in 1..=n {
            let mut d_queue = Vec::new();
            let mut p_queue = Vec::new();
    
            for j in (i.saturating_sub(self.max_len))..i {
                let final_token = &text[j..i];
                let final_value = self.trie.get_value(final_token);
                let curr_d = d[j] + final_value.unwrap_or(0.0);
                let curr_p = final_token.len();
                d[i] = d[i].max(curr_d);
                d_queue.push(curr_d);
                p_queue.push(curr_p);
            }
            
            let mut indices: Vec<usize> = (0..d_queue.len()).collect();
            indices.sort_by(|&a, &b| d_queue[a].partial_cmp(&d_queue[b]).unwrap());
            
            let ids = &indices[d_queue.len().saturating_sub(nbest_size)..];
            p[i] = Some(ids.iter().map(|&z| p_queue[z]).collect());
        }
    
        p
    }

    fn generalized_backward_step(&mut self, text: &str, p: &[Option<Vec<usize>>]) -> Vec<String> {
        let mut idx = p.len();
        let mut tokenization = Vec::new();
        let mut rng = rng();

    
        while idx > 1 {
            if let Some(back_steps_list) = &p[idx - 1] {
                if !back_steps_list.is_empty() {
                    let rand_index = rng.random_range(0..back_steps_list.len());
                    let back_steps = back_steps_list[rand_index];
                    
                    let next_idx = idx.saturating_sub(back_steps);
                    let tok = &text[next_idx.saturating_sub(1)..idx.saturating_sub(1)];
                    tokenization.push(tok.to_string());
                    idx = next_idx;
                } else {
                    break; 
                }
            } else {
                break;
            }
        }
    
        tokenization.reverse();
        tokenization
    }
    
    fn tokenize(&mut self, text: &str, nbest_size: usize) -> Vec<String> {
        let re = Regex::new(" ").unwrap();
        let text = re.replace_all(text, "_").to_string();
        let p = self.generalized_forward_step(&text, nbest_size);
        let tokenisation = self.generalized_backward_step(&text, &p);
        tokenisation
    }

}


//test

fn read_and_clean_file(filename: &str) -> std::io::Result<String> {
    let text = fs::read_to_string(filename)?;
    
    // Replace newlines with spaces
    let text = text.replace('\n', " ");

    // Remove extra spaces using regex
    let re = Regex::new(r"\s+").unwrap();
    let cleaned_text = re.replace_all(&text, " ").to_string();

    Ok(cleaned_text)
}
fn main() {
    match read_and_clean_file("sample_train.txt") {
        Ok(contents) => {
            println!("Text length: {}", contents.len());

            let mut spt = SPT::new();
            let mut bpe = BPE::new();
            let num_merges = 100;

            bpe.fit(&contents, num_merges);
            let mut tokens = bpe.tokens;
            let mut characters = bpe.characters;

            // Adjusting tokens as in Python
            if let Some(space_val) = tokens.remove(" ") {
                tokens.insert("_".to_string(), space_val);
            }

            // Adjusting characters
            characters.remove(&' ');
            characters.insert('_');



            let _ = spt.fit(&contents, &mut tokens, &characters, 100, 0.01, 5, 5);

            println!("Finished tokenization.");
            let nbest_size = 3;
            let string = "so unreal in real";
            for i in 0..3 {
                let x = spt.tokenize(string, nbest_size);
                println!("Sample {}: {:?}", i + 1, x);
            }
            
        }
        Err(err) => eprintln!("Error reading file: {}", err),
    }
}