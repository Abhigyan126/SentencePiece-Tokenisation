use std::collections::HashMap;

pub struct TrieNode {
    children: HashMap<char, TrieNode>,
    value: Option<f64>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            value: None,
        }
    }
}

pub struct Trie {
    root: TrieNode,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            root: TrieNode::new(),
        }
    }

    pub fn add(&mut self, word: &str, value: f64) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        node.value = Some(value);
    }

    pub fn get_value(&self, word: &str) -> Option<f64> {
        let mut node = &self.root;
        for ch in word.chars() {
            match node.children.get(&ch) {
                Some(next_node) => node = next_node,
                None => return None, // Return None when the word is not found
            }
        }
        node.value // Return the Option<f64> directly
    }
    

    pub fn set_value(&mut self, word: &str, value: f64) -> Result<(), &'static str> {
        let mut node = &mut self.root;
        for ch in word.chars() {
            match node.children.get_mut(&ch) {
                Some(next_node) => node = next_node,
                None => return Err("Word not found in trie"),
            }
        }
        if node.value.is_some() {
            node.value = Some(value);
            Ok(())
        } else {
            Err("Word not found in trie")
        }
    }
}