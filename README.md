# SentencePiece Tokenisation in Rust

This repository provides an implementation of Google's SentencePiece tokenisation algorithm in Rust. The code employs dynamic programming along with an Expectation Maximization (EM) approach to iteratively refine token probabilities and determine optimal token segmentation.

## Getting Started

### Prerequisites

- **Rust:** Ensure you have [Rust installed](https://www.rust-lang.org/tools/install).
- **Dependencies:**  
  - Standard library types like `HashMap` and `HashSet`
  - A custom implementation of a **Trie** for token management  
  - The [`regex`](https://crates.io/crates/regex) crate for pattern matching  
  - A function (or crate) providing the `digamma` function for probability computations  
  - A random number generator (e.g., using the [`rand`](https://crates.io/crates/rand) crate)

### Installation

Clone the repository and build the project using Cargo:

```bash
git clone <repository-url>
cd <repository-directory>
cargo build --release
```

### Usage

#### 1. Initialization

Create a new SentencePiece Tokeniser instance:

```rust
let mut spt = SPT::new();
```

#### 2. Preparing Training Data

Prepare your text, token frequency map, and character set. For example:

```rust
use std::collections::{HashMap, HashSet};

let text = "your training text here";
let mut tokens: HashMap<String, usize> = HashMap::new();
// Populate `tokens` with token frequencies, e.g., from an initial vocabulary

// Extract the set of characters from the text
let characters: HashSet<char> = text.chars().collect();
let vocab_size = 1000;    // Desired vocabulary size
let delta = 0.001;        // Convergence threshold for EM
let max_iter = 100;       // Maximum iterations per EM round
let max_round = 10;       // Maximum EM rounds
```

#### 3. Fitting the Model

Fit the model using the `fit` method. This runs multiple rounds of the EM algorithm and prunes tokens to maintain vocabulary size:

```rust
spt.fit(text, &mut tokens, &characters, vocab_size, delta, max_iter, max_round)?;
```

#### 4. Tokenisation

After fitting, you can tokenize new text. The tokenizer replaces spaces with underscores and supports n-best tokenisation:

```rust
let input_text = "your input text here";
let nbest_size = 5;  // Number of best candidates to consider during tokenisation
let tokenization = spt.tokenize(input_text, nbest_size);
println!("Tokenization: {:?}", tokenization);
```
