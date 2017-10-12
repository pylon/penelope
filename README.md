# Penelope

Natural Language Processing (NLP) and Machine Learning (ML) library for Elixir.

## Installation

### Dependencies
This package requires an implementation of `BLAS` for efficient matrix math
and `libsvm` . These can be installed as follows:

#### OSX
BLAS is built into OSX, so you only need to install libsvm.

```bash
brew install libsvm
```

#### Ubuntu
Install `libblas-dev` and `libsvm` via apt.

```bash
sudo apt install libblas-dev libsvm-dev
```

### Hex
```elixir
def deps do
  [
    {:penelope, "~> 0.1.0"}
  ]
end
```
