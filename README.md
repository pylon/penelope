# Penelope

Natural Language Processing (NLP) and Machine Learning (ML) library for Elixir.

## Installation

### BLAS
This package requires an implementation of BLAS for efficient matrix math. It
can be installed as follows:

#### OSX
[openblas](http://www.openblas.net/) is the easiest way to install blas. It
can be installed via homebrew.

```
brew install openblas
```

Then, you will need to add the `CFLAGS` and `LDFLAGS` variables for openblas
to your environment.

#### Ubuntu
Install `libblas-dev` via apt.

```
sudo apt install libblas-dev
```

### Hex
```elixir
def deps do
  [
    {:penelope, "~> 0.1.0"}
  ]
end
```
