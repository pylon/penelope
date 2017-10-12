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

## License

Copyright 2017 Pylon, Inc.

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
