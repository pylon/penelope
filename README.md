# Penelope

Natural Language Processing (NLP) and Machine Learning (ML) library for Elixir.

## Status
[![Hex](http://img.shields.io/hexpm/v/penelope.svg?style=flat)](https://hex.pm/packages/penelope)
[![Test](http://circleci-badges-max.herokuapp.com/img/pylon/penelope?token=:circle-ci-token)](https://circleci.com/gh/pylon/penelope)
[![Coverage](https://coveralls.io/repos/github/pylon/penelope/badge.svg)](https://coveralls.io/github/pylon/penelope)

The API reference is available [here](https://hexdocs.pm/penelope/).

## Installation

### Dependencies
This package requires an implementation of `BLAS` for efficient matrix math.
It can be installed on each platform as follows:

#### OSX
BLAS is built into OSX.

#### Alpine
Install `openblas-dev` via apk.

```bash
sudo apk add openblas-dev
```

#### Ubuntu
Install `libblas-dev` via apt.

```bash
sudo apt install libblas-dev
```

### Hex
```elixir
def deps do
  [
    {:penelope, "~> 0.4"}
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
