defmodule Penelope.NLP.Tokenize.BertTokenizer do
  @moduledoc """
  This is a BERT-compatible wordpiece tokenizer/vectorizer implementation. It
  provides the ability to encode a text string into an integer vector
  containing values derived from a wordpiece vocabulary. The encoded results
  can also be converted back to the original text or a substring of it.

  The initial tokenization is performed by splitting on whitespace. These
  tokens are then further split by punctuation and piecing to find the
  longest matching wordpieces in the vocabulary. Indexes into the original
  whitespace tokenization are maintained, so that the vectorization can
  be inverted without losing anything except non-space whitespace.

  https://arxiv.org/abs/1810.04805
  """

  @defaults %{
    lowercase: true,
    split_regex: ~r/[\s]/u,
    strip_regex: ~r/[\p{Mn}\p{C}\x{0000}\x{FFFD}]/u,
    punct_regex: ~r/[\p{P}$+<=>^`|~]/u,
    special_tokens: ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    unknown_token: "[UNK]",
    piece_prefix: "##"
  }

  @doc """
  tokenizes and vectorizes a string

  The following options are supported:

  |key|description|default|
  |-|-|-|
  |`lowercase`|downcase during vectorization?|``#{
    inspect(@defaults.lowercase)
  }``|
  |`split_regex`|regex used to tokenize the text|``#{
    inspect(@defaults.split_regex)
  }``|
  |`strip_regex`|regex used to remove invalid characters|``#{
    inspect(@defaults.strip_regex)
  }``|
  |`punct_regex`|regex used to split pieces on punctuation|``#{
    String.replace(inspect(@defaults.punct_regex), "|", "\\|")
  }``|
  |`special_tokens`|list of special tokens not to piece|``#{
    inspect(@defaults.special_tokens)
  }``|
  |`unknown_token`|the key used to indicate OOV token|``#{
    inspect(@defaults.unknown_token)
  }``|
  |`piece_prefix`|prefix used to indicate subsequent pieces|``#{
    inspect(@defaults.piece_prefix)
  }``|
  """
  @spec encode(
          text :: String.t(),
          vocab :: %{required(String.t()) => integer()},
          options :: keyword()
        ) :: {[String.t()], [integer()], [integer()]}
  def encode(text, vocab, options \\ []) do
    # apply customizations over the default tokenizer configuration
    config =
      options
      |> Enum.into(@defaults)
      |> Map.put(:vocab, vocab)
      |> Map.update!(:special_tokens, &MapSet.new/1)

    # the base tokenization is simply to split on whitespace
    tokens = String.split(text, config.split_regex, trim: true)

    # convert the tokens to vocabulary keys
    # tokens can be split by punctuation and further split into
    # pieces, so we maintain an index into the original token list
    {indexes, keys} =
      tokens
      |> Enum.with_index()
      |> Enum.flat_map(&split(config, &1))
      |> Enum.flat_map(&piece(config, &1))
      |> Enum.unzip()

    # vectorize the keys through the vocabulary
    values = Enum.map(keys, &config.vocab[&1])

    # return the tokens, piece indexes, and value vector
    {tokens, indexes, values}
  end

  @doc """
  detokenizes a (possibly sub-)sequence of an encoded string
  """
  @spec decode({[String.t()], [integer()], [integer()]}) :: String.t()
  def decode({tokens, indexes, _values}) do
    # to convert back to the original text, retrieve the
    # distinct list of tokens for each requested piece,
    # and then join on spaces
    indexes
    |> Enum.dedup()
    |> Enum.map(&Enum.at(tokens, &1))
    |> Enum.join(" ")
  end

  defp split(config, {token, index}) do
    if token in config.special_tokens do
      # don't split reserved tokens ([PAD], etc.) into pieces,
      # so they can be vectorized as-is
      [{token, index}]
    else
      # normalize the token and split it on punctuation
      token
      |> String.normalize(:nfd)
      |> (&if(config.lowercase, do: String.downcase(&1), else: &1)).()
      |> String.replace(config.strip_regex, "")
      |> String.split(config.punct_regex, include_captures: true, trim: true)
      |> Enum.map(&{&1, index})
    end
  end

  defp piece(config, {token, index}) do
    piece(config, _pieces = [], token, index, _prefix = "")
  end

  defp piece(_config, pieces, "", _index, _prefix) do
    Enum.reverse(pieces)
  end

  defp piece(config, pieces, token, index, prefix) do
    # find the longest matching piece of the token in the vocabulary
    {key, offset} = match(config, token, prefix)

    if key === config.unknown_token do
      # if any piece of the token is unknown, mark the whole token as unknown
      [{index, key}]
    else
      # piece the remainder of the token, with the ## prefix
      pieces = [{index, key} | pieces]
      token = String.slice(token, offset, String.length(token) - offset)
      piece(config, pieces, token, index, config.piece_prefix)
    end
  end

  defp match(config, token, prefix) do
    count = String.length(token)
    key = prefix <> token

    cond do
      # if this piece is in the vocabulary, it is the longest piece
      Map.has_key?(config.vocab, key) -> {key, count}
      # if we couldn't even match a single character, return unknown
      count === 1 -> {config.unknown_token, count}
      # otherwise, try a shorter piece
      true -> match(config, String.slice(token, 0, count - 1), prefix)
    end
  end
end
