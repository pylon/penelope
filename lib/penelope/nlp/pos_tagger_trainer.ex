defmodule Penelope.NLP.POSTaggerTrainer do
  @moduledoc """
  Convenience functions for training and testing a part of speech tagger.

  Training the tagger itself is relatively straightforward; these functions
  are essentially light wrappers around file processing that can be bypassed if
  your training files are in a different format. The only necessity is that
  training files must be completely read into memory; the underlying CRF tagger
  cannot be trained online.
  """

  alias Penelope.NLP.POSTagger

  require Logger

  @doc """
  Trains a part of speech tagger model using the supplied data. The training
  file is expected to include a tokenized sentence and its corresponding
  POS tags on each line; the token and tag sets should be separated by one
  string (the `section_sep` option), and each token and tag should be
  separated by another (the `token_sep` option).

  The resulting model can be exported using `POSTagger.export/1` and then
  saved to disk.

  *Note:* The training file is read into memory during training.

  Example:
  ```
  Bill|saw|her,NNP|VBD|PRP
  ```

  options:
  |key               |default                       |
  |------------------|------------------------------|
  |`token_sep`       |`" "` (space)                 |
  |`section_sep`     |`"\t"`                        |
  """
  @spec train(train_file :: String.t(), options :: keyword) ::
          POSTagger.model()
  def train(train_file, options \\ []) do
    {tokens, tags} = ingest_file(train_file, options)
    Logger.info("Training file processed; fitting model...")
    model = POSTagger.fit(%{}, tokens, tags)
    Logger.info("Training complete.")
    model
  end

  @doc """
  Tests a model on a given file that is formatted the same way as one supplied
  to `train/2`.

  Returns a map of statistics from evaluating the tagger on the test file.
  Currently, the only statistics collected are the total number of test tokens
  and the ratio of predicted tags that matched the ground truth tags supplied by
  the test file.
  """
  @spec test(
          model :: POSTagger.model(),
          test_file :: String.t(),
          options :: keyword
        ) :: map
  def test(model, test_file, options \\ []) do
    {tokens, tags} = ingest_file(test_file, options)

    output =
      tokens
      |> Enum.map(&POSTagger.tag(model, %{}, &1))
      |> Enum.zip(tags)

    # At this point we have a somewhat tortured list, each item of which has
    # the following contents:
    #
    # {[{token, tag}, {token, tag}, ...], [gold_tag1, gold_tag2, ...]}
    #
    # Because `Enum.zip` only operates on the top level of a list, we need
    # to iterate over this list's items and zip the contents found there in
    # order to compare predicted tags to those provided by the test file
    correct =
      for {predictions, gold} <- output,
          combined = Enum.zip(predictions, gold),
          {{_tok, tag}, gold_tag} <- combined do
        tag == gold_tag
      end

    stats =
      Enum.reduce(correct, {0, 0}, fn r, {correct, total} ->
        {(r && correct + 1) || correct, total + 1}
      end)

    %{
      total_tokens: elem(stats, 1),
      accuracy: elem(stats, 0) / elem(stats, 1)
    }
  end

  defp combine_with_defaults(options) do
    %{
      token_sep: Keyword.get(options, :token_sep, " "),
      section_sep: Keyword.get(options, :section_sep, "\t")
    }
  end

  defp ingest_file(file, options) do
    file
    |> File.stream!([:utf8])
    |> Enum.reduce({[], []}, &process_line(&1, &2, options))
  end

  defp process_line(line, {tokens, tags}, options) do
    options = combine_with_defaults(options)

    [line_tokens, line_tags] =
      line
      |> String.trim()
      |> String.split(options.section_sep)

    split_tok = String.split(line_tokens, options.token_sep)
    split_tags = String.split(line_tags, options.token_sep)

    if length(split_tok) != length(split_tags) do
      Logger.warn("length mismatch", tokens: split_tok, tags: split_tags)
      {tokens, tags}
    else
      {
        [String.split(line_tokens, options.token_sep) | tokens],
        [String.split(line_tags, options.token_sep) | tags]
      }
    end
  end
end
