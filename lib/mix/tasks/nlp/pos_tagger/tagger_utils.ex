defmodule Mix.Tasks.Nlp.PosTagger.TaggerUtils do
  @moduledoc """
  Common utility functions for part-of-speech tagger Mix tasks.
  """

  require Logger

  @doc """
  Load a file containing tokenized text along with the tokens'
  part-of-speech tags for either training or testing a tagger.
  """
  @spec ingest_file(file :: String.t(), options :: map) ::
          {[[String.t()]], [[String.t()]]}
  def ingest_file(file, options) do
    file
    |> File.stream!([:utf8])
    |> Enum.reduce({[], []}, &process_line(&1, &2, options))
  end

  defp process_line(line, {tokens, tags}, options) do
    [line_tokens, line_tags] =
      line
      |> String.trim()
      |> String.split(options.section_sep)

    split_tok = String.split(line_tokens, options.token_sep)
    split_tags = String.split(line_tags, options.token_sep)

    if length(split_tok) != length(split_tags) do
      Logger.warn(
        "length mismatch; skipping line",
        tokens: split_tok,
        tags: split_tags
      )

      {tokens, tags}
    else
      {
        [String.split(line_tokens, options.token_sep) | tokens],
        [String.split(line_tags, options.token_sep) | tags]
      }
    end
  end
end
