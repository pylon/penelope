defmodule Penelope.NLP.Tokenize.PennTreebankTokenizer do
  @moduledoc """
    The tokenization scheme used for the creation of the Penn Treebank corpus.
    See ftp://ftp.cis.upenn.edu/pub/treebank/public_html/tokenization.html.

    Some alterations have been made to the original script to better handle
    common Unicode replacement characters.
  """

  @behaviour Penelope.NLP.Tokenize.Tokenizer

  def tokenize(text) do
    text
    |> punctuation()
    |> contractions()
    |> String.replace(~r/\s+/u, " ")
    |> String.replace(~r/^\s+/u, "")
    |> String.replace(~r/\s+$/u, "")
    |> String.split(" ")
  end

  defp punctuation(text) do
    text
    |> String.replace(~r/^"/, "`` ")
    |> String.replace(~r/([ \([{<])"/, "\\1 `` ")
    |> String.replace(~r/\.\.\.|\x{2026}/u, " ... ")
    |> space_pad(~r/[,;:@#$%&]/)
    |> String.replace(~r/([^.])(\.)([\]\)}>"'\x{2019}]*)[\s]*$/u, "\\1 \\2\\3 ")
    |> space_pad(~r/[\]\[\(\)\{\}<>?!\x{2013}\x{2014}\x{2e3a}\x{2e3b}]|--/u)
    |> String.replace(~r/^/, " ")
    |> String.replace(~r/$/, " ")
    |> String.replace(~r/"/, " '' ")
    |> String.replace(~r/([^'])['\x{2019}] /u, "\\1 ' ")
  end

  defp contractions(text) do
    text
    # we might not want to do the 's segmentation here
    |> String.replace(~r/['`\x{2019}](s|m|d|ll|re|ve) /iu, " '\\1 ")
    |> String.replace(~r/n['`\x{2019}]t /iu, " n't ")
    |> String.replace(~r/(c)annot/i, "\\1an not")
    |> String.replace(~r/(d)['`\x{2019}]ye/iu, "\\1' ye")
    |> String.replace(~r/(g)imme/i, "\\1im me ")
    |> String.replace(~r/(g)onna/i, "\\1on na ")
    |> String.replace(~r/(g)otta/i, "\\1ot ta ")
    |> String.replace(~r/(l)emme/i, "\\1em me ")
    |> String.replace(~r/(m)ore['`\x{2019}]n/iu, "\\1ore 'n ")
    |> String.replace(~r/['`\x{2019}](t)(is|was)/iu, "'\\1 \\2 ")
    |> String.replace(~r/(w)anna/i, "\\1an na ")
    |> String.replace(~r/(w)haddya/i, "\\1ha dd ya ")
    |> String.replace(~r/(w)hatcha/i, "\\1ha t cha ")
  end

  defp space_pad(text, regex) do
    String.replace(text, regex, " \\0 ")
  end

end
