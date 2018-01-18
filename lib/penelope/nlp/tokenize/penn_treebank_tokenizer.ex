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
    |> String.replace(
      ~r/([^.])(\.)([\]\)}>"'\x{2019}]*)[\s]*$/u,
      "\\1 \\2\\3 "
    )
    |> space_pad(~r/[\]\[\(\)\{\}<>?!\x{2013}\x{2014}\x{2e3a}\x{2e3b}]|--/u)
    |> String.replace(~r/^/, " ")
    |> String.replace(~r/$/, " ")
    |> String.replace(~r/"/, " '' ")
    |> String.replace(~r/([^'])['\x{2019}] /u, "\\1 ' ")
  end

  defp contractions(text) do
    # we might not want to do the 's segmentation here
    text
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

  @doc """
  Detokenize a string tokenized by the Penn Treebank tokenizer.
  The PTB tokenization scheme is lossy; attributes like capitalization,
  multiple spaces, and padding around certain punctuation will be removed
  from the output.
  """
  def detokenize(tokens) do
    tokens
    |> Enum.join(" ")
    |> repunctuate()
    |> recontract()
  end

  defp repunctuate(text) do
    # best effort at replacing spaces after full stops
    # ideally, this wouldn't be necessary, as inputs to
    # the tokenizer should generally be single sentences,
    # but they're not guaranteed to be
    # - a non-period followed by a period, followed by anything
    #   other than a single letter (except I or A) + word boundary
    text
    |> String.replace(~r/^``/, ~S("))
    |> String.replace(~r/([ \([{<])``/, ~S(\\1"))
    |> String.replace(~r/ \.\.\.\s*/, "...")
    |> String.replace(~r/\s([,;:@%])/, "\\1")
    |> String.replace(~r/([@#$])\s/, "\\1")
    |> String.replace(
      ~r/([^.])\s(\.)([\]\)}>"'\x{2019}]*)\s*$/u,
      "\\1\\2\\3"
    )
    |> String.replace(~r/\s([\]\)\}>?!\x{2014}\x{2e3b}]|--)/u, "\\1")
    |> String.replace(~r/([\[\(\{<\x{2013}\x{2e3a}]|--)\s/u, "\\1")
    |> String.replace(~r/\s''/, ~S("))
    |> String.replace(~r/([^'])['\x{2019}]/u, "\\1'")
    |> String.replace(~r/([^.])\.(?![^\dai]\b)/, "\\1. ")
    |> String.replace(~r/\. (\d|$)/, ".\\1")
    |> String.replace(~r/([?!])(?=\w)/, "\\1 ")
  end

  defp recontract(text) do
    text
    |> String.replace(~r/ '(s|m|d|ll|re|ve)/i, "'\\1")
    |> String.replace(~r/ n't/i, "n't")
    |> String.replace(~r/(c)an not/i, "\\1annot")
    |> String.replace(~r/(d) 'ye/i, "\\1'ye")
    |> String.replace(~r/(g)im me/i, "\\1imme")
    |> String.replace(~r/(g)on na/i, "\\1onna")
    |> String.replace(~r/(g)ot ta/i, "\\1otta")
    |> String.replace(~r/(l)em me/i, "\\1emme")
    |> String.replace(~r/(m)ore 'n/i, "\\1ore'n")
    |> String.replace(~r/'(t) (is|was)/i, "'\\1\\2")
    |> String.replace(~r/(w)an na/i, "\\1anna")
    |> String.replace(~r/(w)ha dd ya/i, "\\1haddya")
    |> String.replace(~r/(w)ha t cha/i, "\\1hatcha")
  end
end
