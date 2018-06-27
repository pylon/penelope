defmodule Penelope.ML.Text.POSFeaturizer do
  @moduledoc """
  The POS featurizer converts a list of lists of tokens into
  nested lists containing feature maps relevant to POS tagging for each token.

  Features used for the POS tagger are largely inspired by
  [A Maximum Entropy Model for Part-Of-Speech
  Tagging](http://www.aclweb.org/anthology/W96-0213); the following is an
  example feature map for an individual token:

  ```
  token_list = ["it", "is", "a", little-known", "fact"]
  token = "little-known"
  %{
    "has_hyphen" => true,
    "has_digit" => false,
    "has_cap" => false,
    "pre_1" => "l",
    "pre_2" => "li",
    "pre_3" => "lit",
    "pre_4" => "litt",
    "suff_1" => "n",
    "suff_2" => "wn",
    "suff_3" => "own",
    "suff_4" => "nown",
    "tok_-2" => "is",
    "tok_-1" => "a",
    "tok_0" => "little-known",
    "tok_1" => "fact",
    "tok_2" => "",
  }
  ```
  """

  @digit_exp ~r/\d/
  @cap_exp ~r/\p{Lu}/
  @max_affix 4
  @max_window 2

  @doc """
  transforms the token lists into lists of feature maps.
  """
  @spec transform(model :: map, context :: map, x :: [[String.t()]]) :: [
          map
        ]
  def transform(_model, _context, x), do: Enum.map(x, &transform_one/1)

  defp transform_one([]), do: []

  defp transform_one(x) do
    windows = window_features(x)

    x
    |> Enum.with_index()
    |> Enum.map(&feature_map(&1, windows))
  end

  defp feature_map({token, index}, windows) do
    token
    |> single_token_features()
    |> Map.merge(Enum.at(windows, index))
  end

  defp single_token_features(token) do
    token
    |> extract_char_features()
    |> Map.merge(extract_affix_features(token))
  end

  defp extract_char_features(token) do
    %{
      "has_hyphen" =>
        token
        |> String.to_charlist()
        |> Enum.any?(fn x -> x == ?- end)
        |> Kernel.||(false),
      "has_digit" => (Regex.run(@digit_exp, token) && true) || false,
      "has_cap" => (Regex.run(@cap_exp, token) && true) || false
    }
  end

  defp extract_affix_features(token) do
    pres = Enum.reduce(1..@max_affix, %{}, &extract_prefix(&1, &2, token))
    suffs = Enum.reduce(1..@max_affix, %{}, &extract_suffix(&1, &2, token))
    Map.merge(pres, suffs)
  end

  defp extract_prefix(index, acc, token) do
    Map.put(acc, "pre_#{index}", String.slice(token, 0, index))
  end

  defp extract_suffix(index, acc, token) do
    suffix = String.slice(token, -index, index)
    suffix = if suffix == "", do: token, else: suffix
    Map.put(acc, "suff_#{index}", suffix)
  end

  defp window_features(tokens) do
    padded = List.duplicate("", @max_window) ++ tokens
    max_offset = length(tokens) - 1
    Enum.map(0..max_offset, &single_window(&1, padded))
  end

  defp single_window(offset, padded) do
    window_size = @max_window * 2 + 1

    end_feats =
      Enum.reduce(1..@max_window, %{}, fn i, acc ->
        Map.put(acc, "tok_#{i}", "")
      end)

    padded
    |> Enum.drop(offset)
    |> Enum.take(window_size)
    |> Enum.reduce({-@max_window, %{}}, &fill_window/2)
    |> elem(1)
    |> Map.merge(end_feats, fn _k, v1, _v2 -> v1 end)
  end

  defp fill_window(token, {index, feats}) do
    {index + 1, Map.put(feats, "tok_#{index}", token)}
  end
end
