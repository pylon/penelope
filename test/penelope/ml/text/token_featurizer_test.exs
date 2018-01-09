defmodule Penelope.ML.Text.TokenFeaturizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Text.TokenFeaturizer

  test "transform" do
    x = [
      [],
      ["test", "sentence"],
      ["another", "test", "sentence"]
    ]
    expect = [
      [],
      [%{"test" => 1.0}, %{"sentence" => 1.0}],
      [%{"another" => 1.0}, %{"test" => 1.0}, %{"sentence" => 1.0}],
    ]
    assert TokenFeaturizer.transform(%{}, %{}, x) === expect
  end
end
