defmodule Penelope.ML.Feature.ContextFeaturizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Feature.ContextFeaturizer

  test "transform" do
    model = %{keys: ["k1", "k2"]}
    context = %{k1: "v1", k2: "v2"}

    x = [[], ["test", "sentence"], ["another"]]
    expect = [
      [],
      [%{k1: "v1", k2: "v2"}, %{k1: "v1", k2: "v2"}],
      [%{k1: "v1", k2: "v2"}]
    ]
    assert ContextFeaturizer.transform(model, context, x) == expect
  end
end
