defmodule Penelope.ML.Feature.MergeFeaturizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Feature.MergeFeaturizer

  @x_train [[], [1], [2, 3]]
  @y_train ["a", "b", "c"]

  @features [
    {Penelope.ML.Feature.MergeFeaturizerTest.DummyFeature, []},
    {Penelope.ML.Feature.MergeFeaturizerTest.Transducer1, [factor: 2.0]},
    {Penelope.ML.Feature.MergeFeaturizerTest.Transducer2, []}
  ]

  test "fit/export/compile" do
    model =
      MergeFeaturizer.fit(
        %{},
        @x_train,
        @y_train,
        @features
      )

    params = MergeFeaturizer.export(model)

    assert params ===
             MergeFeaturizer.export(MergeFeaturizer.compile(params))
  end

  test "transform" do
    model =
      MergeFeaturizer.fit(
        %{},
        @x_train,
        @y_train,
        @features
      )

    expect = [
      [],
      [%{value1: 2.0, value2: 1 / 6}],
      [%{value1: 4.0, value2: 2 / 6}, %{value1: 6.0, value2: 3 / 6}]
    ]

    x = MergeFeaturizer.transform(model, %{}, @x_train)
    assert x === expect

    model = MergeFeaturizer.compile(MergeFeaturizer.export(model))
    x = MergeFeaturizer.transform(model, %{}, @x_train)
    assert x === expect
  end

  defmodule DummyFeature do
  end

  defmodule Transducer1 do
    def transform(model, _context, x) do
      Enum.map(
        x,
        &Enum.map(&1, fn x ->
          %{value1: x * model.factor}
        end)
      )
    end
  end

  defmodule Transducer2 do
    def fit(_context, x, _y, _options) do
      sum =
        x
        |> Enum.map(&Enum.sum/1)
        |> Enum.sum()

      %{factor: 1.0 / sum}
    end

    def transform(model, _context, x) do
      Enum.map(
        x,
        &Enum.map(&1, fn x ->
          %{value2: x * model.factor}
        end)
      )
    end

    def compile(params) do
      %{factor: params["factor"]}
    end

    def export(model) do
      %{"factor" => model.factor}
    end
  end
end
