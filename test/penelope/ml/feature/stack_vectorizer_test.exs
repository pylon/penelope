defmodule Penelope.ML.Feature.StackVectorizerTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Feature.StackVectorizer
  alias Penelope.ML.Vector

  @x_train Enum.map([3, 2, 1, 3, 2, 1], &Vector.from_list([&1]))
  @y_train ["c", "b", "a", "c", "b", "a"]

  @stack [
    {Penelope.ML.Feature.StackVectorizerTest.DummyFeature, []},
    {Penelope.ML.Feature.StackVectorizerTest.Transducer1, [factor: 2]},
    {Penelope.ML.Feature.StackVectorizerTest.Transducer2, []}
  ]

  test "fit/export/compile" do
    model =
      StackVectorizer.fit(
        %{},
        @x_train,
        @y_train,
        @stack
      )

    params = StackVectorizer.export(model)

    assert params ===
             StackVectorizer.export(StackVectorizer.compile(params))
  end

  test "transform" do
    model =
      StackVectorizer.fit(
        %{},
        @x_train,
        @y_train,
        @stack
      )

    expect = [
      [6, 3 / 12],
      [4, 2 / 12],
      [2, 1 / 12],
      [6, 3 / 12],
      [4, 2 / 12],
      [2, 1 / 12]
    ]

    x = StackVectorizer.transform(model, %{}, @x_train)
    assert x === Enum.map(expect, &Vector.from_list/1)

    model = StackVectorizer.compile(StackVectorizer.export(model))
    x = StackVectorizer.transform(model, %{}, @x_train)
    assert x === Enum.map(expect, &Vector.from_list/1)
  end

  defmodule DummyFeature do
  end

  defmodule Transducer1 do
    def transform(model, _context, x) do
      Enum.map(x, &Vector.scale(&1, model.factor))
    end
  end

  defmodule Transducer2 do
    def fit(_context, x, _y, _options) do
      sum =
        x
        |> Enum.map(&Vector.to_list/1)
        |> Enum.map(&Enum.sum/1)
        |> Enum.sum()

      %{factor: 1.0 / sum}
    end

    def transform(model, _context, x) do
      Enum.map(x, &Vector.scale(&1, model.factor))
    end

    def compile(params) do
      %{factor: params["factor"]}
    end

    def export(model) do
      %{"factor" => model.factor}
    end
  end
end
