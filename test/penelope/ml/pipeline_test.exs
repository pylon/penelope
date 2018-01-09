defmodule Penelope.ML.PipelineTest do
  use ExUnit.Case, async: true

  alias Penelope.ML.Pipeline

  @x_train [3, 2, 1, 3, 2, 1]
  @y_train ["c", "b", "a", "c", "b", "a"]

  @pipeline [
    {Penelope.ML.PipelineTest.DummyStage, []},
    {Penelope.ML.PipelineTest.Transducer, [factor: 2]},
    {Penelope.ML.PipelineTest.Predictor,  []}
  ]

  test "fit/export/compile" do
    model = Pipeline.fit(
      %{},
      @x_train,
      @y_train,
      @pipeline
    )

    params = Pipeline.export(model)
    assert params === Pipeline.export(Pipeline.compile(params))
  end

  test "transform" do
    model = Pipeline.fit(
      %{},
      @x_train,
      @y_train,
      @pipeline
    )
    expect = [6, 4, 2, 6, 4, 2]
    assert Pipeline.transform(model, %{}, @x_train) === expect

    model = Pipeline.compile(Pipeline.export(model))
    assert Pipeline.transform(model, %{}, @x_train) === expect
  end

  test "predict class" do
    model = Pipeline.fit(%{}, @x_train, @y_train, @pipeline)
    predictions = Pipeline.predict_class(model, %{}, @x_train)
    assert predictions === @y_train

    model = Pipeline.compile(Pipeline.export(model))
    predictions = Pipeline.predict_class(model, %{}, @x_train)
    assert predictions === @y_train
  end

  test "predict probability" do
    model = Pipeline.fit(%{}, @x_train, @y_train, @pipeline)
    predictions =
      model
      |> Pipeline.predict_probability(%{}, @x_train)
      |> Enum.map(&Enum.max_by(&1, fn {_, v} -> v end))
      |> Enum.map(fn {k, _} -> k end)
    assert predictions === @y_train
  end

  test "predict sequence" do
    model = Pipeline.fit(%{}, @x_train, @y_train, @pipeline)
    predictions =
      model
      |> Pipeline.predict_sequence(%{}, @x_train)
      |> Enum.map(fn [y] -> y end)
    assert predictions === @y_train
  end

  defmodule DummyStage do
  end

  defmodule Transducer do
    def transform(model, _context, x) do
      Enum.map(x, fn x -> x * model.factor end)
    end
  end

  defmodule Predictor do
    def fit(_context, x, y, _options) do
      %{memo: Map.new(Enum.zip(x, y))}
    end

    def predict_class(model, _context, x) do
      Enum.map(x, &model.memo[&1])
    end

    def predict_probability(model, _context, x) do
      Enum.map(x, &%{model.memo[&1] => 1.0})
    end

    def predict_sequence(model, _context, x) do
      Enum.map(x, &[model.memo[&1]])
    end

    def compile(params) do
      %{memo: params["memo"]}
    end

    def export(model) do
      %{"memo" => model.memo}
    end
  end
end
