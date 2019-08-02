defmodule Penelope.ML.SVM.ClassifierTest do
  @moduledoc """
  These tests verify the SVM classifier.

  Stress tests are disabled by default, but they can be used to detect
  memory leaks in the SVM NIF.
  """

  use ExUnit.Case, async: true

  import ExUnitProperties
  import Penelope.TestUtility

  alias Penelope.ML.SVM.Classifier
  alias Penelope.ML.Vector
  alias StreamData, as: Gen

  # embarrassingly separable training data
  @x_train [
             [-1, 1],
             [1, 1],
             [1, -1],
             [-1, 1],
             [1, 1],
             [1, -1]
           ]
           |> Enum.map(&Vector.from_list/1)

  @y_train ["c", "b", "a", "c", "b", "a"]

  test "fit/export/compile" do
    assert_raise(fn ->
      Classifier.fit(%{}, [hd(@x_train)], @y_train)
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, [hd(@y_train)])
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, @y_train, kernel: nil)
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, @y_train, degree: -1)
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, @y_train, gamma: -1)
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, @y_train, c: 0)
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, @y_train, weights: %{1 => nil})
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, @y_train, epsilon: -1)
    end)

    assert_raise(fn ->
      Classifier.fit(%{}, @x_train, @y_train, cache_size: -1)
    end)

    check all(
            kernel <- Gen.one_of([:linear, :rbf, :poly, :sigmoid]),
            degree <- Gen.positive_integer(),
            gamma <- Gen.float(min: 1.0e-5),
            coef0 <- Gen.float(min: 1.0e-5, max: 1),
            c <- Gen.float(min: 1.0e-5, max: 1000),
            weights <- Gen.list_of(Gen.float(min: 1.0e-5), length: 3),
            epsilon <- Gen.float(min: 1.0e-5, max: 0.5),
            cache_size <- Gen.integer(1..16),
            shrinking? <- Gen.boolean(),
            probability? <- Gen.boolean()
          ) do
      options = [
        kernel: kernel,
        degree: degree,
        gamma: gamma,
        coef0: coef0,
        c: c,
        weights:
          ["a", "b", "c"]
          |> Enum.zip(weights)
          |> Enum.into(%{}),
        epsilon: epsilon,
        cache_size: cache_size,
        shrinking?: shrinking?,
        probability?: probability?
      ]

      model = Classifier.fit(%{}, @x_train, @y_train, options)

      params = Classifier.export(model)
      assert params === Classifier.export(Classifier.compile(params))
    end
  end

  test "predict class" do
    model = Classifier.fit(%{}, @x_train, @y_train)
    predictions = Classifier.predict_class(model, %{}, @x_train)
    assert predictions === @y_train
  end

  test "predict probability" do
    assert_raise(fn ->
      model = Classifier.fit(%{}, @x_train, @y_train)
      Classifier.predict_probability(model, %{}, @x_train)
    end)

    model = Classifier.fit(%{}, @x_train, @y_train, probability?: true)

    predictions =
      model
      |> Classifier.predict_probability(%{}, @x_train)
      |> Enum.map(&Enum.max_by(&1, fn {_, v} -> v end))
      |> Enum.map(fn {k, _} -> k end)

    assert predictions === @y_train
  end

  test "global parallelism" do
    tasks =
      Task.async_stream(
        1..1000,
        fn _i ->
          model = Classifier.fit(%{}, @x_train, @y_train)

          params = Classifier.export(model)
          Classifier.compile(params)

          predictions = Classifier.predict_class(model, %{}, @x_train)
          assert predictions === @y_train
        end,
        ordered: false
      )

    Stream.run(tasks)
  end

  test "shared parallelism" do
    model = Classifier.fit(%{}, @x_train, @y_train)

    tasks =
      Task.async_stream(
        1..1000,
        fn _i ->
          predictions = Classifier.predict_class(model, %{}, @x_train)
          assert predictions === @y_train
        end,
        ordered: false
      )

    Stream.run(tasks)
  end

  @tag :stress
  test "fit stress" do
    for _ <- 1..200_000 do
      Classifier.fit(%{}, @x_train, @y_train, probability?: true)
      :erlang.garbage_collect()
    end
  end

  @tag :stress
  test "export/compile stress" do
    model = Classifier.fit(%{}, @x_train, @y_train, probability?: true)

    for _ <- 1..1_000_000 do
      params = Classifier.export(model)
      Classifier.compile(params)
      :erlang.garbage_collect()
    end
  end

  @tag :stress
  test "predict stress" do
    model = Classifier.fit(%{}, @x_train, @y_train, probability?: true)

    for _ <- 1..1_500_000 do
      Classifier.predict_class(model, %{}, @x_train)
      Classifier.predict_probability(model, %{}, @x_train)
      :erlang.garbage_collect()
    end
  end
end
