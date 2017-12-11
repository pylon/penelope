defmodule Penelope.ML.SVM.ClassifierTest do
  @moduledoc """
  These tests verify the SVM classifier.

  Stress tests are disabled by default, but they can be used to detect
  memory leaks in the SVM NIF.
  """

  use ExUnit.Case, async: true

  import ExUnitProperties
  import Penelope.TestUtility

  alias StreamData, as: Gen
  alias Penelope.ML.Vector
  alias Penelope.ML.SVM.Classifier

  # embarrassingly separable training data
  @x_train [[-1,  1],
            [ 1,  1],
            [ 1, -1],
            [-1,  1],
            [ 1,  1],
            [ 1, -1]]
           |> Enum.map(&Vector.from_list/1)
  @y_train [3, 2, 1, 3, 2, 1]

  test "fit/export/compile" do
    assert_raise fn ->
      Classifier.fit %{}, [hd(@x_train)], @y_train
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, [hd(@y_train)]
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, @y_train, kernel: nil
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, @y_train, degree: -1
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, @y_train, gamma: -1
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, @y_train, c: 0
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, @y_train, weights: %{1 => nil}
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, @y_train, epsilon: -1
    end
    assert_raise fn ->
      Classifier.fit %{}, @x_train, @y_train, cache_size: -1
    end

    check all kernel       <- Gen.one_of([:linear, :rbf, :poly, :sigmoid]),
              degree       <- Gen.positive_integer(),
              gamma        <- gen_non_neg_float(),
              coef0        <- gen_float(),
              c            <- gen_pos_float(),
              weights      <- Gen.list_of(gen_non_neg_float(), length: 3),
              epsilon      <- Gen.uniform_float(),
              cache_size   <- Gen.integer(1..16),
              shrinking?   <- Gen.boolean(),
              probability? <- Gen.boolean() do
      options = [
        kernel:       kernel,
        degree:       degree,
        gamma:        gamma,
        coef0:        coef0,
        c:            c,
        weights:      1..3
                      |> Enum.zip(weights)
                      |> Enum.into(%{}),
        epsilon:      epsilon,
        cache_size:   cache_size,
        shrinking?:   shrinking?,
        probability?: probability?
      ]
      {context, _x, _y} = Classifier.fit %{}, @x_train, @y_train, options

      params = Classifier.export(context)
      assert params === context
                        |> Classifier.compile(params)
                        |> Classifier.export()
    end
  end

  test "predict class" do
    {context, _x, _y} = Classifier.fit(%{}, @x_train, @y_train)
    predictions = Enum.map(@x_train, &Classifier.predict_class(context, &1))
    assert predictions === @y_train
  end

  test "predict probability" do
    assert_raise fn ->
      {context, _x, _y} = Classifier.fit(%{}, @x_train, @y_train)
      Classifier.predict_probability context, hd(@x_train)
    end

    {context, _x, _y} =
      Classifier.fit(%{}, @x_train, @y_train, probability?: true)

    predictions =
      @x_train
      |> Enum.map(&Classifier.predict_probability(context, &1))
      |> Enum.map(&Enum.max_by(&1, fn {_, v} -> v end))
      |> Enum.map(fn {k, _} -> k end)
    assert predictions === @y_train
  end

  @tag :stress
  test "fit stress" do
    for _ <- 1..200_000 do
      Classifier.fit(%{}, @x_train, @y_train, probability?: true)
    end
  end

  @tag :stress
  test "export/compile stress" do
    {context, _x, _y} =
      Classifier.fit(%{}, @x_train, @y_train, probability?: true)

    for _ <- 1..5_000_000 do
      params = Classifier.export(context)
      Classifier.compile(context, params)
    end
  end

  @tag :stress
  test "predict stress" do
    {context, _x, _y} =
      Classifier.fit(%{}, @x_train, @y_train, probability?: true)

    for _ <- 1..10_000_000 do
      Classifier.predict_class context, hd(@x_train)
      Classifier.predict_probability context, hd(@x_train)
    end
  end
end
