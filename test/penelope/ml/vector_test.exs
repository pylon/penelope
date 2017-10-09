defmodule Penelope.ML.VectorTest do
  @moduledoc """
  These tests verify the vector module.
  """

  use ExUnit.Case, async: true

  import ExUnitProperties
  import Penelope.ML.Vector

  alias StreamData, as: Gen

  test "size" do
    assert size(empty()) == 0
    assert size(from_list([1])) == 1
    assert size(from_list([1, 2])) == 2
  end

  test "access" do
    assert_raise ArgumentError, fn ->
      get(empty(), 0)
    end
    assert_raise ArgumentError, fn ->
      get(from_list([1]), 1)
    end
    assert_raise ArgumentError, fn ->
      get(from_list([1]), -1)
    end

    assert get(from_list([1]), 0) === 1.0
    assert get(from_list([1, 2]), 0) === 1.0
    assert get(from_list([1, 2]), 1) === 2.0
  end

  test "zeros" do
    assert zeros(0) === empty()
    assert zeros(1) === from_list([0])
    assert zeros(2) === from_list([0, 0])
  end

  test "list conversion" do
    assert to_list(from_list([])) === []
    assert to_list(from_list([1])) === [1.0]
    assert to_list(from_list([1, 2])) === [1.0, 2.0]
  end

  test "scaling" do
    assert scale(empty(), 1) === empty()
    assert scale(from_list([1, 2]), 0) === from_list([0, 0])
    assert scale(from_list([1, 2]), 1) === from_list([1, 2])
    assert scale(from_list([1, 2]), 2) === from_list([2, 4])

    check all a <- gen_float(),
              x <- Gen.list_of(gen_float()) do
      # save the vector to ensure no memory overwrite
      vx = from_list(x)
      vy = scale(vx, a)

      assert vx === from_list(x)
      assert size(vy) === size(vx)

      [x, to_list(vy)]
      |> Enum.zip()
      |> Enum.each(fn {x, y} -> assert float_equals(x * a, y) end)
    end
  end

  test "addition" do
    assert_raise ArgumentError, fn ->
      add(empty(), from_list([1]))
    end
    assert_raise ArgumentError, fn ->
      add(from_list([1]), empty())
    end
    assert_raise ArgumentError, fn ->
      add(from_list([1]), from_list([1, 2]))
    end

    assert add(empty(), empty()) === empty()
    assert add(from_list([1, 2]), from_list([3, 4])) === from_list([4, 6])

    check all n <- Gen.positive_integer(),
              x <- Gen.list_of(gen_float(), length: n),
              y <- Gen.list_of(gen_float(), length: n) do
      # save the vectors to ensure no memory overwrite
      vx = from_list(x)
      vy = from_list(y)
      vz = add(vx, vy)

      assert vx === from_list(x)
      assert vy === from_list(y)
      assert size(vz) === n

      [x, y, to_list(vz)]
      |> Enum.zip()
      |> Enum.each(fn {x, y, z} -> assert float_equals(x + y, z) end)
    end
  end

  test "scale addition" do
    assert_raise ArgumentError, fn ->
      scale_add(empty(), 1, from_list([1]))
    end
    assert_raise ArgumentError, fn ->
      scale_add(from_list([1]), 1, empty())
    end
    assert_raise ArgumentError, fn ->
      scale_add(from_list([1]), 1, from_list([1, 2]))
    end

    assert scale_add(empty(), 1, empty()) === empty()
    assert scale_add(from_list([1, 2]), 2, from_list([3, 4]))
           === from_list([7, 10])

    check all a <- gen_float(),
              n <- Gen.positive_integer(),
              x <- Gen.list_of(gen_float(), length: n),
              y <- Gen.list_of(gen_float(), length: n) do
      # save the vectors to ensure no memory overwrite
      vx = from_list(x)
      vy = from_list(y)
      vz = scale_add(vy, a, vx)

      assert vx === from_list(x)
      assert vy === from_list(y)
      assert size(vz) === n

      [x, y, to_list(vz)]
      |> Enum.zip()
      |> Enum.each(fn {x, y, z} -> assert float_equals(a * x + y, z) end)
    end
  end

  defp gen_float() do
    Gen.map(Gen.tuple({Gen.integer(), Gen.uniform_float()}),
            fn {i, f} -> i * f end)
  end

  defp float_equals(a, b) do
    abs(a - b) < 1.0e-5
  end
end
