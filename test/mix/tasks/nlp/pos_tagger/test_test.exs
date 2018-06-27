defmodule Mix.Tasks.Nlp.PosTagger.TestTest do
  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias Mix.Tasks.Nlp.PosTagger.{Test, Train}

  setup_all do
    model_file = "/tmp/penelope_mix_tasks_nlp_pos_model.json"
    on_exit(fn -> File.rm(model_file) end)
    {:ok, model_file: model_file}
  end

  test "model testing", %{model_file: model_file} do
    input_file = "test/data/pos_train.txt"

    assert capture_log(fn ->
             Train.run([
               input_file,
               model_file,
               "--section-sep",
               " ",
               "--token-sep",
               "|"
             ])
           end) =~ ~r/Model saved/

    assert capture_log(fn ->
             Test.run([
               model_file,
               input_file,
               "--section-sep",
               " ",
               "--token-sep",
               "|"
             ])
           end) =~ ~r/Test complete./
  end
end
