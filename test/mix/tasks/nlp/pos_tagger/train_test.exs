defmodule Mix.Tasks.Nlp.PosTagger.TrainTest do
  use ExUnit.Case, async: false

  import ExUnit.CaptureLog

  alias Mix.Tasks.Nlp.PosTagger.Train

  setup_all do
    model_file = "/tmp/penelope_mix_tasks_nlp_pos_model.json"
    on_exit(fn -> File.rm(model_file) end)
    {:ok, model_file: model_file}
  end

  test "model training and testing", %{model_file: model_file} do
    input_file = "test/data/pos_train.txt"

    assert capture_log(fn ->
             Train.run([
               input_file,
               model_file,
               "--section-sep",
               " ",
               "--token-sep",
               "|",
               "--test-file",
               input_file
             ])
           end) =~ ~r/Test complete./
  end
end
