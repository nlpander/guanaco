from src import ui, segments, temperature
import src.vicuna.segments as vic_segments


from pyllamacpp.model import Model
from tqdm import tqdm
import datetime as dt
import os

from absl import flags, app
import logging

import toml

logger = logging.getLogger(__name__)


# Command-line flags that can override the values in the config file.
FLAGS = flags.FLAGS
flags.DEFINE_string("config", "FredRalph_p1.toml", "Path to the TOML config")
flags.DEFINE_string("model-path", None, "Path to the GGML model")
flags.DEFINE_integer(
    "n-threads", int(os.cpu_count() / 2), "Number of threads to use for running model"
)
flags.DEFINE_float(
    "ratio_keep", float(2 / 3), "Proportion of context to keep in next input"
)
flags.DEFINE_enum("model-type", "vicuna", ["vicuna", "llama"], "Class of LLM")
flags.DEFINE_integer("rounds", None, "Number of rounds to execute")
flags.DEFINE_string("output", None, "Output file")
flags.DEFINE_string(
    "temperature-mode",
    "none",
    "The style of decay or randomness you desire for the conversation",
)
flags.DEFINE_float(
    "max-temp-randomness", 0.0, "Max temperature randomness from baseline"
)
flags.DEFINE_float(
    "period",
    None,
    "number of periods of exponential decay required in the conversation",
)
flags.DEFINE_float("decay-constant", 0.05, "decay constant for the exponential decay")
flags.DEFINE_bool("gradio", False, "Whether to spin up a UI or not")


def exec_round(
    model, cfg, prompt, ratio_keep, conversation_list, temperature, speaker1, speaker2
):
    cfg["gpt_params"]["temp"] = temperature
    output = model.generate(
        prompt,
        **{**cfg["gpt_params"], **{"n_threads": cfg["debate_params"]["n_threads"]}},
    )

    if FLAGS["model-type"].value == "llama":
        prompt, conversation_list = segments.get_new_prompt(
            output,
            conversation_list,
            n_keep=int(cfg["model_params"]["n_ctx"] * ratio_keep),
            speakers=[speaker1, speaker2],
        )

    elif FLAGS["model-type"].value == "vicuna":
        prompt, conversation_list = vic_segments.get_new_prompt(
            output,
            conversation_list,
            n_keep=int(cfg["model_params"]["n_ctx"] * ratio_keep),
            speakers=[speaker1, speaker2],
        )

    return prompt, conversation_list


def cli_main(
    start_prompt,
    temp_mode,
    baseline_temp,
    max_temp_randomness,
    rounds,
    decay_constant,
    period,
    model,
    cfg,
    speaker1,
    speaker2,
    fname_out,
):
    # Keep track of the debate. As the debate progresses, we will add the last utterance of each round to this list.
    conversation_list = []
    conv_len = len(conversation_list)

    # Take the initial prompt and prepare it for the first round.
    prompt = start_prompt

    # Run the debate for the specified number of rounds. Each round results in a new answer from one of the speakers. Speakers
    # rotate after each round.
    for n in tqdm(range(0, rounds)):
        if temp_mode == "none":
            temperature = baseline_temp
        elif temp_mode == "rand":
            temperature = temperature.get_temp(baseline_temp, max_temp_randomness)
        elif temp_mode == "exp":
            temperature = temperature.get_temperature_exp_decay(
                n, baseline_temp, rounds, decay_constant, period
            )

        prompt, conversation_list = exec_round(
            model, cfg, prompt, conversation_list, temperature, speaker1, speaker2
        )

        print("========= output ==========")
        print("\n".join(conversation_list[conv_len:]))
        print("========= output ==========")

        # update conversation length
        conv_len = len(conversation_list)

    with open(fname_out, "w") as f:
        f.write("\n".join(conversation_list))


def main(argv):
    with open(FLAGS.config, "r") as f:
        cfg = toml.load(f)
    now = dt.datetime.now().replace(microsecond=0)
    formatted_date = now.strftime("%d%m%Y_%H%M%S")

    ################
    ## CONFIGURATION
    ################
    cfg["debate_params"]["ratio_keep"] = FLAGS.ratio_keep
    cfg["model_params"]["model_path"] = (
        FLAGS["model-path"].value or cfg["model_params"]["model_path"]
    )
    cfg["debate_params"]["n_threads"] = (
        FLAGS["n-threads"].value or cfg["debate_params"]["n_threads"]
    )
    rounds = FLAGS["rounds"].value or cfg["debate_params"]["rounds"]
    fname_out = (
        FLAGS["output"].value
        or f'{FLAGS["config"].value.strip(".toml")}_{formatted_date}.txt'
    )
    max_temp_randomness = FLAGS["max-temp-randomness"].value
    temp_mode = FLAGS["temperature-mode"].value
    decay_constant = FLAGS["decay-constant"].value
    period = FLAGS["period"].value

    start_prompt = cfg["debate_params"]["initial_prompt"]
    speaker1 = cfg["debate_params"]["speaker1_fullname"].split(" ")[0]
    speaker2 = cfg["debate_params"]["speaker2_fullname"].split(" ")[0]

    # Keep track of baseline temp since we will be modifying this pseudo-randomly.
    baseline_temp = cfg["gpt_params"]["temp"]

    logger.info("Using model: %s", cfg["model_params"]["model_path"])
    logger.info("Rounds: %d", rounds)
    logger.info("Output file: %s", fname_out)
    logger.info("Baseline temp: %f", baseline_temp)
    logger.info("Max temp randomness: %f", max_temp_randomness)
    logger.info("Speaker 1: %s", speaker1)
    logger.info("Speaker 2: %s", speaker2)

    # Load the model.
    model = Model(**cfg["model_params"])

    if FLAGS.gradio:
        frontend = ui.gen_ui(model, start_prompt, exec_round, cfg)
        frontend.queue().launch()
    else:
        cli_main(
            start_prompt,
            temp_mode,
            baseline_temp,
            max_temp_randomness,
            rounds,
            decay_constant,
            period,
            model,
            cfg,
            speaker1,
            speaker2,
            fname_out,
        )


if __name__ == "__main__":
    app.run(main)
