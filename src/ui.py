import json
from typing import Callable, List
from dataclasses import dataclass, asdict
from src import segments
from tqdm import tqdm

import gradio as gr
from pyllamacpp.model import Model


@dataclass
class UIState:
    prompt: str
    conversation_list: List[str]

    def to_json(self):
        return json.dumps(asdict(self))


def gen_ui(model: Model, initial_prompt: str, exec_round: Callable, cfg):
    def _next_round_btn_on_click(
        initial_prompt,
        speaker1,
        speaker2,
        temperature,
        state,
        num_rounds,
        additional_ctx,
        top_k,
        top_p,
        repeat_last_n,
        repeat_penalty,
    ):
        state = UIState(**json.loads(state))
        state.prompt = state.prompt or initial_prompt

        cfg["gpt_params"]["top_k"] = int(top_k)
        cfg["gpt_params"]["top_p"] = float(top_p)
        cfg["gpt_params"]["repeat_last_n"] = int(repeat_last_n)
        cfg["gpt_params"]["repeat_penalty"] = float(repeat_penalty)

        if additional_ctx:
            state.prompt = segments.strip_last_speaker_add_context(
                state.prompt, additional_ctx
            )

        for i in tqdm(range(int(num_rounds))):
            prompt, conversation_list = exec_round(
                model,
                cfg,
                state.prompt,
                cfg["debate_params"]["ratio_keep"],
                state.conversation_list,
                temperature,
                speaker1,
                speaker2,
            )
            state = UIState(prompt, conversation_list)
            yield [
                "\n".join(state.conversation_list),
                state.to_json(),
                gr.Textbox.update(value=""),
            ]

    with gr.Blocks() as ui:
        initial_state = UIState(prompt=None, conversation_list=[])
        state = gr.State(initial_state.to_json())

        with gr.Row() as row:
            with gr.Column() as col:
                speaker1 = gr.Textbox(
                    label="Speaker 1", value=cfg["debate_params"]["speaker1_fullname"]
                )
                speaker2 = gr.Textbox(
                    label="Speaker 2", value=cfg["debate_params"]["speaker2_fullname"]
                )
                initial_prompt = gr.Textbox(
                    label="Prompt", value=cfg["debate_params"]["initial_prompt"]
                )

                with gr.Row() as row:
                    num_rounds = gr.Number(value=1, label="Number of rounds")
                    temperature = gr.Slider(
                        0, 1, value=cfg["gpt_params"]["temp"], label="Temperature"
                    )
                    top_k = gr.Number(value=cfg["gpt_params"]["top_k"], label="Top K")
                    top_p = gr.Number(value=cfg["gpt_params"]["top_p"], label="Top P")
                    repeat_last_n = gr.Number(
                        value=cfg["gpt_params"]["repeat_last_n"], label="Repeat last N"
                    )
                    repeat_penalty = gr.Number(
                        value=cfg["gpt_params"]["repeat_penalty"],
                        label="Repeat Penalty",
                    )

                additional_ctx_textbox = gr.Textbox(label="Provide context")
                next_round_btn = gr.Button("Next round")

            with gr.Column() as col:
                output = gr.Textbox(label="Output")

            next_round_btn.click(
                fn=_next_round_btn_on_click,
                inputs=[
                    initial_prompt,
                    speaker1,
                    speaker2,
                    temperature,
                    state,
                    num_rounds,
                    additional_ctx_textbox,
                    top_k,
                    top_p,
                    repeat_last_n,
                    repeat_penalty,
                ],
                outputs=[output, state, additional_ctx_textbox],
            )

    return ui
