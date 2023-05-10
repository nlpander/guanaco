import json
from typing import Callable, List
from dataclasses import dataclass, asdict


import gradio as gr
from pyllamacpp.model import Model


@dataclass
class UIState:
    prompt: str
    conversation_list: List[str]

    def to_json(self):
        return json.dumps(asdict(self))


def gen_ui(model: Model, initial_prompt: str, exec_round: Callable, cfg):
    def _next_round_btn_on_click(speaker1, speaker2, temperature, state, num_rounds):
        state = UIState(**json.loads(state))
        for i in range(int(num_rounds)):
            prompt, conversation_list = exec_round(
                model,
                cfg,
                state.prompt,
                state.conversation_list,
                temperature,
                speaker1,
                speaker2,
            )
            conversation_list += ["------------------------"]
            state = UIState(prompt, conversation_list)
            yield ["\n".join(state.conversation_list), state.to_json()]

    with gr.Blocks() as ui:
        initial_state = UIState(prompt=initial_prompt, conversation_list=[])
        state = gr.State(initial_state.to_json())

        with gr.Row() as row:
            with gr.Column() as col:
                speaker1 = gr.Textbox(
                    label="Speaker 1", value=cfg["debate_params"]["speaker1_fullname"]
                )
                speaker2 = gr.Textbox(
                    label="Speaker 2", value=cfg["debate_params"]["speaker2_fullname"]
                )
                num_rounds = gr.Number(value=1, label="Number of rounds")
                temperature = gr.Slider(0, 1, label="Temperature")
                next_round_btn = gr.Button("Next round")

            with gr.Column() as col:
                output = gr.Textbox(label="Output")

            next_round_btn.click(
                fn=_next_round_btn_on_click,
                inputs=[speaker1, speaker2, temperature, state, num_rounds],
                outputs=[output, state],
            )

    return ui
