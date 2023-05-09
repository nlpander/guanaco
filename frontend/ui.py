import json
from typing import Callable, List

import gradio as gr
from pyllamacpp.model import Model



def _next_round_btn_on_click(exec_round, speaker1, speaker2, temperature, state):
    parsed = json.loads(state)
    prompt, conv_list = exec_round(model, cfg, parsed['prompt'], parsed['conv_list'], temperature, speaker1, speaker2)
    return ['\n'.join(conv_list), json.dumps({
            'prompt': prompt,
            'conv_list': conv_list
    })]



def gen_ui(
        model: Model,
        initial_prompt: str,
        exec_round: Callable,
    ):
    with gr.Blocks() as ui:
        state = gr.State(json.dumps({
            'prompt': initial_prompt,
            'conv_list': [],
        }))


        with gr.Row() as row:
            with gr.Column() as col:
                speaker1 = gr.Textbox(label='Speaker 1')
                speaker2 = gr.Textbox(label='Speaker 2')
                temperature = gr.Slider(0, 1, label='Temperature')
                
                next_round_btn = gr.Button('Next round')     

            with gr.Column() as col:
                output = gr.Textbox(label='Output')

            next_round_btn.click(
                fn=lambda *args, **kwargs: _next_round_btn_on_click(exec_round, *args, **kwargs),
                inputs=[speaker1, speaker2, temperature, state],
                outputs=[output, state]) 