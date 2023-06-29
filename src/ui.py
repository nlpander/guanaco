import json
from typing import Callable, List
from dataclasses import dataclass, asdict
from src import segments
import src.vicuna.segments as vic_segments
from tqdm import tqdm

import gradio as gr
from pyllamacpp.model import Model


@dataclass
class UIState:
    prompt: str
    conversation_list: List[str]
    rounds_elapsed: int

    def to_json(self):
        return json.dumps(asdict(self))


def gen_ui(
    model: Model, initial_prompt: str, prefix: str, exec_round: Callable, cfg, model_type: str
):
    segs = segments

    if model_type == "vicuna":
        segs = vic_segments
    elif model_type == "llama":
        segs = segments

    def _next_round_btn_on_click(
        initial_prompt,
        prefix,
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
        
        #print(prefix)
        #print(initial_prompt)

        if state.rounds_elapsed == 0:
          state.prompt = prefix + '\n' + initial_prompt
        else:
          state.prompt = state.prompt

        #state.prompt = state.prompt or initial_prompt

        cfg["gpt_params"]["top_k"] = int(top_k)
        cfg["gpt_params"]["top_p"] = float(top_p)
        cfg["gpt_params"]["repeat_last_n"] = int(repeat_last_n)
        cfg["gpt_params"]["repeat_penalty"] = float(repeat_penalty)

        if additional_ctx:
            state.prompt = segs.strip_last_speaker_add_context(
                state.prompt, additional_ctx
            )
        
        for i in tqdm(range(int(num_rounds))):

            stream = exec_round(model, cfg, state.prompt, temperature)
            output = []
            for s in stream:
                if state.conversation_list:
                    state.conversation_list[-1] += s
                else:
                    state.conversation_list = [s]
                state = UIState(state.prompt, state.conversation_list, state.rounds_elapsed)
                output += [s]
                yield [
                    "\n".join(state.conversation_list),
                    state.to_json(),
                    gr.Textbox.update(value=""),
                ]

            if model_type == 'llama':
                prompt, conversation_list = segs.get_new_prompt(
                    prefix,
                    state.prompt + ''.join(output),
                    state.conversation_list,
                    cfg['debate_params']['tokenizer_path'],
                    n_keep=int(2 * cfg["model_params"]["n_ctx"] / 3),
                    speakers=[speaker1, speaker2],
                )                
            
            elif model_type == 'vicuna':
                prompt, conversation_list = segs.get_new_prompt(
                    output,
                    state.conversation_list,
                    n_keep=int(2 * cfg["model_params"]["n_ctx"] / 3),
                    speakers=[speaker1, speaker2],
                )

            state.rounds_elapsed += 1

            print(state.prompt)
            print('###########################')
            print(state.rounds_elapsed)
            print('###########################')        

            yield [
                "\n".join(conversation_list),
                UIState(state.prompt, state.conversation_list,state.rounds_elapsed).to_json(),
                gr.Textbox.update(value=""),
            ]

    with gr.Blocks() as ui:
        initial_state = UIState(prompt=None, conversation_list=[], rounds_elapsed=0)
        state = gr.State(initial_state.to_json())

        with gr.Row() as row:
            with gr.Column() as col:
                speaker1 = gr.Textbox(
                    label="Speaker 1", value=cfg["debate_params"]["speaker1_fullname"]
                )
                speaker2 = gr.Textbox(
                    label="Speaker 2", value=cfg["debate_params"]["speaker2_fullname"]
                )

                prefix_box = gr.Textbox(label="Prefix", value=prefix)

                initial_prompt = gr.Textbox(
                    label="Initial Prompt", value=cfg["debate_params"]["initial_prompt"]
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
                output = gr.Textbox(label="Output", streaming=True, live=True)

            next_round_btn.click(
                fn=_next_round_btn_on_click,
                inputs=[
                    initial_prompt,
                    prefix_box,
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
