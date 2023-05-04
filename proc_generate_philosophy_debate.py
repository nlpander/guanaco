from pyllamacpp.model import Model
from tqdm import tqdm
import datetime as dt
import numpy as np
import pickle as pkl
import json

import gradio as gr

from absl import flags, app
import logging

from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
import re
import os, sys
import toml
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Command-line flags that can override the values in the config file.
FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'FredRalph_p1.toml', 'Path to the TOML config')
flags.DEFINE_string('ggml-model', None, 'Path to the GGML model')
flags.DEFINE_integer('rounds', None, 'Number of rounds to execute')
flags.DEFINE_string('output', None, 'Output file')
flags.DEFINE_string('temperature-mode','none', 'The style of decay or randomness you desire for the conversation')
flags.DEFINE_float('max-temp-randomness', 0.0, 'Max temperature randomness from baseline')
flags.DEFINE_float('period', None, 'number of periods of exponential decay required in the conversation')
flags.DEFINE_float('decay-constant', 0.05, 'decay constant for the exponential decay')

def load_toml(fname):
    
    with open(fname, 'r') as f:
        cfg = toml.load(f)
        
    return cfg


# Determine who the next speaker is based on the prompt.
# To do this, we determine the last mention of each speaker in the prompt and return the one that was mentioned furthest from the end.
def get_next_speaker(prompt, speakers=['Frederich', 'Ralph']):
    mentions = {s: prompt.rfind(f'{s}:') for s in speakers}    
    return min(mentions, key=mentions.get)

# Get the last utterance of the debate.
# To do this, we determine the last mention of each speaker in the output and find the one that was mentioned last. This marks the start
# of the last utterance.
def get_last_utterance(output, speakers=['Frederich', 'Ralph']):
    mentions = {s: output.rfind(f'{s}:') for s in speakers}
    last_speaker_idx = max(mentions.values())
    return output[last_speaker_idx:]

# split the segment if a speaker section begins mid-sentence
def split_segment_speaker_midsentence(segment, speakers=['Frederich','Ralph']):
    
    for speaker in speakers:
        
        tmp = re.findall(f'\S({speaker}: )', segment)
        if len(tmp) != 0:
            subsegs = segment.split(tmp[0])
            output_segment = subsegs[0] + '\n' + speaker + ': ' + subsegs[1]
            break
        else:
            output_segment = segment
            
    return output_segment

#get new prompt
def get_new_prompt(output, conversation_list, n_keep=150, speakers=['Frederich','Ralph']):
    
    ### keep start of prompt     
    start_prompt = '\n'.join(output.split('\n')[0:2])

    ### sometimes we get a leading whitespace in the string - remove this
    start_prompt = re.sub("^\s*","", start_prompt)
    start_prompt += '\n'
    
    N0 = len(TreebankWordTokenizer().tokenize(start_prompt))
    
    ### split the rest of prompt sentence by sentences counting the overall size of the context
    ### add sentences from the end backwards
    ### till the n_keep limit has been reached. In llama.cpp this is set to n_ctx / 2 
    
    total_len = N0
    k = 0 
    
    next_output = '\n'.join(output.split('\n')[2:])
    
    segments = sent_tokenize(next_output)[::-1]
    segments_to_keep = []
    all_segments = []

    for i in range(0,len(segments)):
        
        segment = segments[i]
        N = len(TreebankWordTokenizer().tokenize(segment))
        total_len += N
        
        if total_len < n_keep:
            segments_to_keep.append(segment)
            
        all_segments.append(segment)
        
    segments_to_keep = segments_to_keep[::-1]
    all_segments = all_segments[::-1]
    
    ### construct the new prompt with the start prompt and the segments added 
    
    new_prompt = start_prompt
    #conversation = ''
    
    for j in range(0,len(all_segments)):
        
        segment = all_segments[j].replace('\n','')
        
        ### append full sentences to the conversation list and the new prompt to inject
        if len(re.findall("[.!?]",segment)) != 0:

            #if speaker contained in segment split the segment
            segment = split_segment_speaker_midsentence(segment, speakers)
                        
            if segment not in conversation_list:                
                conversation_list.append(segment)
                
            if j < len(segments_to_keep):
                new_prompt = new_prompt + '\n' + segments_to_keep[j]
 
    ### get the new speaker and append them to the prompt 
    next_speaker = get_next_speaker(new_prompt, speakers)
    new_prompt += '\n' + next_speaker + ': ' 
    
    return new_prompt, conversation_list


def get_temp(baseline_temp, max_randomness):
    return max(0, baseline_temp + ((random.random() - 0.5) * max_randomness))


def get_temperature_exp_decay(n, baseline_temperature, total_rounds, decay_constant=0.05, period=None):
    
    if n < total_rounds:
    
        if period and period < total_rounds:
            schedule = np.array([])
            cycles = int(np.floor(total_rounds/period))

            for i in range(0,cycles):
                x = np.arange(0,period)
                schedule = np.hstack((schedule,baseline_temperature*np.exp(-decay_constant*x)))
        else:

            x = np.arange(0,total_rounds)
            schedule = baseline_temperature*np.exp(-decay_constant*x)

        return schedule[n]
    
    else:
        
        print('conversation round exceeds total rounds')
        return 
            
    
def cli_main(argv):
    cfg = load_toml(FLAGS.config)
    now = dt.datetime.now().replace(microsecond=0)
    formatted_date = now.strftime("%d%m%Y_%H%M%S")
    
    ################
    ## CONFIGURATION
    ################
    cfg['model_params']['ggml_model'] = FLAGS['ggml-model'].value or cfg['model_params']['ggml_model']
    rounds = FLAGS['rounds'].value or cfg['debate_params']['rounds']
    fname_out = FLAGS['output'].value or f'{FLAGS["config"].value.strip(".toml")}_{formatted_date}.txt'
    max_temp_randomness = FLAGS['max-temp-randomness'].value
    temp_mode = FLAGS['temperature-mode'].value
    decay_constant = FLAGS['decay-constant'].value
    period = FLAGS['period'].value

    start_prompt = cfg['debate_params']['initial_prompt']
    speaker1 = cfg['debate_params']['speaker1_fullname'].split(' ')[0]
    speaker2 = cfg['debate_params']['speaker2_fullname'].split(' ')[0]

    # Keep track of baseline temp since we will be modifying this pseudo-randomly.
    baseline_temp = cfg['gpt_params']['temp']
    
    logger.info('Using model: %s', cfg['model_params']['ggml_model'])
    logger.info('Rounds: %d', rounds)
    logger.info('Output file: %s', fname_out)
    logger.info('Baseline temp: %f', baseline_temp)
    logger.info('Max temp randomness: %f', max_temp_randomness)
    logger.info('Speaker 1: %s', speaker1)
    logger.info('Speaker 2: %s', speaker2)
    

    print(speaker1)
    print(speaker2)

    # Load the model.
    model = Model(**cfg['model_params'])

    # Keep track of the debate. As the debate progresses, we will add the last utterance of each round to this list.
    all_outputs = [start_prompt]
    conversation_list = []
    conv_len = len(conversation_list)

    # Take the initial prompt and prepare it for the first round.
    prompt = start_prompt

    # Run the debate for the specified number of rounds. Each round results in a new answer from one of the speakers. Speakers
    # rotate after each round.
    for n in tqdm(range(0,rounds)):            
        
        if temp_mode == 'none':
            cfg['gpt_params']['temp'] = baseline_temp
        elif temp_mode == 'rand':
            temperature = get_temp(baseline_temp, max_temp_randomness)
        elif temp_mode == 'exp':
            cfg['gpt_params']['temp'] = get_temperature_exp_decay(n, baseline_temp, rounds, decay_constant, period)
        
        prompt, conversation_list = exec_round(model, cfg, prompt, converesation_list, temperature, speaker1, speaker2)
        
        print('========= output ==========')

        print('\n'.join(conversation_list[conv_len:]))

        print('========= output ==========')    

        # update conversation length
        conv_len = len(conversation_list)        
        
    with open(fname_out, 'w') as f:
        f.write('\n'.join(conversation_list))

    
def exec_round(model, cfg, prompt, conversation_list, temperature, speaker1, speaker2):
    cfg['gpt_params']['temp'] = temperature
    print('tmp', temperature)
    print('prompt:', prompt)
    output = model.generate(prompt, **{**cfg['gpt_params'],**{'n_threads':cfg['debate_params']['n_threads']}})
    print('output:', output)
    prompt, conversation_list = get_new_prompt(output, conversation_list, \
                            n_keep=int(2*cfg['model_params']['n_ctx']/3),\
                            speakers=[speaker1,speaker2])

    return prompt, conversation_list



with gr.Blocks() as ui:
    cfg = load_toml('/home/fabian/dev/guanaco/JungAurelius_p3.toml')
    model = Model(**cfg['model_params'])
    state = gr.State(json.dumps({
        'prompt': cfg['debate_params']['initial_prompt'],
        'conv_list': [],
    }))

    def gradio_in(speaker1, speaker2, temperature, state):
        parsed = json.loads(state)
        prompt, conv_list = exec_round(model, cfg, parsed['prompt'], parsed['conv_list'], temperature, speaker1, speaker2)
        return ['\n'.join(conv_list), json.dumps({
                'prompt': prompt,
                'conv_list': conv_list
        })]

    with gr.Row() as row:
        with gr.Column() as col:
            speaker1 = gr.Textbox(label='Speaker 1')
            speaker2 = gr.Textbox(label='Speaker 2')
            temperature = gr.Slider(0, 1, label='Temperature')
            
            next_round_btn = gr.Button('Next round')     

        with gr.Column() as col:
            output = gr.Textbox(label='Output')

        next_round_btn.click(
            fn=gradio_in,
            inputs=[speaker1, speaker2, temperature, state],
            outputs=[output, state])

        

ui.launch()
   
        
# # if __name__ == "__main__":
# demo = gr.Interface(
#     fn=gradio_in,
#     inputs=["text", "text", gr.Slider(0, 1)],
#     outputs="text")
# demo.launch()
# # app.run(main)
