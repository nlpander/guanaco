from pyllamacpp.model import Model
from tqdm import tqdm
import datetime as dt
import pickle as pkl

from absl import flags, app
import logging

from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
import re
import os, sys
import toml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Command-line flags that can override the values in the config file.
FLAGS = flags.FLAGS
flags.DEFINE_string('config', 'FredRalph_p1.toml', 'Path to the TOML config')
flags.DEFINE_string('ggml-model', None, 'Path to the GGML model')
flags.DEFINE_integer('rounds', None, 'Number of rounds to execute')
flags.DEFINE_string('output', None, 'Output file')


def load_toml(fname):
    
    with open(fname, 'r') as f:
        cfg = toml.load(f)
        
    return cfg


def get_next_speaker(prompt, speakers=['Frederich', 'Ralph']):
    sp_ord = {s:0 for s in speakers}
    tokens = TreebankWordTokenizer().tokenize(prompt)
    
    for ti in range(0,len(tokens)-1):
        for s in speakers:
            if tokens[ti] == s and tokens[ti+1] == ':':
                sp_ord[s] = ti
    
    next_speaker = min(sp_ord, key=sp_ord.get)
            
    return next_speaker


def get_new_prompt(output, n_keep=150, speakers=['Frederich','Ralph']):
    
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
    
    while total_len < n_keep and k < len(segments):
        
        segment = segments[k]
        N = len(TreebankWordTokenizer().tokenize(segment))

        total_len += N
        segments_to_keep.append(segment)
            
        k += 1
    
    segments_to_keep = segments_to_keep[::-1]
    
    ### construct the new prompt with the start prompt and the segments added 
    
    new_prompt = start_prompt
    
    for segment in segments_to_keep:
        
        if len(re.findall("[.!?]",segment)) != 0:
            new_prompt = new_prompt + '\n' + segment
        
    next_speaker = get_next_speaker(new_prompt, speakers)
    new_prompt += '\n' + next_speaker + ': ' 
    
    return new_prompt


def main(argv):
        
    cfg = load_toml(FLAGS.config)

    now = dt.datetime.now().replace(microsecond=0)
    formatted_date = now.strftime("%d%m%Y_%H%M%S")
    
    # Override the model path if specified.
    cfg['model_params']['ggml_model'] = FLAGS['ggml-model'].value or cfg['model_params']['ggml_model']
    rounds = FLAGS['rounds'].value or cfg['debate_params']['rounds']
    fname_out = FLAGS['output'].value or f'{FLAGS.config.value.strip(".toml")}_{formatted_date}.txt'

    start_prompt = cfg['debate_params']['initial_prompt']
    speaker1 = cfg['debate_params']['speaker1_fullname'].split(' ')[0]
    speaker2 = cfg['debate_params']['speaker2_fullname'].split(' ')[0]

    logger.info('Using model: %s', cfg['model_params']['ggml_model'])
    logger.info('Rounds: %d', rounds)
    logger.info('Output file: %s', fname_out)

    model = Model(**cfg['model_params'])

    all_outputs = []
    all_prompts = []

    prompt = start_prompt

    for n in tqdm(range(0,rounds)):

        output = model.generate(prompt, **{**cfg['gpt_params'],**{'n_threads':os.cpu_count()}})
        prompt = get_new_prompt(output, n_keep=int(cfg['model_params']['n_ctx']/2),speakers=[speaker1,speaker2])

        all_outputs.append(output)
        all_prompts.append(prompt)

        print(output)
        print('')
        print('###########################################')

    with open(fname_out, 'w') as f:
        f.write('\n'.join(all_outputs))
    
        
if __name__ == "__main__":
    app.run(main)