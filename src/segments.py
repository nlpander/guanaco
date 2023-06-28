import re
from nltk.tokenize import sent_tokenize
from transformers import LlamaTokenizer

def get_llama_tokenizer(path):
    return LlamaTokenizer.from_pretrained(path)

# Determine who the next speaker is based on the prompt.
# To do this, we determine the last mention of each speaker in the prompt and return the one that was mentioned furthest from the end.
def get_next_speaker(prompt, speakers=["Frederich", "Ralph"]):
    mentions = {s: prompt.rfind(f"{s}:") for s in speakers}
    return min(mentions, key=mentions.get)

# sometimes markdown is generated - remove this 
def remove_markdown(output_text):
    return re.sub(r'\\[a-z]+\{[^\}]*\}|\$[^\$]*\$|\\\w+', '', output_text)

# split the segment if a speaker section begins mid-sentence
def split_segment_speaker_midsentence(segment, speakers=["Frederich", "Ralph"]):
    for speaker in speakers:
        tmp = re.findall(f"\S({speaker}: )", segment)
        if len(tmp) != 0:
            subsegs = segment.split(tmp[0])
            output_segment = subsegs[0] + "\n" + speaker + ": " + subsegs[1]
            break
        else:
            output_segment = segment

    return output_segment

def get_first_speaker_segments(segments_to_keep, speakers = ['Ray', 'Warren']):

    speaker1_str_ = speakers[0] + ': '
    speaker2_str_ = speakers[1] + ': '

    first_speaker = ''

    k = 0 
    while first_speaker == '':
        spkrs = re.findall(f"({speaker1_str_})|({speaker2_str_})",segments_to_keep[k])
        if len(spkrs) != 0:
            for s in spkrs[0]:
                if s != '':
                    first_speaker = s
        k += 1

    if k != 0:

        if first_speaker == speaker2_str_:
            segments_to_keep[0] = speaker1_str_ + segments_to_keep[0]
        elif first_speaker == speaker1_str_:
            segments_to_keep[0] = speaker2_str_ + segments_to_keep[0]
    
    return segments_to_keep

# get new prompt
def get_new_prompt(prefix, output, conversation_list, tokenizer_path, n_keep=150, speakers=["Frederich", "Ralph"]):  
    
    tokenizer = get_llama_tokenizer(tokenizer_path)
    output = tokenizer.convert_tokens_to_string(list(output))

    ### remove any markdown 
    output = remove_markdown(output)

    start_prompt = prefix
    start_prompt += "\n"

    N0 = len(tokenizer.tokenize(start_prompt))

    ### split the rest of prompt sentence by sentences counting the overall size of the context
    ### add sentences from the end backwards
    ### till the n_keep limit has been reached. In llama.cpp this is set to n_ctx / 2

    total_len = N0

    ### reverse sentences
    segments = sent_tokenize(output)[::-1]
    segments_to_keep = []
    all_segments = []
    
    for i in range(0, len(segments)):
        segment = segments[i]
        N = len(tokenizer.tokenize(segment))
        total_len += N

        if total_len < n_keep and len(re.findall("[.!?]", segment)) != 0:
            segments_to_keep.append(segment)

        all_segments.append(segment)

    segments_to_keep = segments_to_keep[::-1]
    all_segments = all_segments[::-1]
    
    ### after triming the previous output for the new input context we need to add the first speaker 
    segments_to_keep = get_first_speaker_segments(segments_to_keep, speakers)
    
    ### construct the new prompt with the start prompt and the segments added
    new_prompt = start_prompt
    # conversation = ''
    
    for j in range(0, len(all_segments)):
        segment = all_segments[j]
        
        ### append full sentences to the conversation list and the new prompt to inject
        if len(re.findall("[.!?]", segment)) != 0:
            # if speaker contained in segment split the segment
            segment = split_segment_speaker_midsentence(segment, speakers)
            
            if segment not in conversation_list:
                conversation_list.append(segment)

            if j < len(segments_to_keep):
                new_prompt = new_prompt + "\n" + segments_to_keep[j]

    ### if there is no additional context get the new speaker and append them to the prompt
    next_speaker = get_next_speaker(new_prompt, speakers)
    new_prompt += "\n" + next_speaker + ": "

    return new_prompt, conversation_list

def strip_last_speaker_add_context(prompt, additional_ctx):
    segments = prompt.split("\n")
    final_segment = segments[-1]
    final_segment = re.sub("\w+: ", "", final_segment)

    if final_segment == "":
        final_segment = "Host: " + additional_ctx
    else:
        final_segment = final_segment + ". " + "\nHost: " + additional_ctx

    segments[-1] = final_segment
    new_prompt = "\n".join(segments)

    return new_prompt
