import re
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer


# Determine who the next speaker is based on the prompt.
# To do this, we determine the last mention of each speaker in the prompt and return the one that was mentioned furthest from the end.
def get_next_speaker(prompt, speakers=["Frederich", "Ralph"]):
    mentions = {s: prompt.rfind(f"{s}:") for s in speakers}
    return min(mentions, key=mentions.get)


# get new prompt
def get_new_prompt(
    output, conversation_list, n_keep=150, speakers=["Frederich", "Ralph"]
):
    output = "".join(list(output))
    ### keep start of prompt
    start_prompt = "\n".join(output.split("\n")[0:2])

    ### sometimes we get a leading whitespace in the string - remove this
    start_prompt = re.sub("^\s*", "", start_prompt)
    start_prompt += "\n"

    N0 = len(TreebankWordTokenizer().tokenize(start_prompt))

    ### split the rest of prompt sentence by sentences counting the overall size of the context
    ### add sentences from the end backwards
    ### till the n_keep limit has been reached. In llama.cpp this is set to n_ctx / 2

    total_len = N0
    k = 0

    next_output = "\n".join(output.split("\n")[2:])

    segments = sent_tokenize(next_output)[::-1]
    segments_to_keep = []
    all_segments = []

    for i in range(0, len(segments)):
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
    # conversation = ''

    for j in range(0, len(all_segments)):
        segment = all_segments[j].replace("\n", "")

        ### append full sentences to the conversation list and the new prompt to inject
        if len(re.findall("[.!?]", segment)) != 0:
            if segment not in conversation_list:
                conversation_list.append(segment)

            if j < len(segments_to_keep):
                new_prompt = new_prompt + "\n" + segments_to_keep[j]

    ### if there is no additional context get the new speaker and append them to the prompt - from llama segments
    # next_speaker = get_next_speaker(new_prompt, speakers)
    # new_prompt += "\n" + next_speaker + ": "

    return new_prompt, conversation_list


def strip_last_speaker_add_context(prompt, additional_ctx):
    segments = prompt.split("\n")
    final_segment = segments[-1]
    final_segment = re.sub("\w+: ", "", final_segment)

    if final_segment == "":
        final_segment = "### Human: " + additional_ctx
    else:
        final_segment = final_segment + ". " + "\n### Human: " + additional_ctx

    segments[-1] = final_segment
    new_prompt = "\n".join(segments)

    return new_prompt
