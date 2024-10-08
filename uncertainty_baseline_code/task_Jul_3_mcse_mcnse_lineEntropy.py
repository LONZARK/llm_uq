"""
The following code is related to task_Jul_3. 
Implementation of entropy calculation of each token in the output generated by LLM.

07-03-2024 @ Jia Li
"""



from typing import List, Optional, Tuple, Dict
import fire
from llama import Dialog, Llama
import torch
from datasets import load_dataset
import os
import pickle
import time
from llama.tokenizer import Tokenizer
import numpy as np
import re

def initialize_llama3(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    
    # Initialize the LLaMA model
    model = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    return model


def gen_resp_LLM(model, prompt: str, max_gen_len, top_p, temperature) -> Tuple[List[str], List[torch.Tensor]]:
    """
    This function is used to generate the response as a list of decoded tokens (not token ids).
    
    Args:
    model: the LLM model object
    prompt: a string representing the query
    
    Returns:
    Tuple[List[str], List[torch.Tensor]]: a list of decoded tokens, and a list of probability vectors representing the predictive distribution for the tokens
    """

    # Generate response
    dialog = [{"role": "user", "content": prompt}]
    response, token_logits_dict, prompt_tokens_id = model.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # print('token_logits_dict', token_logits_dict)

    print('dialog', dialog)
    print('response', response)

    decoded_tokens_id = []
    decoded_tokens = []
    cur_probs_value = []
    prob_vector = []
    for key, value in token_logits_dict.items():
        decoded_tokens_id.append(value['next_tokens_ids'])
        decoded_tokens.append(value['next_tokens'])
        cur_probs_value.append(torch.tensor(value['cur_probs_value']))
        prob_vector.append(value['prob_vector'])

    # tokenizer = Tokenizer(model_path=tokenizer_path)

    prompt_tokens = []
    for item in prompt_tokens_id[0]:
        prompt_tokens.append(model.tokenizer.decode([item]))

    return prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, cur_probs_value, prob_vector



def partition_into_sentences(decoded_tokens: List[str], decoded_tokens_id: List[List[int]], cur_probs_value: List[torch.Tensor]) -> List[Tuple[List[str], List[int], List[torch.Tensor]]]:
    """
    Partition the response into sentences.
    
    Args:
    decoded_tokens (List[str]): List of decoded tokens
    decoded_tokens_id (List[List[int]]): List of token IDs corresponding to decoded_tokens
    cur_probs_value (List[torch.Tensor]): List of probability vectors for each token

    Returns:
    List[Tuple[List[str], List[int], List[torch.Tensor]]]: List of sentences, where each sentence is a tuple containing
        (list of tokens, list of token IDs, list of probability vectors)
    """
    sentences = []
    current_sentence_tokens = []
    current_sentence_ids = []
    current_sentence_prob = None

    # Iterate over the tokens and their corresponding prob_vector
    for token, token_id, prob_vector in zip(decoded_tokens, decoded_tokens_id, cur_probs_value):
        # Check if the current token contains a newline character
        if '\n' in token:
            # Split the token at each newline character
            sub_tokens = token.split('\n')
            
            for i, sub_token in enumerate(sub_tokens):
                if sub_token:  # If the sub_token is not empty
                    current_sentence_tokens.append(sub_token)
                    current_sentence_ids.extend(token_id)
                    
                    # Update the sentence probability
                    if current_sentence_prob is None:
                        current_sentence_prob = torch.log(torch.tensor(prob_vector))
                    else:
                        current_sentence_prob = torch.logaddexp(current_sentence_prob, torch.log(torch.tensor(prob_vector)))

                # If it's not the last sub_token, end the current sentence
                if i < len(sub_tokens) - 1:
                    sentences.append((current_sentence_tokens, current_sentence_ids, current_sentence_prob))
                    current_sentence_tokens = []
                    current_sentence_ids = []
                    current_sentence_prob = None
        else:
            # If no newline, just add the token to the current sentence
            current_sentence_tokens.append(token)
            current_sentence_ids.extend(token_id)
            
            # Update the sentence probability
            if current_sentence_prob is None:
                current_sentence_prob = torch.log(torch.tensor(prob_vector))
            else:
                current_sentence_prob = torch.logaddexp(current_sentence_prob, torch.log(torch.tensor(prob_vector)))

    # Add any remaining tokens as the last sentence
    if current_sentence_tokens:
        sentences.append((current_sentence_tokens, current_sentence_ids, current_sentence_prob))
    return sentences





def get_samples_MC(model, prompt: List[str], prompt_id, primary_response, primary_response_id, sentence: List[str], n_samples: int) -> List[Dict]:
    """
    Get Monte Carlo samples and their probabilities for a given sentence.
    Perform Monte-Carlo sampling for a sentence in the primary response of the model given a prompt.

    Args:
    model: LLM model object
    prompt (List[str]): The input prompt, a list of tokens
    primary_response (List[str]): The primary response (response when temperature=0), a list of tokens
    sentence (List[str]): A sentence in the primary response, a list of tokens
    n_samples (int): Number of Monte Carlo samples to generate

    Returns:
    List[Dict]: A list of dictionaries, each containing a sample sentence and its sequence probability
    """
    # Find the index of the sentence in the primary response
    sentence_start_index = primary_response.index(sentence[0][0])
    
    # Construct the context by combining the prompt and the text up to the current sentence
    c = [item[0] for item in primary_response_id[:sentence_start_index]]
    context = [prompt_id[0]+c]
    samples = []
    
    for _ in range(n_samples):
        # Generate a new sample using the model with non-zero temperature
       
        generation_tokens, generation_logprobs, token_logits_dict = model.generate(
            prompt_tokens=context,
            max_gen_len=512,
            temperature= 0.99,
            # logprobs=logprobs,
        ) 

        decoded_tokens_id = []
        decoded_tokens = []
        cur_probs_value = []
        for key, value in token_logits_dict.items():
            decoded_tokens_id.append(value['next_tokens_ids'])
            decoded_tokens.append(value['next_tokens'])
            cur_probs_value.append(torch.tensor(value['cur_probs_value']))

        sentences = partition_into_sentences(decoded_tokens, decoded_tokens_id, cur_probs_value)
        samples.append(sentences[0])

    return samples


def calculate_mcse(sentences):
    """
    Calculates the Monte Carlo Sequence Entropy (MCSE) for a given set of sentences.

    Args:
    sentences (List[Tuple[List[str], List[int], torch.Tensor]]): A list of sentences, 
        where each sentence is a tuple containing tokens, token IDs, and log probability.

    Returns:
    float: The calculated MCSE
    """
    total_log_prob = 0
    total_sentence = 0

    for sentence in sentences:
        tokens, _, log_prob = sentence
        
        # Sum the negative log probabilities
        total_log_prob -= log_prob.sum().item()
        
        # Count the total number of tokens
        # total_tokens += len(tokens)
        total_sentence += 1

    # Calculate the average negative log probability
    mcse = total_log_prob / total_sentence

    return mcse


def cal_MCSE(model, prompt: List[str], n_samples: int) -> List[Dict]:
    """
    Calculate MCSE (Monte Carlo Sequence Entropy) uncertainty score given the model and prompt.
    
    Args:
    model: LLM model object
    prompt (List[str]): The input prompt, a list of tokens
    n_samples (int): The setting of sample size in Monte-Carlo sampling
    
    Returns:
    List[Dict]: A list of dictionaries containing MCSE scores for each sentence
    """
    # Step 1: Generate the primary response
    prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, cur_probs_value, prob_vector = gen_resp_LLM(model, prompt, max_gen_len = 256, top_p = 0, temperature = 0) 
    
    # Step 2: Partition the primary response into sentences
    sentences = partition_into_sentences(decoded_tokens, decoded_tokens_id, cur_probs_value)

    results = []

    # Step 3: Calculate MCSE for each sentence
    for sentence in sentences:
        # Get Monte Carlo samples and their probabilities
        if len(sentence[0]) == 0:
            pass
        else:
            mc_samples = get_samples_MC(model, prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, sentence,  n_samples =5)

            # Calculate MCSE
            mcse = calculate_mcse(mc_samples)
            
            # Store results
            results.append({
                "sentence": sentence,
                "MC_samples": mc_samples,
                "MCSE": mcse
            })

    return results



def calculate_mcnse(samples: List[Tuple[List[str], List[int], torch.Tensor]]) -> float:
    """
    Calculates the Monte Carlo Normalized Sequence Entropy (MCNSE) for a given set of samples.

    Args:
    samples (List[Tuple[List[str], List[int], torch.Tensor]]): A list of tuples, each containing
        a list of tokens, a list of token IDs, and a tensor of log probability

    Returns:
    float: The calculated Monte Carlo Normalized Sequence Entropy
    """
    mcnse_values = []

    for tokens, _, log_prob in samples:
        if len(tokens) == 0:
            continue  # Skip empty sequences

        # Convert log probability tensor to a Python float
        neg_log_prob = -log_prob.item()

        # Calculate the normalized negative log probability
        normalized_neg_log_prob = neg_log_prob / (len(tokens) )

        mcnse_values.append(normalized_neg_log_prob)

    # Return the average MCNSE across all samples
    return np.mean(mcnse_values)


def cal_MCNSE(model, prompt: List[str], n_samples: int) -> List[Dict]:
    """
    Calculate MCNSE (Monte Carlo Normalized Sequence Entropy) uncertainty score given the model and prompt.
    
    Args:
    model: LLM model object
    prompt (List[str]): The input prompt, a list of tokens
    n_samples (int): The setting of sample size in Monte-Carlo sampling
    
    Returns:
    List[Dict]: A list of dictionaries containing MCSE scores for each sentence
    """
    # Step 1: Generate the primary response
    prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, cur_probs_value, prob_vector = gen_resp_LLM(model, prompt, max_gen_len = 256, top_p = 0, temperature = 0) 
    
    # Step 2: Partition the primary response into sentences
    sentences = partition_into_sentences(decoded_tokens, decoded_tokens_id, cur_probs_value)

    results = []

    # Step 3: Calculate MCSE for each sentence
    for sentence in sentences:
        # Get Monte Carlo samples and their probabilities
        if len(sentence[0]) == 0:
            pass
        else:
            mc_samples = get_samples_MC(model, prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, sentence,  n_samples =5)

            # Calculate MCSE
            mcse = calculate_mcnse(mc_samples)
            
            # Store results
            results.append({
                "sentence": sentence,
                "MC_samples": mc_samples,
                "MCSE": mcse
            })

    return results



def calculate_entropy(prob_vec: torch.Tensor) -> float:
    """
    This function calculates the entropy from a probability vector that represents a categorical distribution.
    
    The input probability vector should be normalized to be valid. For example, if the input probability vector 
    representing predicted probability of Top-3 choices is [0.5, 0.1, 0.2], it should be normalized to [0.625, 0.125, 0.25].
    
    Args:
    prob_vec: a probability vector or dictionary that stores the predicted probabilities of some next token
    
    Returns:
    List[float]: a list of scalar values representing the entropy for each probability vector.
    """
    entropy_list = []

    for prob in prob_vec:
        # Ensure the probabilities are normalized
        prob = prob / prob.sum(dim=-1, keepdim=True)
        # Calculate entropy
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=-1)  # Adding a small value to avoid log(0)
        entropy_list.append(entropy.item())

    return entropy_list


def calculate_linewise_entropy_from_tokens(model, prompt: str, max_gen_len, top_p, temperature):
    """
    This function generates the primary response, as well as the line-wise entropy for each token in the response.
    The line-wise entropy for a line is calculated as the average token-wise entropy for all the tokens in it after
    all tokens and entropies have been collected for that line.

    Args:
    model: the LLM model object
    prompt: a string representing the querys
    temperature: the parameter that controls the randomness in the response
    top_p: the parameter for nucleus sampling
    max_gen_len: maximum sequence length for the output
    
    Returns:
        List[Tuple[str, float]]: A list of tuples, each containing a line of text as a string and the average entropy 
        of that line as a float.
    """
    prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, cur_probs_value, prob_vector = gen_resp_LLM(model, prompt, max_gen_len = 256, top_p = 0, temperature = 0)
    token_entropy_list = calculate_entropy(prob_vector)
    assert len(decoded_tokens) == len(token_entropy_list), "The lengths of decoded_tokens and entropy_list do not match"

    # Initialize empty lists to store lines and their corresponding lists of entropies
    lines = []
    entropies_per_line = []
    # Initialize temporary lists to accumulate tokens and entropies for the current line
    current_line = []
    current_entropies = []

    # Iterate over the tokens and their corresponding entropies
    for token, entropy in zip(decoded_tokens, token_entropy_list):
        # Check if the current token contains a newline character
        if '\n' in token:
            # Split the token at each newline character
            parts = token.split('\n')

            # Iterate over each part of the token split by newline
            for i, part in enumerate(parts):
                if i < len(parts) - 1:
                    # Add to the current line
                    current_line.append(part)
                    current_entropies.append(entropy)
                    # Combine and store the current line and its list of entropies, then reset for a new line
                    lines.append(''.join(current_line))
                    entropies_per_line.append(current_entropies)
                    current_line = []
                    current_entropies = []
                else:
                    # Continue adding to the current line
                    current_line.append(part)
                    current_entropies.append(entropy)
        else:
            # If no newline in token, add token and entropy to the current line
            current_line.append(token)
            current_entropies.append(entropy)

    # After the loop, if there's any remaining data in the current line, add it to the final lists
    if current_line:
        lines.append(''.join(current_line))
        entropies_per_line.append(current_entropies)

    line_entropies = [sum(ents) / len(ents) if ents else 0 for ents in entropies_per_line]
    result = list(zip(lines, line_entropies))
    return result




def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0,
    top_p: float = 0,
    max_seq_len: int = 512,
    max_batch_size: int = 22,
    max_gen_len: Optional[int] = 256,
):
    
    model = initialize_llama3(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len)
    prompt = """ Who is Donald Trump? Answer with each sentence in a new line. """

    # cal_MCNSE_results = cal_MCNSE(model, prompt, 3)
    # print('cal_MCNSE results', cal_MCNSE_results)

    # cal_MCSE_results = cal_MCSE(model, prompt, 3)
    # print('cal_MCSE results', cal_MCSE_results)


    line_entropies = calculate_linewise_entropy_from_tokens(model, prompt, max_gen_len, top_p, temperature)
    print('line_entropies', line_entropies)

    # prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, cur_probs_value, prob_vector = gen_resp_LLM(model, prompt, max_gen_len, top_p, temperature) 

    # # # print('============ prompt_tokens =============')    
    # # # print('prompt_tokens', prompt_tokens)    
    # # # print('============ prompt_tokens_id =============')    
    # # # print('prompt_tokens_id', prompt_tokens_id)    
    # print('============ decoded_tokens ============')    
    # print('decoded_tokens', decoded_tokens)   
    # print('============ decoded_tokens_id =============')    
    # print('decoded_tokens_id', decoded_tokens_id)    
    # # # print('============ cur_probs_value =============')    
    # # # print('cur_probs_value', cur_probs_value)    

    # sentences = partition_into_sentences(decoded_tokens, decoded_tokens_id, cur_probs_value)
    # # print('============ sentences =============')    
    # # print('sentences', sentences)    

    # for sentence in sentences:
    #     # Get Monte Carlo samples and their probabilities
    #     if len(sentence[0]) == 0:
    #         pass
    #     else:
    #         samples = get_samples_MC(model, prompt_tokens, prompt_tokens_id, decoded_tokens, decoded_tokens_id, sentence,  n_samples =5)
    #         print('============ samples =============')    
    #         print('samples', samples)
            
    #         mcse = calculate_mcse(samples)
    #         print('mcse', mcse)

    #         mcnse = calculate_mcnse(samples)
    #         print('mcnse', mcnse)





if __name__ == "__main__":
    fire.Fire(main)

# conda activate llmuq; cd /people/cs/j/jxl220096/llmuq/llama3/; torchrun --nproc_per_node 1 task_Jul_3.py --ckpt_dir Meta-Llama-3-8B-Instruct/ --tokenizer_path Meta-Llama-3-8B-Instruct/tokenizer.model --max_seq_len 512 --max_batch_size 6

