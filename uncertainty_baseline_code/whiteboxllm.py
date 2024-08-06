import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.generation import is_chat
import time
import pdb
import numpy as np

def set_device(device='cuda:0'):
    # Check if the specified device is a GPU and parse its index
    if device.startswith('cuda'):
        # Try to parse out the GPU index after 'cuda:'
        gpu_index = device.split(':')[-1]
        try:
            gpu_index = int(gpu_index)
            if gpu_index >= torch.cuda.device_count() or not torch.cuda.is_available():
                raise ValueError("GPU index out of range or CUDA is not available.")
        except ValueError as e:
            print(f"Specified device '{device}' is not available. Error: {e}")
            device = 'cpu'
    elif device not in ['cpu']:
        print(f"Invalid device '{device}' specified. Falling back to 'cpu'.")
        device = 'cpu'
    return device

def print_gpu_memory_usage():
    mem_allocated = torch.cuda.memory_allocated()/1024**3
    mem_reserved = torch.cuda.memory_reserved()/1024**3
    print("Allocated memory:", mem_allocated, "GB")
    print("Cached memory:", mem_reserved, "GB")
    # return mem_allocated, mem_reserved
    
class WhiteBoxLLM:
    VALID_MODEL_TYPES = ["AutoModelForCausalLM"]
    VALID_UQ_METHODS = ["MSP", "MTE", "PP" , "P(True)", "MCSE"]# "MCNSE", "PMI", "CPMI"]

    def __init__(self, model_type, use_multi_gpu=False):
        if model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(f"Invalid model type. Choose from {self.VALID_MODEL_TYPES}")
        self.use_multi_gpu = use_multi_gpu
        
        self.model_type = model_type
        self.model = None
        self.pipeline=None
        self.tokenizer = None
        self.device = None
        self.dtype = torch.bfloat16
        self.max_line_lenth = 30  # The number of tokens to generate in line-level pTrue, MC sampling

    def load_pretrained(self, model_path, device='cpu', dtype=torch.bfloat16):
        """
        Load a pretrained model and tokenizer.

        Args:
        model_path (str): Path to the pretrained model.
        device (str): Device to load the model on ('cpu' or 'cuda').
        dtype (torch.dtype): Data type to use for the model.
        """
        print(f"loading pretrained model from: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.dtype = dtype
        self.device = set_device(device=device)
        
        if self.use_multi_gpu:
            ### use multiple GPUs with transformers.pipeline
            start = time.time()
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": self.dtype},
                device_map="auto"
            )
            self.tokenizer = self.pipeline.tokenizer
            self.model = self.pipeline.model
            end = time.time()
            print(f"Pipeline constructed in {str(end-start)} seconds. Model is loaded from path:\n {model_path}.")
        else:
            ### load model on device
            start = time.time()
            self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device, dtype=self.dtype)    
            print(f"loading {type(self.model)}")
            end = time.time()
            print(f"Model loaded from path: {model_path}.")
            print(f"Time: {str(end-start)} seconds")  

        
    def tokenize_input(self, prompt, mode="text", add_special_tokens=True, tokenize=True):
        """
        Tokenize the input prompt.

        Args:
        prompt (str or list): The input prompt.
        mode (str): Mode of tokenization, either 'text' or 'chat'.
        add_special_tokens (bool): Whether to add special tokens.
        tokenize (bool): If True, returns tokenized input; if False, returns token values.

        Returns:
        dict or list: Tokenized input if `tokenize` is True, otherwise list of token values.
        """
        if mode not in ["text", "chat"]:
            raise ValueError("Mode must be either 'text' or 'chat'")

        if mode == "text":
            if not isinstance(prompt, str):
                raise ValueError("Prompt must be a string for text mode.")
            
        elif mode == "chat":
            ### generation prompt will be added
            if not isinstance(prompt, list) or not all(isinstance(msg, dict) and 'role' in msg and 'content' in msg for msg in prompt):
                raise ValueError("Prompt must be a list of dictionaries with 'role' and 'content' keys for chat mode.")
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        encoding = self.tokenizer(
            prompt,
            add_special_tokens=add_special_tokens,
            return_tensors='pt' if tokenize else None
        )

        if tokenize:
            return encoding  # Returns tokenized input
        else:
            return self.tokenizer.convert_ids_to_tokens(encoding.input_ids[0])  # Returns list of token values

    def generate(self, input, max_new_tokens=512, temperature=1.0, top_p=None, top_k=None, do_sample=False, add_special_tokens = True, return_type='raw'):
        """
        Generate text using the loaded model.

        Args:
            input (dict, list, or str): Tokenized input containing input_ids and attention_mask(optional), or a chat object, or plain text.
            max_new_tokens (int): Maximum number of new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling.
            top_k (int): Top-k sampling.
            do_sample (bool): Whether to use sampling; use greedy decoding otherwise.
            return_in_dict (bool): If True, return detailed information in a dictionary.
            
        Returns:
            str or dict: raw output of model.generate() or a dictionary with detailed results if return_in_dict is True.
        """
        VALID_RETURN_TYPES = ['generation_text', 'generation_dict', 'pred_logprobs', 'raw']
        
        if return_type not in VALID_RETURN_TYPES:
            print(f"Invalid return type '{return_type}' specified. Valid return types: {str(VALID_RETURN_TYPES)}. Falling back to 'generation_text'.")
            return_type = 'generation_text'

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        terminators = self.tokenizer.eos_token_id
        
        
        # Handling different input types

        if isinstance(input, list):  # Assuming input is a chat object
            if not is_chat(input):
                raise ValueError("The list input must be a valid chat format.")
            tokenized_input = self.tokenize_input(input, mode='chat', add_special_tokens=add_special_tokens, tokenize=True)
        elif isinstance(input, str):  # Input is a plain text string
            tokenized_input = self.tokenize_input(input, mode='text', add_special_tokens=add_special_tokens, tokenize=True)
        elif 'input_ids' in input:  # Input contains input ids for the tokens in the input sequence, i.e.,input has been tokenized
            tokenized_input = input
        else:
            raise ValueError("Input format not supported. It must be either a dict, list (chat), or string (text).") ####

        # set attention masks
        if 'attention_mask' not in input:
            tokenized_input['attention_mask'] = tokenized_input['input_ids'].ne(self.tokenizer.pad_token_id).long()

        input_ids = tokenized_input['input_ids'].to(self.model.device)
        attention_mask = tokenized_input['attention_mask'].to(self.model.device)

        # Model inference
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        if return_type=='raw':
            return output
        
        if return_type=='pred_logprobs':
            ### return the predicted log-probs of new tokens generated in a tuple
            ### apply exponential to get the predicted probability vector for each newly generated token
            ### further apply max on predicted probability vector to get the predicted probability of greedily selected token
            pred_logprobs = tuple(torch.nn.functional.log_softmax(score, dim=1) for score in output.scores)
            return pred_logprobs
        

        input_len = input_ids.size(1)
        
        output_ids = output.sequences[0]  
        new_ids = output_ids[input_len:]
        
        new_text = self.tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        if return_type=='generation_text':
            ### return the generated text
            return new_text
        
        
        input_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        output_tokens = self.tokenizer.convert_ids_to_tokens(output_ids)
        new_tokens = self.tokenizer.convert_ids_to_tokens(new_ids)
           
        if return_type=='generation_dict':
            ### return the generation dict with rich information
            return {
                "paras": {
                    "encoding": {"add_special_tokens": True},
                    "generation": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "do_sample": do_sample
                    },
                    "decoding": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
                },
                "input_ids": input_ids[0].tolist(),
                "new_ids": new_ids.tolist(),
                "output_ids": output_ids.tolist(),
                "input_tokens": input_tokens,
                "new_tokens": new_tokens,
                "output_tokens": output_tokens,
                "input_text": input_text,
                "new_text": new_text,
                "output_text": output_text,
                "tokenized_input": {
                    "input_ids": input_ids[0].tolist(),
                    "attention_mask": attention_mask[0].tolist()
                }
            }
        
        return None
        




    def batch_generate():
        pass

    def generate_uncertainty_scores(self, generation_dict, uq_methods=None, debug=False):
        """
        Generate uncertainty scores with the specified uncertainty quantification methods.

        Args:
        generation_dict (dict): Dictionary containing generation information.
        uq_methods (list): List of strings containing the names for the methods.

        Returns:
        dict: Uncertainty dictionary with generated uncertainty scores.
        """
        if uq_methods is None:
            uq_methods = self.VALID_UQ_METHODS

        # Check for invalid methods and remove them from uq_methods
        invalid_methods = [method for method in uq_methods if method not in self.VALID_UQ_METHODS]

        if invalid_methods:
            print(f"Invalid UQ methods: {invalid_methods}. They will be ignored.")
            uq_methods = [method for method in uq_methods if method in self.VALID_UQ_METHODS]
        print(f"Valid UQ methods to execute: {uq_methods}")

        # Create a copy of the generation_dict to uncertainty_dict
        uncertainty_dict = generation_dict.copy()

        input_ids = torch.tensor([generation_dict['input_ids']])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        input_dict = dict(input_ids=input_ids, attention_mask=attention_mask)
        
        pred_logprobs = self.generate(input_dict, max_new_tokens=512, return_type='pred_logprobs')

        # Iterate over uq_methods and call the corresponding method
        for method in uq_methods:
            start_time = time.time()
            start_time = time.perf_counter()
            # Call the method (dummy implementation)
            getattr(self, method.lower().replace('(', '').replace(')', ''))(uncertainty_dict, pred_logprobs)

            temp_time = time.time()
            temp_time = time.perf_counter()
            if debug == True:
                print('========', method)
                # print('uncertainty_dict', uncertainty_dict)
                print('running time', str(temp_time - start_time))

        return uncertainty_dict


    def split_lines(self, token_ids):
        token_split = []
        id_split = []
        position_split = []
        cur_token_split = []
        cur_id_split = []
        cur_position = 0

        tokens = [self.tokenizer.decode(id) for id in token_ids]

        for token, token_id in zip(tokens, token_ids):
            cur_token_split.append(token)
            cur_id_split.append(token_id)
            if '\n' in token:
                token_split.append(cur_token_split)
                id_split.append(cur_id_split)
                position_split.append(cur_position)
                cur_token_split = []
                cur_id_split = []
                cur_position = len(token_split[-1]) + cur_position

        # Append any remaining tokens that did not end with a newline
        if cur_token_split:
            token_split.append(cur_token_split)
            id_split.append(cur_id_split)
            position_split.append(cur_position)

        # Create list of lines by joining tokens
        line_split = [' '.join(split) for split in token_split]

        return line_split, token_split, id_split, position_split

    def token_indices_by_line(self, token_ids):
        line_split, token_split,_,_ = self.split_lines(token_ids)
        token_indices_split =[]
        for i, split in enumerate(token_split):
            # print(token_split[i])
            start_index = sum(len(token_split[j]) for j in range(i))
            token_indices = [k for k in range(start_index, start_index + len(split))]
            token_indices_split.append(token_indices)
        return token_indices_split



    def mte(self, uncertainty_dict, pred_logprobs):

        new_ids = uncertainty_dict['new_ids']
        new_text = uncertainty_dict['new_text']

        # Compute the entropy for each tensor in the tuple
        entropy_list = []
        for log_prob_tensor in pred_logprobs:
            # Convert log probabilities to probabilities
            probabilities = torch.exp(log_prob_tensor)
            # Compute entropy using the probabilities and their log probabilities
            entropy = -(probabilities * log_prob_tensor).sum(dim=1)
            entropy_list.append(entropy)    

        
        all_entropies = torch.cat(entropy_list)
        global_mte = all_entropies.mean().item()

        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'MTE': global_mte})

        # Split the text into lines
        line_split, token_split,_,_ = self.split_lines(new_ids)
        token_indices_split = self.token_indices_by_line(new_ids)

        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []
        
        for i,line_indices in enumerate(token_indices_split):
        # for line_indices in token_indices_split:
            if not line_indices:  # Skip empty lines
                continue
            # line_tokens = [new_text[i] for i in line_indices]
            # line_text = ''.join(line_tokens)
            line_text = line_split[i]
            token_entropies_line = []

            for index in line_indices:
                token_entropies_line.append(all_entropies[index])
            # mte_score = np.mean(token_entropies_line)
            # Convert list to tensor
            token_entropies_line_tensor = torch.tensor(token_entropies_line, device='cuda')
            mte_score = torch.mean(token_entropies_line_tensor).tolist()

            new_scores = {"MTE": mte_score}
            
            # Update or append line scores
            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))
        
        return uncertainty_dict


    def msp(self, uncertainty_dict, pred_logprobs):

        new_ids = uncertainty_dict['new_ids']
        new_text = uncertainty_dict['new_text']

        # Compute the probabilities for each tensor in the tuple
        prob_list = []
        for i, log_prob_tensor in enumerate(pred_logprobs):
            # Convert log probabilities to probabilities
            probabilities = torch.exp(log_prob_tensor)
            index = new_ids[i]
            prob_list.append(probabilities[0, index].item())    

        # Global MSP score
        total_prob = 1.0
        for prob in prob_list:
            total_prob *= prob
        global_msp = 1 - total_prob
        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'MSP': global_msp})

        # Linewise MSP scores
        # Split the text into lines
        line_split, token_split,_,_ = self.split_lines(new_ids)
        token_indices_split = self.token_indices_by_line(new_ids)

        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []
        
        for i,line_indices in enumerate(token_indices_split):
            if not line_indices:  # Skip empty lines
                continue

            line_text = line_split[i]
            line_prob = 1.0
            for index in line_indices:
                line_prob *= prob_list[index]
            line_msp = 1 - line_prob
            new_scores = {"MSP": line_msp}

            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))

        return uncertainty_dict

    def pp(self, uncertainty_dict, pred_logprobs):

        new_ids = uncertainty_dict['new_ids']
        new_text = uncertainty_dict['new_text']

        log_prob_list = []
        for i, log_prob_tensor in enumerate(pred_logprobs):
            index = new_ids[i]
            log_prob_list.append(pred_logprobs[i][0, index].item())

        # Global PP score
        avg_log_prob = sum(log_prob_list) / len(log_prob_list)
        global_pp = 2 ** (-avg_log_prob)

        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'PP': global_pp})

        # Linewise PP scores
        # Split the text into lines
        line_split, token_split,_,_ = self.split_lines(new_ids)
        token_indices_split = self.token_indices_by_line(new_ids)

        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []
        
        for i,line_indices in enumerate(token_indices_split):
            if not line_indices:  # Skip empty lines
                continue
            line_text = line_split[i]
            line_logprobs = [log_prob_list[i] for i in line_indices]
            avg_line_log_prob = sum(line_logprobs) / len(line_logprobs)
            line_pp =  2 ** (-avg_line_log_prob)
            new_scores = {"PP": line_pp}

            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))

        return uncertainty_dict


    def generate_pTrue_prompt(self, base_prompt: str):
        system_message = """Answer the following question. Answer True or False without explanation:\n"""
        user_message = f"{base_prompt}"

        chat = [
            {"role":"system", "content":system_message},
            {"role":"user", "content":user_message}
            ]
        return chat


    def extract_ptrue_score(self, generation_dict, pred_logprobs):
        # Find the token for True or False
        ptrue = 0.0
        found_token = False
        for index, (token_id) in enumerate(generation_dict['new_ids']):
            token = self.tokenizer.decode(token_id)
            if 'True' in token:
                ptrue = pred_logprobs[index][0, token_id].item()
                found_token = True
                break
            elif 'False' in token:
                true_token_id = self.tokenizer('True')['input_ids'][0]
                ptrue = pred_logprobs[index][0, true_token_id].item()
                found_token = True
                break
        
        # If no True or False token is found, use the first token's probability to be True
        if not found_token:
            true_token_id = self.tokenizer('True')['input_ids'][0]
            ptrue = pred_logprobs[0][0, true_token_id].item()
        return 1 - ptrue

    def ptrue(self, uncertainty_dict, pred_logprobs):

        input_text = uncertainty_dict['input_text']
        new_text = uncertainty_dict['new_text']
        new_ids = uncertainty_dict['new_ids']

        verification_prompt = f"Question: {input_text}\nProposed Answer: {new_text}\nIs the content in the proposed answer correct?\n(A) True\n(B) False\nAnswer True or False without explanation:"
        verification_prompt = self.generate_pTrue_prompt(verification_prompt)

        global_ptrue_generation_dict = self.generate(verification_prompt, max_new_tokens=512, return_type='generation_dict')
        global_ptrue_pred_logprobs = self.generate(verification_prompt, max_new_tokens=512, return_type='pred_logprobs')

        global_ptrue = self.extract_ptrue_score(global_ptrue_generation_dict, global_ptrue_pred_logprobs)
        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'pTrue': global_ptrue})

        # Linewise ptrue scores
        # Split the text into lines
        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []

        line_split, token_split,_,_ = self.split_lines(new_ids)

        token_indices_split = self.token_indices_by_line(new_ids)
        for i,line_indices in enumerate(token_indices_split):
            if not line_indices:  # Skip empty lines
                continue
            line_text = line_split[i]

            line_verif_prompt = f"Question: {input_text}\nProposed Answer: {new_text}\nIs the sentence \"{line_text}\" in the proposed answer correct?\n(A) True\n(B) False\nAnswer True or False without explanation:"
            line_verif_prompt = self.generate_pTrue_prompt(line_verif_prompt)

            line_ptrue_generation_dict = self.generate(line_verif_prompt, max_new_tokens=self.max_line_lenth, return_type='generation_dict')
            line_ptrue_pred_logprobs = self.generate(line_verif_prompt, max_new_tokens=self.max_line_lenth, return_type='pred_logprobs')

            line_ptrue = self.extract_ptrue_score(line_ptrue_generation_dict, line_ptrue_pred_logprobs)
            new_scores = {"pTrue": line_ptrue}

            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))

        return uncertainty_dict

    def generate_mcse_prompt(self, input_text, line_responce):
        system_message = """Generate a response to the following input:\n"""
        base_prompt = input_text + '\n' + line_responce
        user_message = f"{base_prompt}"

        chat = [
            {"role":"system", "content":system_message},
            {"role":"user", "content":user_message}
            ]
        return chat

    def mcse(self, uncertainty_dict, pred_logprobs):
        
        input_text = uncertainty_dict['input_text']
        new_ids = uncertainty_dict['new_ids']
        input_ids = torch.tensor([uncertainty_dict['input_ids']])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        input_dict = dict(input_ids=input_ids, attention_mask=attention_mask)

        num_samples = 3
        total_logprob = 0
        for _ in range(num_samples):
            # pred_logprobs = self.generate(input_text, max_new_tokens=512, return_type='pred_logprobs')
            pred_logprobs = self.generate(input_dict, max_new_tokens=512, return_type='pred_logprobs')

            log_prob_list = []
            for i, log_prob_tensor in enumerate(pred_logprobs):
                index = new_ids[i]
                log_prob_list.append(pred_logprobs[i][0, index].item())
            
            temp_logprob = sum(log_prob_list)
            total_logprob += temp_logprob
        
        global_mcse = -total_logprob / num_samples
            
        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'MCSE': global_mcse})

        # Linewise ptrue scores
        # Split the text into lines
        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []
        line_split, _,_,_ = self.split_lines(new_ids)

        line_responce = ''
        token_indices_split = self.token_indices_by_line(new_ids)
        for i,line_indices in enumerate(token_indices_split):
            if not line_indices:  # Skip empty lines
                continue
            line_text = line_split[i]
            temp_mcse_prompt = self.generate_mcse_prompt(input_text, line_responce)
            line_total_logprob = 0
            for _ in range(num_samples):
                
                temp_generation_dict = self.generate(temp_mcse_prompt, max_new_tokens=self.max_line_lenth, return_type='generation_dict')
                temp_pred_logprobs = self.generate(temp_mcse_prompt, max_new_tokens=self.max_line_lenth, return_type='pred_logprobs')
            
                temp_new_ids = temp_generation_dict['new_ids']

                temp_token_indices_split = self.token_indices_by_line(temp_new_ids)

                temp_log_prob_list = []
                for i, log_prob_tensor in enumerate(temp_pred_logprobs):
                    index = temp_new_ids[i]
                    temp_log_prob_list.append(temp_pred_logprobs[i][0, index].item())

                # we only need the first sentence in the 'new_text'
                line_logprobs = sum([temp_log_prob_list[i] for i in temp_token_indices_split[0]])
                line_total_logprob += line_logprobs
            
            line_mcse = -line_total_logprob / num_samples
            new_scores = {"MCSE": line_mcse}

            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)

            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))
            line_responce += line_text
        return uncertainty_dict


    def mcnse(self, uncertainty_dict, pred_logprobs):
        
        input_text = uncertainty_dict['input_text']
        new_ids = uncertainty_dict['new_ids']
        input_ids = torch.tensor([uncertainty_dict['input_ids']])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        input_dict = dict(input_ids=input_ids, attention_mask=attention_mask)

        num_samples = 3

        total_normalized_logprob = 0
        for _ in range(num_samples):
            # Generate predicted log probabilities for each token in the sequence
            pred_logprobs = self.generate(input_dict, max_new_tokens=512, return_type='pred_logprobs')

            log_prob_list = []
            sequence_length = 0  # To track the length of each generated sequence

            # Calculate log probabilities for each generated token
            for i, log_prob_tensor in enumerate(pred_logprobs):
                index = new_ids[i]  # Index of the current token in the model's vocabulary
                log_prob_list.append(log_prob_tensor[0, index].item())
                sequence_length += 1  # Increment the sequence length for each token processed

            # Sum the log probabilities for the current sequence
            temp_logprob = sum(log_prob_list)

            # Normalize the summed log probabilities by the sequence length
            if sequence_length > 0:  # Avoid division by zero
                normalized_logprob = temp_logprob / sequence_length
                total_normalized_logprob += normalized_logprob

        # Compute the Monte Carlo Normalized Sequence Entropy
        global_mcnse = -total_normalized_logprob / num_samples
            
        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'MCNSE': global_mcnse})

        # Linewise ptrue scores
        # Split the text into lines
        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []

        line_split, _,_,_ = self.split_lines(new_ids)

        line_responce = ''
        token_indices_split = self.token_indices_by_line(new_ids)
        for i,line_indices in enumerate(token_indices_split):
            if not line_indices:  # Skip empty lines
                continue
            line_text = line_split[i]

            print('line_text', line_text)

            temp_mcse_prompt = self.generate_mcse_prompt(input_text, line_responce)

            line_total_normalized_logprob = 0
            for _ in range(num_samples):
                # Generate a generation dictionary and corresponding log probabilities
                temp_generation_dict = self.generate(temp_mcse_prompt, max_new_tokens=self.max_line_lenth, return_type='generation_dict')
                temp_pred_logprobs = self.generate(temp_mcse_prompt, max_new_tokens=self.max_line_lenth, return_type='pred_logprobs')
                
                # Extract new token IDs from the generation dictionary
                temp_new_ids = temp_generation_dict['new_ids']

                # Split the token IDs by line to process each line individually
                temp_token_indices_split = self.token_indices_by_line(temp_new_ids)

                # Initialize a list to store log probabilities
                temp_log_prob_list = []
                for i, log_prob_tensor in enumerate(temp_pred_logprobs):
                    index = temp_new_ids[i]
                    temp_log_prob_list.append(temp_pred_logprobs[i][0, index].item())

                # Calculate log probabilities for the first sentence in the 'new_text'
                line_logprobs = sum([temp_log_prob_list[i] for i in temp_token_indices_split[0]])
                line_length = len(temp_token_indices_split[0])  # Length of the first line

                # Normalize the log probabilities by the line length if the line length is not zero
                if line_length > 0:
                    normalized_line_logprobs = line_logprobs / line_length
                    line_total_normalized_logprob += normalized_line_logprobs

            # Compute the Monte Carlo Normalized Sequence Entropy for the line
            line_mcnse = -line_total_normalized_logprob / num_samples
            new_scores = {"MCNSE": line_mcnse}

            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))

            line_responce += line_text


        return uncertainty_dict

    def pmi(self, uncertainty_dict, pred_logprobs):
        input_text = uncertainty_dict['input_text']  # The initial prompt
        new_ids = uncertainty_dict['new_ids']  # List of token IDs for the generated response
        new_text = uncertainty_dict['new_text']  # The generated text

        # Calculate log probabilities for each generated token
        log_prob_list = []
        for i, log_prob_tensor in enumerate(pred_logprobs):
            index = new_ids[i]  # Index of the current token in the model's vocabulary
            log_prob_list.append(log_prob_tensor[0, index].item())

        # Calculate log probabilities for each generated token without x.
        # Generate len(new_ids) times
        log_prob_list_nox = []
        temp_prompt = '\n'
        for i in range(len(new_ids)):
            print('temp_prompt', temp_prompt)
            temp_generation_dict = self.generate(temp_prompt, max_new_tokens=512, return_type='generation_dict')
            temp_pred_logprobs = self.generate(temp_prompt, max_new_tokens=512, return_type='pred_logprobs')
            temp_new_ids = temp_generation_dict['new_ids']  # List of token IDs for the generated response
            index = temp_new_ids[0]
            log_prob_list_nox.append(temp_pred_logprobs[0][0,index].item())
            temp_prompt = self.tokenizer.decode(new_ids[: i])

        global_pmi_list = [a - b for a, b in zip(log_prob_list_nox, log_prob_list)]
        global_pmi = sum(global_pmi_list)/len(global_pmi_list)

        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'PMI': global_pmi})

        # Linewise ptrue scores
        # Split the text into lines
        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []
        line_split, _,_,_ = self.split_lines(new_ids)
        token_indices_split = self.token_indices_by_line(new_ids)

        for i, line_indices in enumerate(token_indices_split):
            if not line_indices:
                continue
            line_text = line_split[i]

            line_logprobs = [log_prob_list[i] for i in line_indices]
            line_logprobs_nox = []
            temp_line_prompt = '\n'
            for j in line_indices:
                print('temp_line_prompt', temp_line_prompt)
                temp_line_generation_dict = self.generate(temp_line_prompt, max_new_tokens=1, return_type='generation_dict')
                temp_line_pred_logprobs = self.generate(temp_line_prompt, max_new_tokens=1, return_type='pred_logprobs')
                temp_line_new_ids = temp_line_generation_dict['new_ids']  # List of token IDs for the generated response
                line_index = temp_line_new_ids[0]
                line_logprobs_nox.append(temp_line_pred_logprobs[0][0,line_index].item())

                temp_line_prompt = self.tokenizer.decode(new_ids[: j])

            line_pmi_list = [a - b for a, b in zip(line_logprobs_nox, line_logprobs)]
            line_pmi = sum(line_pmi_list)/len(line_pmi_list)

            new_scores = {"PMI": line_pmi}

            # # Check if the key exists in the dictionary
            # if 'line_unc_scores' in uncertainty_dict:
            #     # If the key exists, proceed to find the item
            #     existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            # else:
            #     # If the key does not exist, handle the error (e.g., by logging or raising an exception)
            #     print(f"Key 'line_unc_scores' not found in uncertainty_dict. Available keys: {list(uncertainty_dict.keys())}")
            #     existing_line = None  # Or handle the error as appropriate

            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))

        return uncertainty_dict

    def cpmi(self, uncertainty_dict, pred_logprobs):

        lambda_val = 0.5
        tau = 0.5

        input_text = uncertainty_dict['input_text']  # The initial prompt
        new_ids = uncertainty_dict['new_ids']  # List of token IDs for the generated response
        new_text = uncertainty_dict['new_text']  # The generated text

        # Calculate log probabilities for each generated token
        log_prob_list = []
        for i, log_prob_tensor in enumerate(pred_logprobs):
            index = new_ids[i]  # Index of the current token in the model's vocabulary
            log_prob_list.append(log_prob_tensor[0, index].item())

        # Calculate log probabilities for each generated token without x.
        # Generate len(new_ids) times
        log_prob_list_nox = []
        temp_prompt = '\n'
        for i in range(len(new_ids)):
            print('temp_prompt', temp_prompt)
            temp_generation_dict = self.generate(temp_prompt, max_new_tokens=512, return_type='generation_dict')
            temp_pred_logprobs = self.generate(temp_prompt, max_new_tokens=512, return_type='pred_logprobs')
            temp_new_ids = temp_generation_dict['new_ids']  # List of token IDs for the generated response
            index = temp_new_ids[0]
            log_prob_list_nox.append(temp_pred_logprobs[0][0,index].item())
            temp_prompt = self.tokenizer.decode(new_ids[: i])

        # Compute the entropy for each tensor in the tuple
        entropy_list = []
        for log_prob_tensor in pred_logprobs:
            # Convert log probabilities to probabilities
            probabilities = torch.exp(log_prob_tensor)
            # Compute entropy using the probabilities and their log probabilities
            entropy = -(probabilities * log_prob_tensor).sum(dim=1)
            entropy_list.append(entropy)    

        global_cpmi_list = [a - b for a, b in zip(log_prob_list_nox, log_prob_list)]
        # Initialize CPMI components
        first_term = -sum(global_cpmi_list) / len(global_cpmi_list) if len(global_cpmi_list) > 0 else 0
        # Calculate the second term, incorporating the entropy conditional
        second_term = 0
        for p_no_x, entropy in zip(log_prob_list_nox, entropy_list):
            if entropy > tau:
                second_term += p_no_x
        
        # Scale the second term by lambda
        second_term *= lambda_val
        # Calculate final CPMI
        global_cpmi = first_term + second_term

        if 'global_unc_score' not in uncertainty_dict:
            uncertainty_dict['global_unc_score'] = {}
        uncertainty_dict['global_unc_score'].update({'CPMI': global_cpmi})

        # Linewise ptrue scores
        # Split the text into lines
        if 'line_unc_scores' not in uncertainty_dict:
            uncertainty_dict['line_unc_scores'] = []

        line_split, _,_,_ = self.split_lines(new_ids)
        token_indices_split = self.token_indices_by_line(new_ids)

        for i, line_indices in enumerate(token_indices_split):
            if not line_indices:
                continue
            line_text = line_split[i]

            line_logprobs = [log_prob_list[i] for i in line_indices]
            line_entropy_list = [entropy_list[i] for i in line_indices]
            line_logprobs_nox = []
            temp_line_prompt = '\n'
            for j in line_indices:
                print('temp_line_prompt', temp_line_prompt)
                temp_line_generation_dict = self.generate(temp_line_prompt, max_new_tokens=1, return_type='generation_dict')
                temp_line_pred_logprobs = self.generate(temp_line_prompt, max_new_tokens=1, return_type='pred_logprobs')
                temp_line_new_ids = temp_line_generation_dict['new_ids']  # List of token IDs for the generated response
                line_index = temp_line_new_ids[0]
                line_logprobs_nox.append(temp_line_pred_logprobs[0][0,line_index].item())

                temp_line_prompt = self.tokenizer.decode(new_ids[: j])

            line_cpmi_list = [a - b for a, b in zip(line_logprobs_nox, line_logprobs)]
            # Initialize CPMI components
            line_first_term = -sum(line_cpmi_list) / len(line_cpmi_list) if len(line_cpmi_list) > 0 else 0
            # Calculate the second term, incorporating the entropy conditional
            line_second_term = 0
            for line_p_no_x, line_entropy in zip(line_logprobs_nox, line_entropy_list):
                if line_entropy > tau:
                    second_term += line_p_no_x
            
            # Scale the second term by lambda
            line_second_term *= lambda_val
            # Calculate final CPMI
            line_cpmi = line_first_term + line_second_term
            new_scores = {"CPMI": line_cpmi}

            existing_line = next((item for item in uncertainty_dict['line_unc_scores'] if item[0] == line_text), None)
            if existing_line:
                existing_line[1].update(new_scores)
            else:
                uncertainty_dict['line_unc_scores'].append((line_text, new_scores))

        return uncertainty_dict

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        """
        Decode the token IDs into a string.

        Args:
        token_ids (list or torch.Tensor): Token IDs to decode.
        skip_special_tokens (bool): Whether to skip special tokens.
        clean_up_tokenization_spaces (bool): Whether to clean up tokenization spaces.

        Returns:
        str: Decoded string.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=clean_up_tokenization_spaces)


# from datasets import load_dataset
# dataset = load_dataset("evalplus/humanevalplus")
# user_message = dataset['test'][0]['prompt']
# task_id = dataset['test'][0]['task_id']

# system_message = '''You are a Python code generator. Generate a complete and functioning Python function based on the provided code snippet.
# Ensure the function includes the original instructions in the comments, in-line comments for each line of code, and import statements for any required dependencies.
# Do not include main function. Enclose your code inside a ```python``` block.'''


# chat = [
#     {"role": "system", "content": system_message},
#     {"role": "user", "content": user_message}
#     ]

# # chat = [
# #     {"role": "system", "content": 'Reply in two lines'},
# #     {"role": "user", "content": 'hello \n how are you?'}
# #     ]

# # llm = WhiteBoxLLM('AutoModelForCausalLM')
# # llm.load_pretrained("xiaodongguaAIGC/llama-3-debug")

# llm = WhiteBoxLLM('AutoModelForCausalLM')
# llm.load_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", use_multi_gpu=True)



# # tokenized_input = llm.tokenize_input(chat, mode='chat', add_special_tokens=True, tokenize=True)
# # print(type(tokenized_input))
# # pdb.set_trace()
# # VALID_RETURN_TYPES = ['generation_text', 'generation_dict', 'pred_logprobs', 'raw']


# # generation_text = llm.generate(tokenized_input, max_new_tokens=512, return_type='generation_text')
# generation_dict = llm.generate(chat, max_new_tokens=512, return_type='generation_dict')
# # pred_logprobs = llm.generate(chat, max_new_tokens=512, return_type='pred_logprobs')
# # raw = llm.generate(chat, max_new_tokens=512, return_type='raw')
# uncertainty_dict = llm.generate_uncertainty_scores(generation_dict, debug=True)

# print('uncertainty_dict', uncertainty_dict)