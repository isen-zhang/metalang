import copy
import json
import argparse
import os
# import lm_eval
# from lm_eval.models.huggingface import HFLM
import random
from typing import List
from prompts.utils import tokenize_conversation
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# def lm_evaluate(
#     model_name_or_path: str,
#     tasks: List[str] = None,
#     batch_size: int = 1,
#     output_path: str = None,
#     device: str = "cuda",
#     log_samples = False,
# ):
#     # create your model (could be running finetuning with some custom modeling code)
#     lm = HFLM(model_name_or_path, batch_size=batch_size)

#     # optional: the task_manager indexes tasks including ones
#     # specified by the user through `include_path`.
#     # task_manager = lm_eval.tasks.TaskManager(
#     #     include_path="/path/to/custom/yaml"
#     # )

#     results = lm_eval.simple_evaluate(
#         model=lm,
#         tasks=tasks,
#         device=device,
#         log_samples=log_samples,
#     )
#     results_dict = results['results']
#     samples = results['samples']
#     # print(results)
#     result_name = model_name_or_path.replace("/", "_")+"_results"
#     samples_name = model_name_or_path.replace("/", "_")+"_samples"
#     if output_path:
#         with open(os.path.join(output_path, f"{result_name}.json"), "w") as f:
#             json.dump(results_dict, f, indent=2)
#         with open(os.path.join(output_path, f"{samples_name}.json"), "w") as f:
#             json.dump(samples, f, indent=2)
def format_sample_with_prompt(task_name, traj, question, prompts, prompt):
    conversation_prompt = prompts[prompt]
    if prompt in {'zero-shot', 'cot-zero-shot'}:
        prompt_messages = conversation_prompt['conversations']
        question_prompt = conversation_prompt['question_prompt']
        for message in prompt_messages:
            message['loss'] = False

        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
            
    elif prompt in {
        'zero-shot-target-aware',
        'zero-shot-target-aware-careful-1',
        'zero-shot-target-aware-prefix',
        'zero-shot-target-aware-2',
        'zero-shot-target-aware-meaningless',
        'cot-zero-shot-target-aware',
        'zero-shot-target-aware-inverse',
        'zero-shot-target-aware-prefix-correct',
        'zero-shot-target-aware-suffix-correct',
        'zero-shot-target-aware-prefix-good',
        'zero-shot-target-aware-prefix-laptop',
        'zero-shot-target-aware-prefix-AB',
        'zero-shot-target-aware-suffix-AB',
        'zero-shot-target-aware-prefix-correct-inverse',
        'zero-shot-target-aware-good',
        'zero-shot-target-aware-random-sentence',
    }:
        prompt_messages = conversation_prompt['conversations']
        if task_name in {'gsm8k', 'ASDiv', 'MultiArith', 'SVAMP'}:
            question_prompt = conversation_prompt['correct_question_prompt'] if traj['info']['reward'] else conversation_prompt['incorrect_question_prompt']
        elif task_name in {'hotpotqa', "strategyqa"}:
            question_prompt = conversation_prompt['correct_question_prompt'] if traj['info']['reward'] else conversation_prompt['incorrect_question_prompt']
        else:
            raise NotImplementedError(f"Task {task_name} is not supported.")

        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif prompt in {'zero-shot-target-aware-soft'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = conversation_prompt['conversations']
        if traj['info']['f1'] > 0.2:
            question_prompt = conversation_prompt['correct_question_prompt']
        elif traj['info']['f1'] <= 0.2:
            question_prompt = conversation_prompt['incorrect_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif prompt in {'zero-shot-target-aware-three-class'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = conversation_prompt['conversations']
        if traj['info']['f1'] == 0.0:
            question_prompt = conversation_prompt['class_1_question_prompt']
        elif traj['info']['f1'] > 0.0 and traj['info']['f1'] < 1.0:
            question_prompt = conversation_prompt['class_2_question_prompt']
        elif traj['info']['f1'] == 1.0:
            question_prompt = conversation_prompt['class_3_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif prompt in {'zero-shot-target-aware-four-class'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = conversation_prompt['conversations']
        if traj['info']['f1'] == 0.0:
            question_prompt = conversation_prompt['class_1_question_prompt']
        elif traj['info']['f1'] > 0.0 and traj['info']['f1'] < 0.4:
            question_prompt = conversation_prompt['class_2_question_prompt']
        elif traj['info']['f1'] >= 0.4 and traj['info']['f1'] < 1.0:
            question_prompt = conversation_prompt['class_3_question_prompt']
        elif traj['info']['f1'] == 1.0:
            question_prompt = conversation_prompt['class_4_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif prompt in {'zero-shot-target-aware-five-class'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = conversation_prompt['conversations']
        if traj['info']['f1'] == 0.0:
            question_prompt = conversation_prompt['class_1_question_prompt']
        elif traj['info']['f1'] > 0.0 and traj['info']['f1'] < 0.2:
            question_prompt = conversation_prompt['class_2_question_prompt']
        elif traj['info']['f1'] >= 0.2 and traj['info']['f1'] < 0.4:
            question_prompt = conversation_prompt['class_3_question_prompt']
        elif traj['info']['f1'] >= 0.4 and traj['info']['f1'] < 1.0:
            question_prompt = conversation_prompt['class_4_question_prompt']
        elif traj['info']['f1'] == 1.0:
            question_prompt = conversation_prompt['class_5_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif prompt in {'zero-shot-target-aware-six-class'}:
        assert task_name == 'hotpotqa' and 'f1' in traj['info']
        prompt_messages = conversation_prompt['conversations']
        if traj['info']['f1'] == 0.0:
            question_prompt = conversation_prompt['class_1_question_prompt']
        elif traj['info']['f1'] > 0.0 and traj['info']['f1'] < 0.2:
            question_prompt = conversation_prompt['class_2_question_prompt']
        elif traj['info']['f1'] >= 0.2 and traj['info']['f1'] < 0.4:
            question_prompt = conversation_prompt['class_3_question_prompt']
        elif traj['info']['f1'] >= 0.4 and traj['info']['f1'] < 0.8:
            question_prompt = conversation_prompt['class_4_question_prompt']
        elif traj['info']['f1'] >= 0.8 and traj['info']['f1'] < 1.0:
            question_prompt = conversation_prompt['class_5_question_prompt']
        elif traj['info']['f1'] == 1.0:
            question_prompt = conversation_prompt['class_6_question_prompt']
        else:
            raise ValueError(f"Invalid f1 score: {traj['info']['f1']}")
        for message in prompt_messages:
            message['loss'] = False
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    elif prompt in {'NAT-sampling'}:
        prompt_messages = conversation_prompt['conversations']
        positive_question_prompts = conversation_prompt['correct_question_prompts']
        negative_question_prompts = conversation_prompt['incorrect_question_prompts']
        if traj['info']['reward']:
            question_prompt = random.sample(list(positive_question_prompts.values()), 1)[0]
        else:
            question_prompt = random.sample(list(negative_question_prompts.values()), 1)[0]
        prompt_messages_with_question = prompt_messages + [
            {
                'role': 'user',
                'content': question_prompt.format(question=question),
                'loss': False
            }
        ]
    else:
        raise NotImplementedError(f"Prompt {prompt} is not supported")

    formatted_traj = copy.deepcopy(traj)
    formatted_traj['conversations'] = prompt_messages_with_question + formatted_traj['conversations']

    return formatted_traj


def compute_perplexity(
    model_name_or_path: str,
    trajectory_file,
    prompt_file,
    question_file: str,
    prompt: str,
    template: str,
    output_file: str,
):
    with open(trajectory_file, 'r') as f:
        trajs = json.load(f)
    with open(prompt_file, 'r') as f:
        prompts = json.load(f)
    with open(question_file, 'r') as f:
        questions = json.load(f)
    questions_dict = {}
    for q in questions:
        questions_dict[q['id']] = q


    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    all_perplexities = []
    for traj in trajs:
        formatted_traj = format_sample_with_prompt("gsm8k", traj, questions_dict[traj['id']]['question'], prompts, prompt=prompt)
        conversation = formatted_traj['conversations']
        inputs = tokenize_conversation(conversation, tokenizer, conv_template=template, max_length=4096)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            perplexity = torch.exp(outputs.loss)
            print(f"Perplexity: {perplexity.item()}")
            all_perplexities.append(perplexity.item())

    print(f"Average perplexity: {sum(all_perplexities) / len(all_perplexities)}")
    base_name = model_name_or_path.split("/")[-1]
    with open(output_file, 'a') as f:
        f.write(json.dumps({
            "model": base_name,
            "perplexity": sum(all_perplexities) / len(all_perplexities)
        }) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--trajectory_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--question_file", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--template", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()
    compute_perplexity(
        args.model_name_or_path,
        args.trajectory_file,
        args.prompt_file,
        args.question_file,
        args.prompt,
        args.template,
        args.output_file,
    )