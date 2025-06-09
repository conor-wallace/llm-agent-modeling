import os
import openai
from vllm import LLM, SamplingParams

# import dashscope
# import replicate
# from http import HTTPStatus


class GPT35API:
    def __init__(self) -> None:
        pass

    def response(self, mes):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=mes,
            top_p=0.95,
            temperature=1,
        )
        return response.choices[0].message.content


class GPT4API:
    def __init__(self, model_id) -> None:
        self.model_id = model_id

    def response(self, mes):
        response = openai.chat.completions.create(
            model=self.model_id,
            messages=mes,
            top_p=0.95,
            temperature=1,
        )
        return response.choices[0].message.content


class DeepSeekR1API:
    def __init__(self, model_id, device) -> None:
        self.model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
        # self.model = LLM(model=self.model_id, tensor_parallel_size=2)
        self.model = LLM(model=self.model_id)
        self.sampling_params = self.model.get_default_sampling_params()
        self.sampling_params.max_tokens = 12288

    def response(self, mes):
        response = self.model.chat(mes, self.sampling_params, use_tqdm=False)
        return response[0].outputs[0].text


class LlamaAPI:
    def __init__(self, model_id, device) -> None:
        self.model_id = model_id
        self.model = LLM(model=self.model_id)
        self.sampling_params = self.model.get_default_sampling_params()
        self.sampling_params.max_tokens = 12288

    def response(self, mes):
        response = self.model.chat(mes, self.sampling_params, use_tqdm=False)
        return response[0].outputs[0].text


# class llama2_70b_chatAPI:

#     def response(self, mes):
#         os.environ["REPLICATE_API_TOKEN"] = "YOUR KEY"
#         system_prompt = ""
#         prompt = ""
#         for item in mes:
#             if item.get('role') == 'system':
#                 system_prompt = item.get('content')
#             if item.get('role') == 'user':
#                 prompt = item.get('content')
#         try:
#             iterator = replicate.run(
#                 "meta/llama-2-70b-chat",
#                 input={
#                     "system_prompt": system_prompt,
#                     "prompt": prompt,
#                     "temperature": 1,
#                     "top_p": 0.95,
#                     "max_new_tokens": 4000,
#                 },
#             )
#             result_string = ''.join(text for text in iterator)
#         except replicate.exceptions.ModelError as e:
#             with open("/replicate_modelerror_times.txt", "a") as f:
#                 f.write("llama2_70b_chat:replicate_modelerror_times\n")
#             print(e)
#         except Exception as e:
#             with open("Exception.txt", "a") as f:
#                 f.write("llama2_70b_chat:Exception_times\n")
#             print(e)
#         return result_string


# class QwenAPI:

#     def __init__(self) -> None:
#         dashscope.api_key = 'YOUR KEY'

#     def response(self, mes):
#         response = dashscope.Generation.call(
#             'qwen-72b-chat',
#             messages=mes,
#             temperature=1,
#             top_p=0.95,
#             result_format='message',
#         )
#         if response.status_code == HTTPStatus.OK:
#             data_res = response['output']['choices'][0]['message']['content']
#             return data_res
#         else:
#             print(
#                 'Request id: %s, Status code: %s, error code: %s, error message: %s'
#                 % (response.request_id, response.status_code, response.code,
#                    response.message))
