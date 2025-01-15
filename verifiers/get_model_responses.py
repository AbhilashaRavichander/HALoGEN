import spacy
import openai

import os
from together import Together
import requests

nlp = spacy.load("en_core_web_sm")


def get_instruction_model_response(client, prompt, instruction_model):

    try:
        response = client.chat.completions.create(
            model=instruction_model,
            messages=[{"role": "user", "content":  prompt}],
        )

        print(response.choices[0].message.content)
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return ""


def get_response_target(constructed_prompt, model_name, api_key):

    max_output_tokens = 128

    if "gpt" in model_name:

        if constructed_prompt:

            openai.api_key = api_key  # YOUR_API_KEY

            full_prompt = constructed_prompt

            print(full_prompt)

            # Put your API key here
            # while responses is None:
            response_text = ""
            num_tries, num_tries_limit = 0, 3
            # print(full_prompt)
            # print(token)
            print("\n\n----------------\n\n")

            if "instruct" not in model_name:

                response = openai.chat.completions.create(model=model_name, messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": full_prompt}], max_tokens=max_output_tokens,
                                                          temperature=0.0)

                response_text = response.choices[0].message.content

            else:
                # response = openai.chat.completions.create(model=modelname, messages=[{"role": "system", "content": "You are a helpful assistant."},{"role":"user", "content":full_prompt}], max_tokens=max_output_tokens,
                #                                 temperature=1.0)

                # response_text = response.choices[0].message.content

                from openai import OpenAI

                client = OpenAI(
                    api_key=api_key)

                completion = client.completions.create(
                    model=model_name, prompt=full_prompt)

                response_text = completion.choices[0].text

            return response_text  # ,responses['choices'][0]

        return False

    else:

        if constructed_prompt:
            full_prompt = constructed_prompt
            print(full_prompt)

            client = Together(
                api_key=api_key)
            response = get_instruction_model_response(
                client, full_prompt, model_name)

            return response
