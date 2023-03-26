from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")
sys.path.append(os.environ.get("PACKAGE_PATH"))

OPENAI_API_KEY = os.environ.get('OPENAI-API-KEY')
import openai
openai.api_key = OPENAI_API_KEY

import asyncio

from package.sync_api import ChatV2
from package.async_api_v2 import run_chat_async
from package.utils import pretty_print, get_template_schema, get_example_schema
from package.document import Document

class Schema():
    def __init__(self, template, example, model='gpt-3.5-turbo'):
        self.template = template
        self.example = example
        self.model = model
    
    def generate_schemas(self, contexts):
        print("Generating schema...")
        prompts = [f"""
                Here is an example schema:
                {self.template[1]}

                Instructions:
                    1. I have given you a section from an article
                    2. Identify the main idea
                    3. Identify sub ideas of this section, if applicable, else leave blank
                    4. Use the example schema to generate a new schema with your ideas.
                    5. Do not include any other text, only output the schema and nothing else 

                Here is an example output:
                {self.example}

                Context:
                {context}

                Output:
                """
            for context in contexts]

        data = []
        asyncio.run(run_chat_async(prompts, data, model=self.model))
        return [item["response"].replace("\n", " ").strip() for item in data]

    def pretty_print(self, data):
        return pretty_print(data)

def main():
    fpath = "/Users/beneverman/Documents/Coding/semantic-search/ai-summarizer/data/handgun_suicide_ml.pdf"
    doc = Document(fpath)
    doc.process_doc().get_chunks("gpt-3.5-turbo", 1000).get_citation('APA')
    template = get_template_schema()
    example = get_example_schema()
    schema_gen = Schema(template, example)
    schemas = schema_gen.generate_schemas(doc.chunks)
    pretty_print(schemas)


if __name__ == "__main__":
    main()