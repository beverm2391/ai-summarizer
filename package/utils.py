import re
import tiktoken
import os

# this prevents the metadata from being None which causes errors with the vectorstore


def sanitize_metadata(data):
    for item in data:
        meta = item.metadata
        for key, value in meta.items():
            if value is None:
                meta[key] = ""
    return data


def sanitize_text(text):
    if not isinstance(text, str):
        return None
    # Replace any non-alphanumeric character with a space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace any multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def unpack(data):
    return [{'page': idx + 1, 'content': page.page_content, 'metadata': page.metadata} for idx, page in enumerate(data)]


class TokenUtil():
    def __init__(self, model):
        assert model is not None, "Model must be specified. (passed as positional argument)"
        self.model = self.validate_model(model)

    def validate_model(self, model):
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            self.encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            print(
                "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            self.model = "gpt-3.5-turbo-0301"
            self.tokens_per_message = 4
            self.tokens_per_name = -1  # if there's a name, the role is omitted
            # self.encoding = tiktoken.get_encoding("gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            print(
                "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            self.model = "gpt-4-0314"
            self.tokens_per_message = 3
            self.tokens_per_name = 1
            # self.encoding = tiktoken.get_encoding("gpt-4-0314")

    def get_tokens(self, text):
        # sanitize text
        text = sanitize_text(text)
        if text is None:
            return 0

        num_tokens = 0
        if isinstance(text, str):
            num_tokens += len(self.encoding.encode(text))
        elif isinstance(text, list):
            for message in text:
                num_tokens += self.tokens_per_message
                for key, value in message.items():
                    num_tokens += len(self.encoding.encode(value))
                    if key == "name":
                        num_tokens += self.tokens_per_name
        else:
            raise TypeError("text must be a string or list of messages.")

        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def encode(self, text):
        return self.encoding.encode(text)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def split_tokens(self, text, max_tokens):
        assert isinstance(
            text, str), f"input text for split_tokens must be a string, not {type(text)}"
        # sanitize text
        text = sanitize_text(text)
        if text is None:
            return 0

        tokens = self.encoding.encode(text)
        split_tokens = []
        while len(tokens) > max_tokens:
            split_tokens.append(tokens[:max_tokens])
            tokens = tokens[max_tokens:]
        split_tokens.append(tokens)
        split_text = [self.encoding.decode(tokens) for tokens in split_tokens]
        print(f"Returning {len(split_text)} split messages.")
        return split_text


def get_tokens(text, model):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return get_tokens(text, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return get_tokens(text, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        # every message follows <im_start>{role/name}\n{content}<im_end>\n
        tokens_per_message = 4
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    # else:
    #     raise NotImplementedError(
    #         f"""get_tokens() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

    num_tokens = 0
    if isinstance(text, str):
        num_tokens += len(encoding.encode(text))
    elif isinstance(text, list):
        for message in text:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        raise TypeError("text must be a string or list of messages.")

    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def increment_filename(base_path, base_filename, extension):
    index = 1
    while True:
        filename = os.path.join(
            base_path, f"{base_filename}_{index}.{extension}")
        if not os.path.exists(filename):
            return filename
        index += 1


def write_file(base_path, base_filename, extension, content):
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    filename = increment_filename(base_path, base_filename, extension)
    with open(filename, 'w') as f:
        f.write(content)
    return filename

def pretty_print(data, indent=0):
    if isinstance(data, dict):
        for key, value in data.items():
            print('\t' * indent + str(key) + ':')
            pretty_print(value, indent + 1)
    elif isinstance(data, list):
        for item in data:
            pretty_print(item, indent + 1)
    else:
        print('\t' * indent + str(data))


def get_template_schema():
    """Generates a JSON file containing the instructions for each task."""
    data = [
        {
            "title": "Introduction",
            "sublinks": [
                {
                    "subtitle": "What is this?",
                    "prompt": "Answer this question.",
                },
            ]
        },
        {
            "title": "heading 1",
            "sublinks": [
                {
                    "subtitle": "subheading 1",
                    "prompt": "Answer this question.",
                },
                {
                    "subtitle": "subheading 2",
                    "prompt": "Answer this question.",
                }
            ]
        },
        {
            "title": "heading 2",
            "sublinks": [
                {
                    "subtitle": "subheading 1",
                    "prompt": "Answer this question.",
                },
                {
                    "subtitle": "subheading 2",
                    "prompt": "Answer this question.",
                }
            ]
        },
        {
            "title": "Conclusion",
            "prompt": "Write a conclusion.",
        }
    ]
    return data


def get_example_schema():
    data = {
        "title": "Main Idea: Healthy Eating",
        "sublinks": [
            {
                "subtitle": "Sub Idea: Benefits of Healthy Eating",
                "prompt": "What are the benefits of healthy eating?",
            },
            {
                "subtitle": "Sub Idea: Components of a Healthy Diet",
                "prompt": "What are the components of a healthy diet?",
            }
        ]
    }
    return data