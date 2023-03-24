import re
import tiktoken

# this prevents the metadata from being None which causes errors with the vectorstore
def sanitize_metadata(data):
    for item in data:
        meta = item.metadata
        for key, value in meta.items():
            if value is None:
                meta[key] = ""
    return data

def sanitize_text(text):
    # Replace any non-alphanumeric character with a space
    text = re.sub(r'[^\w\s]', ' ', text)
    # Replace any multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text

def unpack (data):
    return [{'page' : idx + 1, 'content' : page.page_content, 'metadata' : page.metadata} for idx, page in enumerate(data)]

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
            print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            self.model="gpt-3.5-turbo-0301"
            self.tokens_per_message = 4
            self.tokens_per_name = -1  # if there's a name, the role is omitted
            # self.encoding = tiktoken.get_encoding("gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            print(
                "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            self.model="gpt-4-0314"
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