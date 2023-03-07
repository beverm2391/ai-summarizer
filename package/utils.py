import re

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