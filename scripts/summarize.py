from dotenv import load_dotenv
import os
import sys
load_dotenv(".env")

sys.path.append(os.environ.get("PACKAGE_PATH"))
from package.mongodoc import Mongodoc

def main():
    print("running main.py")
    fpath = "/Users/beneverman/Documents/Coding/semantic-search/ai-summarizer/data/powers2017.pdf"    
    print("Complete")

if __name__ == "__main__":
    main()