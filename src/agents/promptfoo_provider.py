import os
import sys
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-3.1-flash-lite-preview")

def main():
    try:
        # Promptfoo passes the prompt string as JSON to stdin
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"error": "No input provided on stdin"}))
            sys.exit(1)
            
        # The prompt is a simple JSON string or object containing '{ "prompt": "..." }'
        parsed = json.loads(input_data)
        
        # Depending on how promptfoo calls this, it may be a direct string or a dict
        if isinstance(parsed, dict) and 'prompt' in parsed:
            prompt = parsed['prompt']
        elif isinstance(parsed, str):
            prompt = parsed
        else:
            prompt = str(parsed)
            
        response = model.generate_content(prompt)
        
        # Promptfoo expects the response inside a JSON object: { "output": "..." }
        print(json.dumps({"output": response.text}))
        sys.exit(0)
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
