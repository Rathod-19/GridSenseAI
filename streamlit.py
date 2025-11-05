import os
import subprocess

def handler(event, context):
    subprocess.run(["streamlit", "run", "app.py"])
    return {"statusCode": 200, "body": "Streamlit app is running"}