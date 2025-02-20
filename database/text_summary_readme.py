'''
After running `test_get_all_repo.py`, it retrieves all open-source project README files from the Datawhale organization repository.
But these README files contain a lot of irrelevant information.
Running this script can generate a summary for each README 
and save it to `../knowledge_db/readme_summary`, which serves as part of the knowledge base.

- `remove_urls()` filters out web links in README files and removes words that may trigger LLM security restrictions.
- `extract_text_from_md()` extracts plain text from Markdown (`.md`) files.
- `generate_llm_summary()` generates a summary for each README using an LLM.
'''


import os
from dotenv import load_dotenv
import openai
from test_get_all_repo import get_repos
from bs4 import BeautifulSoup
import markdown
import re
import time

load_dotenv()
TOKEN = os.getenv('TOKEN')
openai_api_key = os.environ["OPENAI_API_KEY"]

# Filter out URLs to prevent potential LLM security restrictions
def remove_urls(text):
    # Regular expression pattern to match URLs
    url_pattern = re.compile(r'https?://[^\s]*')
    # Replace all matched URLs with an empty string
    text = re.sub(url_pattern, '', text)
    # Regular expression pattern to filter out specific words/phrases
    specific_text_pattern = re.compile(r'扫描下方二维码关注公众号|提取码|关注|科学上网|回复关键词|侵权|版权|致谢|引用|LICENSE'
                                       r'|组队打卡|任务打卡|组队学习的那些事|学习周期|开源内容|打卡|组队学习|链接')
    # Replace all matched words/phrases with an empty string
    text = re.sub(specific_text_pattern, '', text)
    
    return text

# Extract text from a Markdown file
def extract_text_from_md(md_content):
    # Convert Markdown to HTML
    html = markdown.markdown(md_content)
    # Use BeautifulSoup to extract text
    soup = BeautifulSoup(html, 'html.parser')

    return remove_urls(soup.get_text())

# Generate a summary for a README file using LLM
def generate_llm_summary(repo_name, readme_content,model):
    prompt = f"1：这个仓库名是 {repo_name}. 此仓库的readme全部内容是: {readme_content}\
               2:请用约200以内的中文概括这个仓库readme的内容,返回的概括格式要求：这个仓库名是...,这仓库内容主要是..."
    openai.api_key = openai_api_key

    messages = [{"role": "system", "content": "你是一个人工智能助手"},
                {"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    return response.choices[0].message["content"]

def main(org_name,export_dir,summary_dir,model):
    # Retrieve the list of repositories
    repos = get_repos(org_name, TOKEN, export_dir)

    # Create a directory to save summaries
    os.makedirs(summary_dir, exist_ok=True)

    for id, repo in enumerate(repos):
        repo_name = repo['name']
        readme_path = os.path.join(export_dir, repo_name, 'README.md')
        print(repo_name)
        if os.path.exists(readme_path):
            with open(readme_path, 'r', encoding='utf-8') as file:
                readme_content = file.read()
            # Extract text from the README
            readme_text = extract_text_from_md(readme_content)
            
            # Generate a summary for the README
            time.sleep(60)  # API rate limit: One request per minute
            print('第' + str(id) + '条' + 'summary开始')
            try:
                summary = generate_llm_summary(repo_name, readme_text,model)
                print(summary)
                # Write summary to a Markdown file in the summary directory
                summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary.md")
                with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(f"# {repo_name} Summary\n\n")
                    summary_file.write(summary)
            except openai.OpenAIError as e:
                summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary风控.md")
                with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                    summary_file.write(f"# {repo_name} Summary风控\n\n")
                    summary_file.write("README内容风控。\n")
                print(f"Error generating summary for {repo_name}: {e}")
                # print(readme_text)
        else:
            print(f"File not found: {readme_path}")
            # If README doesn't exist, create an empty Markdown file
            summary_file_path = os.path.join(summary_dir, f"{repo_name}_summary不存在.md")
            with open(summary_file_path, 'w', encoding='utf-8') as summary_file:
                summary_file.write(f"# {repo_name} Summary不存在\n\n")
                summary_file.write("README文件不存在。\n")


if __name__ == '__main__':
    # Organization name
    org_name = 'datawhalechina'
    
    # Set export_dir
    export_dir = "./readme_db"  # Replace with the actual path
    summary_dir="../knowledge_db/readme_summary"
    
    # Set the LLM model
    model="gpt-3.5-turbo"  #deepseek-chat,gpt-3.5-turbo,moonshot-v1-8k
    
    main(org_name,export_dir,summary_dir,model)
