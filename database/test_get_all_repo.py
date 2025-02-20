'''
Collect all open-source project README files from the Datawhale organization repository.
'''

import json
import requests
import os
import base64
import loguru
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv('TOKEN')

# Fetch repository list from an organization
def get_repos(org_name, token, export_dir):
    headers = {
        'Authorization': f'token {token}',
    }
    url = f'https://api.github.com/orgs/{org_name}/repos'
    response = requests.get(url, headers=headers, params={'per_page': 200, 'page': 0})
    if response.status_code == 200:
        repos = response.json()
        loguru.logger.info(f'Fetched {len(repos)} repositories for {org_name}.')
        # Define file path to save repository names using export_dir
        repositories_path = os.path.join(export_dir, 'repositories.txt')
        with open(repositories_path, 'w', encoding='utf-8') as file:
            for repo in repos:
                file.write(repo['name'] + '\n')
        return repos
    else:
        loguru.logger.error(f"Error fetching repositories: {response.status_code}")
        loguru.logger.error(response.text)
        return []

# Fetch README files from a repository
def fetch_repo_readme(org_name, repo_name, token, export_dir):
    headers = {
        'Authorization': f'token {token}',
    }
    url = f'https://api.github.com/repos/{org_name}/{repo_name}/readme'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        readme_content = response.json()['content']
        # Decode base64 content
        readme_content = base64.b64decode(readme_content).decode('utf-8')
        # Define file path to save the README using export_dir
        repo_dir = os.path.join(export_dir, repo_name)
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir)
        readme_path = os.path.join(repo_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as file:
            file.write(readme_content)
    else:
        loguru.logger.error(f"Error fetching README for {repo_name}: {response.status_code}")
        loguru.logger.error(response.text)


if __name__ == '__main__':
    # Organization name
    org_name = 'datawhalechina'
    # Set export_dir
    export_dir = "database/readme_db"  # Replace with the actual directory path
    # Fetch the list of repositories
    repos = get_repos(org_name, TOKEN, export_dir)
    # Print repository names and fetch each repository's README
    if repos:
        for repo in repos:
            repo_name = repo['name']
            fetch_repo_readme(org_name, repo_name, TOKEN, export_dir)
    
    # Clean up temporary files (if needed)
    # if os.path.exists('temp'):
    #     shutil.rmtree('temp')