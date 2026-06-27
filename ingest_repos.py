import requests
import zipfile
import io
import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_API_TOKEN")
ALLOWED_EXTENSIONS = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".yml", ".yaml"}
SKIP_KEYWORDS = {"node_modules", "__pycache__", ".git", "dist/", "build/", "docs", "scripts", "test"}

def get_repo_files(owner, repo, token=GITHUB_TOKEN):
    url = f"https://api.github.com/repos/{owner}/{repo}/zipball/HEAD"
    headers = {"Authorization": f"token {token}"}
    
    print("Downloading zip")
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    files = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for name in z.namelist():
            # strip the top level folder github adds
            clean_path = "/".join(name.split("/")[1:])
            
            if not clean_path:
                continue
            # skip files that don't end in allowed extensions
            if not any(clean_path.endswith(ext) for ext in ALLOWED_EXTENSIONS):
                continue
            # skip common noise
            if any(skip in clean_path for skip in SKIP_KEYWORDS):
                continue
                
            with z.open(name) as f:
                try:
                    content = f.read().decode("utf-8", errors="ignore")
                    if content.strip():  # skip empty files
                        files.append({"path": clean_path, "content": content})
                except Exception:
                    continue
    
    print(f"Fetched {len(files)} files")
    return files


# ------ test -------
if __name__ == "__main__":
    files = get_repo_files("HarshitK150", "micro-foods-market")
    print(f"Fetched {len(files)} files!")
    if (len(files) > 0):
        print(files[0])