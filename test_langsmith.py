from dotenv import load_dotenv
load_dotenv(override=True)

from langsmith import Client

client = Client()
print("Resolved API URL:", client.api_url)

try:
    projects = list(client.list_projects(limit=1))
    print("Auth works. Projects visible:", [p.name for p in projects])
except Exception as e:
    print("Auth failed:", e)