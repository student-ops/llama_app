import pickle
import os
from datetime import datetime 

from llama_index import download_loader, GPTVectorStoreIndex
download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader

docs = None
if os.path.exists("docs.pkl"):
    current_datetime = datetime.now()
    with open('pkl_update_log.txt', 'a') as f:
        f.write(str(current_datetime) + '\n')
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "student-ops",
        repo =                   "raspi_go_TE",
        filter_directories =     (["python","go_serial"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".py",".go",".md"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="master")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

index = GPTVectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("このレポジトリではgoはどのように使用されていますか簡単に説明して")
# response = index.query("Explain each LlamaIndex class?")
print(response)