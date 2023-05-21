import pickle
import os

from llama_index import download_loader, GPTVectorStoreIndex
download_loader("GithubRepositoryReader")

from llama_index.readers.llamahub_modules.github_repo import GithubClient, GithubRepositoryReader

docs = None
if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        docs = pickle.load(f)

if docs is None:
    github_client = GithubClient(os.getenv("GITHU_TOKEN"))
    loader = GithubRepositoryReader(
        github_client,
        owner =                  "jerryjliu",
        repo =                   "llama_index",
        filter_directories =     (["gpt_index", "docs"], GithubRepositoryReader.FilterType.INCLUDE),
        filter_file_extensions = ([".py"], GithubRepositoryReader.FilterType.INCLUDE),
        verbose =                True,
        concurrent_requests =    10,
    )

    docs = loader.load_data(branch="main")

    with open("docs.pkl", "wb") as f:
        pickle.dump(docs, f)

index = GPTVectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine()
response = query_engine.query("Explain each LlamaIndex class?")
# response = index.query("Explain each LlamaIndex class?")
print(response)