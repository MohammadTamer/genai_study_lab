from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document


mydocument = Document(text="Hello, this is a test. I am from Singapore.")
documents = SimpleDirectoryReader("my_folder", recursive=True).load_data()
documents = SimpleDirectoryReader(input_files=["my_folder/file1.txt", "my_folder/file2.txt"]).load_data()
documents = SimpleDirectoryReader(input_dir="my_folder", required_exts=[".pdf"]).load_data()



node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = node_parser.get_nodes_from_documents(documents)

print(mydocument.dict())
