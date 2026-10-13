from pathlib import Path
from app.core.config import get_settings
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents

settings  = get_settings()

folder  = Path(settings.sample_kb_dir)

## i'm using list comprehension for listing all my files as list , for that logic is 
## if file exist add the file in to list

files = [source_file for source_file in folder.iterdir() if source_file.is_file()]


## look through the files and create chunk each file and embed it and add it to vector data base
all_docs =[]
for path in files:
    all_docs.extend(load_file(path))
    chunks = chunk_documents(all_docs)
    ids = add_documents(chunks)
    print(f"Indexed {len(files)} files -> {len(chunks)} chunks -> {len(ids)} Pinecone vectors")
