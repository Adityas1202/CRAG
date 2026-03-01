import os
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_data():
    path = './data/raw/'
    docs = []
    
    print("--- Loading Documents ---")
    
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        ext = os.path.splitext(file)[1].lower()
        
        try:
            if ext == ".pdf":
                loader = PyMuPDFLoader(file_path)
                docs.extend(loader.load())
                print(f"✅ Successfully loaded PDF: {file}")
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding='utf-8')
                docs.extend(loader.load())
                print(f"✅ Successfully loaded TXT: {file}")
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")

    if not docs:
        print("\nFATAL ERROR: No text could be extracted from your files.")
        print("Tip: If your PDF is a scanned image, try a .txt file instead!")
        return

   # 2. Split into chunks (Reduced chunk_size for small files)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,  # Lowered from 500
        chunk_overlap=20,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    
    # CRITICAL CHECK: Stop if no chunks were made
    if not splits:
        print("⚠️ Warning: Document was loaded but no text chunks were created.")
        print("Try adding more text to your data.txt file.")
        return
    
    # 3. Create Vector DB
    print(f"--- Indexing {len(splits)} chunks into the database ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./data/vector_db"
    )
    print("🚀 FINISHED: Your AI now has memory.")

if __name__ == "__main__":
    ingest_data()