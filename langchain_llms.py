from langchain.document_loaders import YoutubeLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
import pickle
from PyPDF2 import PdfReader
import os
import textract
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

load_dotenv()
embeddings = OpenAIEmbeddings()

def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=100)
    docs  = text_splitter.split_documents(transcript)

    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query_for_youtube(db, query, k=4):
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name='text-davinci-003')

    prompt = PromptTemplate(
        input_variables = ['question','docs'],
        template = 
        """You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
        )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query,docs = docs_page_content)
    response = response.replace('\n','')

    return response, docs


# def create_vector_db_from_pdf(pdf) -> FAISS:
    # pdf_reader = PdfReader(pdf)
    # text = ""
    # for page in pdf_reader.pages:
    #     text += page.extract_text()

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size = 1000,
    #     chunk_overlap = 200,
    #     length_function = len
    #     )
    # chunks = text_splitter.split_text(text=text)
    # store_name = pdf.name[:-4]
    #     # st.write(chunks)
 
    # if os.path.exists(f"{store_name}.pkl"):
    #     with open(f"{store_name}.pkl", "rb") as f:
    #         VectorStore = pickle.load(f)
    #     # st.write('Embeddings Loaded from the Disk')s
    # else:
    #     embeddings = OpenAIEmbeddings()
    #     VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
    #     with open(f"{store_name}.pkl", "wb") as f:
    #         pickle.dump(VectorStore, f)

    # return VectorStore

# def get_response_from_query_for_pdf(VectorStore, query, k=4):
#     docs = VectorStore.similarity_search(query=query, k=3)
#     llm = OpenAI()
#     chain = load_qa_chain(llm=llm, chain_type="stuff")
#     with get_openai_callback() as cb:
#         response = chain.run(input_documents=docs, question=query)
#     response = response.replace('\n','')

#     return response, VectorStore
    
    
def create_vector_db_from_pdf(pdf) -> FAISS:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
       text += page.extract_text()

    # with open('attention_is_all_you_need.txt', 'w') as f:
    #     f.write(text.decode('utf-8'))

    # with open('attention_is_all_you_need.txt', 'r') as f:
    #     text = f.read()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap  = 24,
        length_function = count_tokens,
    )

    chunks = text_splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)
    return db

def get_response_from_query_for_pdf(db, query, k=4):
    prompt = PromptTemplate(
        input_variables = ['question','docs'],
        template = 
        """You are a helpful assistant that that can answer questions about pdf documents 
        based on the pdf's text.
        
        Answer the following question: {question}
        By searching the following pdf file text: {docs}
        
        Only use the factual information from the file to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
        )
    
    docs = db.similarity_search(query)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model_name='text-davinci-003')
    chain = LLMChain(llm=llm, prompt=prompt)
    #chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff", prompt = prompt)
    
    response = chain.run(docs=docs_page_content, question=query)

    response = response.replace('\n','')

    return response, docs