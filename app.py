import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
from pyvis.network import Network
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import PydanticOutputParser
from langchain.docstore.document import Document
import json
import os
import re
import spacy
from spacy.matcher import Matcher
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

docs = None
total_info = {
    "summary": None,
    "notes": None,
    "questions": None,
    "answers": None,
    "Chat History": [],
    "Mind Map": None,
}

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    # user_question is the input question
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # load the local faiss db
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # using similarity search, get the answer based on the input
    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True,
    )

    print(response)
    st.markdown(f"**Reply:** {response['output_text']}")

class MindMap(BaseModel):
    main_title: str = Field("Main title/concept(< 3 words) of the video.")
    concepts: List[str] = Field(description="Major Concepts(< 3 words each) in the text")
    subconcepts: List[List[str]] = Field(description="Corresponding list of sub-concepts(< 3 words each) for each concept with 3 subconcepts for each concept")

def extract_mindmap_data(json_response):
    try:
        # Remove the triple backticks and "json" identifier from the response
        json_string = json_response.strip().strip('`').strip('json').strip()
        # Parse the JSON data
        data = json.loads(json_string)
        return {
            "main_title": data.get("main_title", "No Title Found"),
            "concepts": data.get("concepts", []),
            "subconcepts": data.get("subconcepts", [])
        }
    except json.JSONDecodeError:
        return {
            "main_title": "Error parsing JSON",
            "concepts": [],
            "subconcepts": []
        }

def generate_quizlet(is_quizlet, num_mcq, num_frq):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(is_quizlet)
    chain = get_conversation_chain(task_type='quizlet')
    quizlet_prompt = f"Generate {num_mcq} multiple-choice questions and {num_frq} free-response questions based on this text. For each question, also provide the answer in the format 'Question: <question>\nAnswer: <answer>'\n\nText: {is_quizlet}"
    quizlet_response = chain({"input_documents": docs, "question": quizlet_prompt}, return_only_outputs=True)

    try:
        quizlet_data = quizlet_response.get('output_text', '')
        print("Quizlet Data:", quizlet_data)
    except json.JSONDecodeError:
        st.error("Failed to decode the response into JSON.")
        return

    # Parsing the quizlet data
    mcq_questions = {}
    frq_questions = {}

    question_type = None
    current_question = None
    st.write(quizlet_data)
    for line in quizlet_data.split('\n'):
        if line.startswith('Multiple-Choice Questions'):
            question_type = 'mcq'
        elif line.startswith('Short Answer Questions'):
            question_type = 'frq'
        elif line.startswith('Question:'):
            if question_type == 'mcq':
                question = line[10:].strip()
                mcq_questions[question] = {"options": [], "answer": ""}
                current_question = question
            elif question_type == 'frq':
                question = line[10:].strip()
                frq_questions[question] = ""
                current_question = question
        elif line.startswith('('):
            if question_type == 'mcq' and current_question:
                mcq_questions[current_question]["options"].append(line.strip())
        elif line.startswith('Answer:'):
            answer = line.split(':')[1].strip()
            if question_type == 'mcq' and current_question:
                mcq_questions[current_question]["answer"] = answer
            elif question_type == 'frq' and current_question:
                frq_questions[current_question] = answer

    return mcq_questions, frq_questions

def check_mcq_answer(user_answer, correct_answer):
    if user_answer.lower() == correct_answer.lower():
        st.success("Correct!")
    else:
        st.error(f"Incorrect. The correct answer is: {correct_answer}")


def generate_study_guide(is_study_guide):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(is_study_guide)
    chain = get_conversation_chain(task_type='study_guide')
    study_guide_prompt = f"Generate a study guide based on the area of {is_study_guide}"
    study_guide_response = chain({"input_documents": docs, "question": study_guide_prompt}, return_only_outputs=True)

    return study_guide_response.get('output_text', '')

def generate_cheatsheet(is_cheatsheet, num_pages):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(is_cheatsheet)
    chain = get_conversation_chain(task_type='cheatsheet')
    cheatsheet_prompt = f"Generate a {num_pages}-page cheatsheet based on the following text:\n\n{is_cheatsheet}"
    cheatsheet_response = chain({"input_documents": docs, "question": cheatsheet_prompt}, return_only_outputs=True)

    return cheatsheet_response.get('output_text', '')

def generate_mindmap(is_mindmap):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(is_mindmap)
    chain = get_conversation_chain()

    # Define the prompt for generating the mind map
    mindmap_prompt = f"""
    Based on the following text, generate a mind map in JSON format with the following structure:
    {{
        "main_title": "Main title/concept(< 3 words) of the video.",
        "concepts": ["Major Concept 1(< 3 words)", "Major Concept 2(< 3 words)", ...],
        "subconcepts": [
            ["Subconcept 1.1(< 3 words)", "Subconcept 1.2(< 3 words)", "Subconcept 1.3(< 3 words)"],
            ["Subconcept 2.1(< 3 words)", "Subconcept 2.2(< 3 words)", "Subconcept 2.3(< 3 words)"],
            ...
        ]
    }}

    Text: {docs}
    """

    # Use the chain to generate the mind map data
    response = chain({"input_documents": docs, "question": mindmap_prompt}, return_only_outputs=True)

    mindmap_data = extract_mindmap_data(response["output_text"])
    st.write("Mind Map Data:", mindmap_data)

    if not mindmap_data["concepts"]:  # Check if concepts are empty
        st.error("No data available to generate the mind map.")
        return

    net = Network(notebook=True, height='750px', width='100%', cdn_resources='remote')
    net.show_buttons(filter_=['physics'])

    main_title = mindmap_data["main_title"]
    net.add_node(main_title, label=main_title, title=main_title, shape='box', color='red')

    for concept, subconcepts in zip(mindmap_data["concepts"], mindmap_data["subconcepts"]):
        net.add_node(concept, label=concept, title=concept, shape='box', color='lightblue')
        net.add_edge(main_title, concept)
        for subconcept in subconcepts:
            net.add_node(subconcept, label=subconcept, title=subconcept, shape='box', color='lightgreen')
            net.add_edge(concept, subconcept)

    net_html = net.generate_html()
    components.html(net_html, height=800)

def get_conversation_chain(task_type='question'):
    """Creates and returns a conversation chain based on the specified task type."""

    template_dict = {
        'question': """
            Answer the question as detailed as possible from the provided context, make sure to provide all the details,
            if the answer is not in the provided context just say, "answer is not available in the context",
            don't provide the wrong answer.\n\nContext:\n{context}?\nQuestion: \n{question}\nAnswer:
        """,
        'mindmap': """
            Use the following pieces of context to generate a structured mind map.
            Context: {context}.\nFrom this text, formulate a mind map with main concepts and sub-concepts.
        """,
        'quizlet': """
            Generate a comprehensive quizlet based on the following text. Include multiple-choice questions and short answer questions.\n\nText: {context}
        """,
        'study_guide': """
            Generate a study guide based on the following text.\n\nText: {context}
        """,
        'cheatsheet': """
            Generate a cheatsheet based on the following text.\n\nText: {context}
        """
    }

    if task_type not in template_dict:
        raise ValueError("Unsupported task type provided to the conversation chain generator.")

    prompt_template = template_dict[task_type]
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def main():
    st.set_page_config("Chat PDF")
    st.markdown("<h1 style='color: #2E86C1;'>Chat with PDF using Gemini</h1>", unsafe_allow_html=True)

    # Create separate sections for different functionalities
    upload_container = st.container()
    question_container = st.container()
    mindmap_container = st.container()
    quizlet_container = st.container()
    study_guide_container = st.container()
    cheatsheet_container = st.container()

    with upload_container:
        st.markdown("<h2 style='color: #E67E22;'>Upload PDF Files</h2>", unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    with question_container:
        st.markdown("<h2 style='color: #E67E22;'>Ask a Question</h2>", unsafe_allow_html=True)
        user_question = st.text_input("Enter your question:")
        if user_question:
            user_input(user_question)

    with mindmap_container:
        st.markdown("<h2 style='color: #E67E22;'>Generate a Mind Map</h2>", unsafe_allow_html=True)
        is_mindmap = st.text_input("Enter a topic for the mind map:")
        if is_mindmap:
            generate_mindmap(is_mindmap)

    with quizlet_container:
        st.markdown("<h2 style='color: #E67E22;'>Test Your Knowledge</h2>", unsafe_allow_html=True)
        is_quizlet = st.text_input("Enter a topic for the quizlet:")
        num_mcq = st.number_input("Number of Multiple Choice Questions", min_value=1, max_value=10, value=5)
        num_frq = st.number_input("Number of Free Response Questions", min_value=1, max_value=10, value=5)

        if is_quizlet:
            mcq_questions, frq_questions = generate_quizlet(is_quizlet, num_mcq, num_frq)

            if mcq_questions:
                st.markdown("<h3 style='color: #27AE60;'>Multiple Choice Questions</h3>", unsafe_allow_html=True)
                for question, data in mcq_questions.items():
                    options = data["options"]
                    answer = data["answer"]
                    st.markdown(f"<p>{question}</p>", unsafe_allow_html=True)
                    user_answer = st.radio("Select your answer", options, key=question)
                    if st.button("Check Answer", key=f"check_{question}"):
                        check_mcq_answer(user_answer, answer)

            if frq_questions:
                st.markdown("<h3 style='color: #27AE60;'>Short Answer Questions</h3>", unsafe_allow_html=True)
                for question, answer in frq_questions.items():
                    st.markdown(f"<p>{question}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p>Answer: {answer}</p>", unsafe_allow_html=True)

    with study_guide_container:
        st.markdown("<h2 style='color: #E67E22;'>Generate a Study Guide</h2>", unsafe_allow_html=True)
        is_study_guide = st.text_input("Enter a topic for the study guide:")
        if is_study_guide:
            study_guide = generate_study_guide(is_study_guide)
            st.markdown(f"<div style='background-color: #F2F2F2; padding: 10px; border-radius: 5px;'>{study_guide}</div>", unsafe_allow_html=True)
    with cheatsheet_container:
        st.markdown("<h2 style='color: #E67E22;'>Generate a Cheatsheet</h2>", unsafe_allow_html=True)
        is_cheatsheet = st.text_input("Enter a topic for the cheatsheet:")
        num_pages = st.number_input("Number of Pages", min_value=1, max_value=10, value=1)

        if is_cheatsheet:
            cheatsheet = generate_cheatsheet(is_cheatsheet, num_pages)
            st.markdown(f"<div style='background-color: #F2F2F2; padding: 10px; border-radius: 5px;'>{cheatsheet}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 