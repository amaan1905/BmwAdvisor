from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os


# Retrieving API key
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
llm = OpenAI(api_key=api_key)

# SQLAlchemy setup
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///conversations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'supersecretkey'
db = SQLAlchemy(app)

#database columns
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()

with open('pdf-text.txt', 'r', encoding='utf8') as file:
    raw_text = file.read()

# Setting parameters for chunks
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()

# Vectors
document_search = FAISS.from_texts(texts, embeddings)

qa_chain = load_qa_chain(llm, chain_type='stuff')

prompt = open('pdf-text.txt', 'r', encoding='utf-8').read()

bmw_assistant_template = prompt + """ You are a manager for a BMW branch, you can advise on anything 
related to the BMW 3 series. This includes any general questions about the BMW M3. You do not provide
information on any other cars. If a question is not about a BMW, respond with 'Sorry, I can't help with that.'
Question: {question}
Answer: 
"""

bmw_assistant_prompt_template = PromptTemplate(
    input_variables=['question'],
    template=bmw_assistant_template
)

# LLM params
llm_chat = OpenAI(model='gpt-3.5-turbo-instruct',
                  temperature=0)

# Link prompt and LLM
llm_chain = bmw_assistant_prompt_template | llm_chat

#set conversation to conversation.db
def store_conversation(session_id, message, response):
    conversation = Conversation(session_id=session_id, message=message, response=response)
    db.session.add(conversation)
    db.session.commit()

#get
def get_conversation_history(session_id):
    return Conversation.query.filter_by(session_id=session_id).order_by(Conversation.timestamp).all()

# Find related chunks and provide a response
def query_llm(question, session_id):
    # Fetch conversation history
    history = get_conversation_history(session_id)
    history_messages = "\n".join([f"User: {conv.message}\nBot: {conv.response}" for conv in history])
    
    # Add history to context
    context = f"Conversation history:\n{history_messages}\n\nCurrent question:\n{question}"
    
    # Find relevant chunks
    docs = document_search.similarity_search(question)
    qa_response = qa_chain.run(input_documents=docs, question=context)
    
    # Store the conversation
    store_conversation(session_id, question, qa_response)

    return qa_response

@app.route('/')
def index():
    return render_template('index.html')

#check for existing session, if not make a new one.
@app.route('/chatbot_pdf', methods=['POST'])
def chatbot_pdf():
    data = request.get_json()
    question = data['question']
    session_id = data.get('session_id')
    if not session_id:
        session_id = os.urandom(24).hex()
        session['session_id'] = session_id
    response = query_llm(question, session_id)
    return jsonify({'response': response, 'session_id': session_id})

if __name__ == '__main__':
    app.run(debug=True)
