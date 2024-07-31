from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from flask import Flask, render_template, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import glob
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

# Database columns
class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

with app.app_context():
    db.create_all()

def load_texts(directory):
    texts = []
    for file_path in glob.glob(os.path.join(directory, '*.txt')):
        with open(file_path, 'r', encoding='utf8') as file:
            texts.append(file.read())
    return texts

raw_texts = load_texts(r'C:\Users\amaan\Desktop\python-workspace\chatbot_pdf\textfiles')
all_texts = '\n'.join(raw_texts)

# Setting parameters for chunks
text_splitter = CharacterTextSplitter(
    separator='\n',
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)

texts = text_splitter.split_text(all_texts)

embeddings = OpenAIEmbeddings()

# Vectors
document_search = FAISS.from_texts(texts, embeddings)

qa_chain = load_qa_chain(llm, chain_type='stuff')

def get_model_prompt_template(model):
    if model == 'bmw m3':
        template = """You are a manager for a BMW branch, you can only advise on anything 
        related to the BMW M3. You do not provide information on any other cars including any other BMW models. If a question is not about a BMW M3, respond with 'Sorry, I can't help with that.'
        Question: {question}
        Answer: 
        """
    elif model == 'bmw m5':
        template = """You are a manager for a BMW branch, you can only advise on anything 
        related to the BMW M5. You do not provide information on any other cars including any other BMW models. If a question is not about a BMW M5, respond with 'Sorry, I can't help with that.'
        Question: {question}
        Answer: 
        """
    else:
        template = """Talk about rainbows
        Question: {question}
        Answer: 
        """
    
    return PromptTemplate(
        input_variables=['question'],
        template=template
    )

# LLM params
llm_chat = OpenAI(model='gpt-3.5-turbo-instruct', temperature=0)

# Store conversation in the database
def store_conversation(session_id, message, response):
    conversation = Conversation(session_id=session_id, message=message, response=response)
    db.session.add(conversation)
    db.session.commit()

# Retrieve conversation history from the database
def get_conversation_history(session_id):
    return Conversation.query.filter_by(session_id=session_id).order_by(Conversation.timestamp).all()

# Query the LLM based on the provided question and model
def query_llm(question, session_id, model):
    # Get the appropriate prompt template based on the selected model
    model_prompt_template = get_model_prompt_template(model)
    
    # Prepare the prompt for the model
    prompt = model_prompt_template.format(question=question)
    
    # Ensure the prompt fits within the token limit of the model
    prompt = prompt[:3200]
    
    try:
        # Use the QA chain to get a response based on the prompt
        qa_response = qa_chain.run(input_documents=[], question=prompt)
    except Exception as e:
        qa_response = f"Error generating response: {str(e)}"
    
    
    store_conversation(session_id, question, qa_response)

    return qa_response

@app.route('/')
def index():
    return render_template('index.html')

# Check for existing session; if not, create a new one
@app.route('/chatbot_pdf', methods=['POST'])
def chatbot_pdf():
    data = request.get_json()
    question = data['question']
    model = data.get('model', 'BMW')
    session_id = data.get('session_id')
    if not session_id:
        session_id = os.urandom(24).hex()
        session['session_id'] = session_id
    response = query_llm(question, session_id, model)
    return jsonify({'response': response, 'session_id': session_id})

if __name__ == '__main__':
    app.run(debug=True)
