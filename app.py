import time
import streamlit as st
# from dotenv import dotenv_values
from dotenv import load_dotenv
import os
import pickle
from groq import Groq
from PyPDF2 import PdfReader
# from streamlit_extras.add_vertical_space import add_vertical_space
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.VectorStores import FAISS
# from langchain.chains import ConversationChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate

def chat_with_groq(client, prompt, model):
    """
    This function sends a prompt to the Groq API and retrieves the AI's response.

    Parameters:
    client (Groq): The Groq API client.
    prompt (str): The prompt to send to the AI.
    model (str): The AI model to use for the response.

    Returns:
    str: The content of the AI's response.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return completion.choices[0].message.content

@st.cache_data()
def get_summarization(_client, user_doc, model, language_option=None):
    """
    This function generates a summarization prompt based on the user's document.
    It then sends this summarization prompt to the Groq API and retrieves the AI's response.

    Parameters:
    client (Groqcloud): The Groq API client.
    user_doc (str): The user's document.
    model (str): The AI model to use for the response.

    Returns:
    str: The content of the AI's response to the summarization prompt.
    """
    if language_option == 'None' or language_option == 'English':
        prompt = '''
        A user have uploaded a insurance document.:
    
        {user_doc}
    
        In a few sentences, summarize the data in 10 most important points.
        '''.format(user_doc=user_doc)
    elif language_option == 'Hindi':
        prompt = '''
            A user have uploaded a insurance document.:
    
            {user_doc}
    
            In a few sentences, summarize the data in hindi language in 10 most important points.
            '''.format(user_doc=user_doc)

    return chat_with_groq(_client, prompt, model)

def get_answers(client, user_doc, question, model, language_option=None):
    """
    This function generates a question answer prompt based on the user's question and uploaded document.
    It then sends this question answer prompt to the Groq API and retrieves the AI's response.

    Parameters:
    client (Groqcloud): The Groq API client.
    user_doc (str): The user's document.
    question (str): The user's question.
    model (str): The AI model to use for the response.

    Returns:
    str: The content of the AI's response to the question answer prompt.
    """
    # Ensure user_doc is not empty and question is provided
    if user_doc and question:
        if language_option == 'None' or language_option == 'English':
            prompt = f'''
            A user has uploaded an insurance document and asked the following question:
            
            {question}
            
            Please provide an answer based on the attached document:
            
            {user_doc}
        '''
        # Send the prompt to the Groq API using the specified model
            response = chat_with_groq(client, prompt, model)
            return response
        elif language_option == 'Hindi':
            prompt = f'''
            A user has uploaded an insurance document and asked the following question:

            {question}

            Please provide an answer in hindi language based on the attached document:

            {user_doc}
        '''
            # Send the prompt to the Groq API using the specified model
            response = chat_with_groq(client, prompt, model)
            return response
    else:
        return "Error: Unable to generate answer. Please provide both document and question."
def main():
    # Sidebar contents
    with st.sidebar:
        st.title('ISureScan - Insurance Policy(PDF) Chatbot')
        st.markdown('''
        ## About
        ISureScan.ai offers a revolutionary approach to insurance document management. 
        Leveraging state-of-the-art natural language processing (NLP) and Generative AI (GenAI) technology,
        our app empowers users to effortlessly upload insurance documents and extract essential information.
        Ensure your health coverage! Upload your medical insurance policy in the provided box for personalized insights.
        Need assistance or have questions? Use our AI chatbot for instant support and clarity. 
        Your well-being matters to us!

        ''')
    #    add_vertical_space(5)
        st.write('Made in India :flag-in: by [StirPot](https://stirpot.in/)')

        model = "llama3-8b-8192"

    # Initialize text variable
    text = ""
    language_option = ""

    # Get the Groq API key and create a Groq client
    # groq_api_key = dotenv_values(".env")(st.secrets['GROQ'])
    load_dotenv()
    os.environ['GROQ'] == st.secrets['GROQ']
    groq_api_key = os.environ['GROQ']
    client = Groq(
        api_key=groq_api_key,
        # base_url=st.secrets["GROQ_BASE_URL"]
    )

    st.header("Chat with your Insurance Policy", divider='rainbow')

    language_option = st.radio('Select your preferred Language of interaction',
                               ('English', 'Hindi'), index=0, horizontal=True)

    # upload a pdf file
    pdf = st.file_uploader("Upload your Insurance policy file (pdf only) ", type='pdf')

    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        summarization = get_summarization(client, text, model, language_option)
        with st.spinner(text='In Progress'):
            time.sleep(5)
            st.success(summarization)
        # st.write(summarization)

        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=1000,
        #     chunk_overlap=200,
        #     length_function=len
        #     )
        # chunks = text_splitter.split_text(text=text)

        # embeddings
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS(chunks)
        # st.write(chunks)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                file_contents = pickle.load(f)
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(text, f)

        # memory = ConversationBufferWindowMemory(k=5)
        # if 'chat_history' not in st.session_state:
        #     st.session_state.chat_history = []
        # else:
        #     for message in st.session_state.chat_history:
        #         memory.save_context({'input': message['human']}, {'output': message['AI']})

        # Initialize Groq Langchain chat object and conversation
        # groq_chat = ChatGroq(
        #     groq_api_key=groq_api_key,
        #     model_name=model
        # )

        # conversation = ConversationChain(
        #     llm=groq_chat,
        #     memory=memory
        # )

    # Accept user inputs/questions
    with st.chat_message("user"):
        st.write("Hello 👋")
    query = st.chat_input("Ask questions about your Insurance PDF")
    # If the user has asked a question,
    if query:
        # Get answers using the Groq API based on the user's query and uploaded document
        llm_response = get_answers(client, user_doc=text, question=query, model=model, language_option=language_option)
        st.write("ISureScan:", llm_response)

    #     response = conversation(query)
    #     message = {'human': query, 'AI': response['response']}
    #     st.session_state.chat_history.append(message)
    #     st.write("ISureScan:", response['response'])


if __name__ == '__main__':
    main()
