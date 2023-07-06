# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import warnings
warnings.filterwarnings("ignore")

import os
import textwrap

import langchain
from langchain.llms import HuggingFacePipeline

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline


### Multi-document retriever
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, VectorDBQA
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

import seaborn as sns
import streamlit as st


st.title("ProServGPT")

##### LLM Code #####

class CFG:
    model_name = 'falcon' # wizardlm, llama, bloom, falcon

def get_model(model = CFG.model_name):
    
    print('\nDownloading model: ', model, '\n\n')
    
    if CFG.model_name == 'wizardlm':
        tokenizer = AutoTokenizer.from_pretrained('TheBloke/wizardLM-7B-HF')
        
        model = AutoModelForCausalLM.from_pretrained('TheBloke/wizardLM-7B-HF',
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True
                                                    )
        max_len = 1024
        task = "text-generation"
        T = 0
        
    elif CFG.model_name == 'llama':
        tokenizer = AutoTokenizer.from_pretrained("aleksickx/llama-7b-hf")
        
        model = AutoModelForCausalLM.from_pretrained("aleksickx/llama-7b-hf",
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                    )
        max_len = 1024
        task = "text-generation"
        T = 0.1

    elif CFG.model_name == 'bloom':
        tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
        
        model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1",
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                    )
        max_len = 1024
        task = "text-generation"
        T = 0
        
    elif CFG.model_name == 'falcon':
        tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2")
        
        model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2",
                                                     load_in_8bit=True,
                                                     device_map='auto',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True
                                                    )
        max_len = 1024
        task = "text-generation"
        T = 0        
        
    else:
        print("Not implemented model (tokenizer and backbone)")
        
    return tokenizer, model, max_len, task, T


# tokenizer, model, max_len, task, T = get_model(CFG.model_name)

st.write(CFG.model_name)

##### LLM Code #####

st.header("Update VectorDB")
st.divider()
st.markdown('Documents uploaded here will be embedded into the Vector Database and can be referenced by the LLM model below')
uploaded_files = st.file_uploader("Choose a PDF file", accept_multiple_files=True)
def save_uploaded_file(uploaded_file, destination_directory, destination_filename):
    destination_path = os.path.join(destination_directory, destination_filename)
    
    with open(destination_path, 'wb') as destination_file:
        destination_file.write(uploaded_file.read())
    
    # Optionally, you can return the destination path if needed
    return destination_path

for uploaded_file in uploaded_files:
    save_uploaded_file(uploaded_file, "/home/cdsw/cml/", uploaded_file.name )
  
st.empty()
st.empty()
st.empty()
    

st.header("ProServGPT")    
st.divider()
with st.chat_message("user"):
    st.write("Hello 👋")
        
prompt = st.chat_input("Say something")
if prompt:
    st.write(f"User has sent the following prompt: {prompt}")