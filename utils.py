# utils.py
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from config import *
import os
from py2neo import Graph
from dotenv import load_dotenv
load_dotenv()