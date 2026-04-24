from ...base_agent import BaseAgent

import time

import argparse

from ....utils import Message

from pathlib import PurePosixPath
import os
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import PromptTemplate
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from openagi.src.agents.base import BaseAgent

class RAGAgent(BaseAgent):
    def __init__(self,
                 agent_name,
                 task_input,
                 llm,
                 agent_process_queue,
                 agent_process_factory,
                 log_mode: str
        ):
        BaseAgent.__init__(self, agent_name, task_input, llm, agent_process_queue, agent_process_factory, log_mode)

    def run(self):
        request_waiting_times = []
        request_turnaround_times = []
        query = self.task_input

        self.logger.log(f"{query}\n", level="info")

        context = self.retrive(query)
        prompt = self.build_prompt(context_str=context, query_str=query)

        rounds = 0

        response, start_times, end_times, waiting_times, turnaround_times = self.get_response(
            message = Message(
                prompt = prompt,
                tools = None
            )
        )

        self.set_start_time(start_times[0])
        rounds += 1

        request_waiting_times.extend(waiting_times)
        request_turnaround_times.extend(turnaround_times)

        response_message = response.response_message

        self.logger.log(f"Final result is: {response.response_message}\n", level="info")
        final_result = response_message

        self.set_status("done")
        self.set_end_time(time=time.time())

        return {
            "agent_name": self.agent_name,
            "result": final_result,
            "rounds": rounds,
            "agent_waiting_time": self.start_time - self.created_time,
            "agent_turnaround_time": self.end_time - self.created_time,
            "request_waiting_times": request_waiting_times,
            "request_turnaround_times": request_turnaround_times,
        }

    def retrive(self, query: str):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.data_path = PurePosixPath(script_dir, "data", "paul_graham").as_posix()
        self.db_path = PurePosixPath(script_dir, "chroma_db").as_posix()
        self.collection_name = "quickstart"
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

        self.create_db_if_not_exists()

        db = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = db.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self.embed_model,
        )

        retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
        retrieved_result = retriever.retrieve(query)
        context_str = retrieved_result[0].get_content()
        return context_str

    def create_db_if_not_exists(self):
        if os.path.exists(self.db_path):
            pass
        else:
            print("store documents to vector db!")
            documents = SimpleDirectoryReader(self.data_path).load_data()

            chroma_client = chromadb.PersistentClient(path=self.db_path)
            chroma_collection = chroma_client.create_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context, embed_model=self.embed_model
            )
            index.storage_context.persist(persist_dir=self.db_path)

    def build_prompt(self, context_str: str, query_str: str):
        prompt_template_literal = (
            "{query_str}"

            "Use the following context as your learned knowledge, inside <context></context> XML tags.\n"
            "<context>\n"
                "{context_str}"
            "</context>\n"

            "Avoid mentioning that you obtained the information from the context.\n"
        )
        prompt_template = PromptTemplate(prompt_template_literal)
        final_prompt = prompt_template.format(context_str=context_str, query_str=query_str)
        return final_prompt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run RagAgent')
    parser.add_argument("--agent_name")
    parser.add_argument("--task_input")

    args = parser.parse_args()
    agent = RAGAgent(args.agent_name, args.task_input)
    agent.run()
