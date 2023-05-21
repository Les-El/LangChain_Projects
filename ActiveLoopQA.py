from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackManager
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import re

class ActiveLoopQAChain:
    def __init__(
        self,
        dataset_path: str,
        openai_api_key: str,
        llm=None,
        history=None,
        callback_manager=None,
        distance_metric="cos",
        fetch_k=10,
        maximal_marginal_relevance=False,
        k=10,
        verbose=False,
    ):
        """
        ActiveLoopQAChain class for question-answering using Deep Lake.
        Takes a str variable as input and returns a str variable as output.
        USAGE: Do not tell the model to refer to the dataset; it doesn't understand that it's a dataset. Just assume your dataset is already context that the model can refer to.

        Example usage:
        # Instantiate the LLM
        langchain_llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.15, openai_api_key=OPENAI_API_KEY)
        # Instantiate the QA chain
        langchain_qa = ActiveLoopQAChain(dataset_path='hub://activeloopuser/xyzdataset', openai_api_key=OPENAI_API_KEY, llm=langchain_llm, verbose=True)
        # Ask a question of the chain
        langchain_qa.run('Give me a broad overview of the XYZ trends in 2023.')
        # Wrap the chain in a Tool so it can be used by Agents
        langchain_tool = Tool(name='LangChain Codebase QA', func = langchain_qa.run, description='A tool for answering questions about the LangChain codebase.')

        References:
        https://python.langchain.com/en/latest/use_cases/code/code-analysis-deeplake.html
        https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/deeplake.html

        Args:
            dataset_path (str): The path to the dataset in Deep Lake.
            openai_api_key (str): The API key for OpenAI.
            llm: An instance of a language model, defaults to ChatOpenAI with 'gpt-3.5-turbo' model and temperature 0.15.
            history: An object to hold chat history, defaults to None.
            callback_manager: An instance of a callback manager, defaults to None.
            distance_metric (str, optional): The distance metric used for determining similarity between vectors. Distance function 'L2' for Euclidean, 'L1' for Nuclear, 'Max l' for infinity distance, 'cos' for cosine similarity, 'dot' for dot product; defaults to 'cos'.
            fetch_k (int, optional): The number of most similar vectors to fetch from the Vector Store, defaults to 10.
            maximal_marginal_relevance (bool, optional): Whether to use Maximal Marginal Relevance for diversity in responses, defaults to False.
            k (int, optional): The number of most similar vectors to consider when processing a query, defaults to 10.
            verbose (bool, optional): Whether to print additional information during execution, defaults to False.
        """

        # Initialize objects
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.db = DeepLake(
            dataset_path=dataset_path,
            read_only=True,
            embedding_function=self.embeddings,
        )
        self.retriever = self.db.as_retriever()
        self.retriever.search_kwargs = {
            "distance_metric": distance_metric,
            "fetch_k": fetch_k,
            "maximal_marginal_relevance": maximal_marginal_relevance,
            "k": k,
        }

        # Initialize LLM and QA chain
        if llm is None:
            llm = ChatOpenAI(
                openai_api_key=openai_api_key,
                model_name="gpt-3.5-turbo",
                temperature=0.15,
            )
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.retriever,
            callback_manager=callback_manager,
            verbose=verbose,
        )

        # Initialize chat history
        self.chat_history = history if history is not None else []

    def run(self, question: str) -> str:
        # Preprocess the question
        processed_question = self.preprocess_question(question)

        # Run the QA chain
        result = self.qa_chain(
            {"question": processed_question, "chat_history": self.chat_history}
        )

        # Update chat history and return the answer
        self.chat_history.append((processed_question, result["answer"]))

        # Return the output
        return print(f"**Answer**: {result['answer']} \n")
    
    @staticmethod
    def preprocess_question(question: str) -> str:
        # Remove non-standard characters and emojis
        question = re.sub(r"[^\w\s]", "", question)
        return question.strip()
