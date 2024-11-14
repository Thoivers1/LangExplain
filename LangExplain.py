import os
import fire
import langchain
from llama import Llama
from transformers import AutoTokenizer, TextStreamer, pipeline
from typing import List

from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

#Define CAMEL agent helper class

class CAMELAgent:
    def __init__(
        self,
        system_message: SystemMessage,
        model: ChatOpenAI
    ) -> None:
        self.system_message = system_message
        self.model = model
        self.init_messages()

    def reset(self) -> None:
        self.init_messages()
        return self.stored_messages

    def init_messages(self) -> None:
        self.stored_messages = [self.system_message]

    def update_messages(self, message: BaseMessage) -> List[BaseMessage]:
        self.stored_messages.append(message)
        return self.stored_messages
    
    def step(
        self,
        input_message: HumanMessage,
    ) -> AIMessage:
        messages = self.update_messages(input_message)

        output_message = self.model.invoke(messages)
        self.update_messages(output_message)

        return output_message
    

class ExplainerAgent(CAMELAgent):
    def __init__(self, system_message, model):
        super().__init__(system_message, model)

    def evaluate_conversation(self, conversation_history):
        # Combine the conversation history into a single string
        conversation_text = "\n".join([msg.content for msg in conversation_history])
        
        # Create the evaluation prompt
        evaluation_prompt = f"{self.system_message.content}\n\nConversation History:\n{conversation_text}\n\nEvaluation:"

        # Use the model to generate the evaluation
        evaluation = self.model.invoke(evaluation_prompt)
        return evaluation


#Setup OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

# Initialize the Llama 2 model
llama2_model = Llama.build(
    ckpt_dir= "",
    tokenizer_path="",
    max_seq_len=5120,
    max_batch_size=4
)

# LLama2 prompt format
def format_llama_prompt(system_prompt, user_message):
    formatted_prompt = f"""
        <s>[INST] <<SYS>>
        { system_prompt }
        <</SYS>>

        { user_message } [/INST]
        """

    return formatted_prompt

#Assign agent roles and task 
assistant_role_name = "AI assistant"
user_role_name = "User"
task = ""
word_limit = 50  # word limit for task brainstorming

#Create inception prompts for User, Assistant, and explainer

user_inception_prompt = """

As {user_role_name}, your primary role is to seek information by posing clear and concise questions directly related to the {task}. Your interactions should strictly involve asking questions to gather specific insights or data pertinent to the task at hand.

- Focus solely on asking questions to deepen your understanding of the task and topic.
- Avoid providing your own insights, analyses, or additional information.
- Avoid repeating the same question. 

Continue to ask follow-up questions based on the assistant's responses to delve deeper into the topic. Once you feel all aspects of the task have been thoroughly addressed, conclude the conversation by stating <CAMEL_TASK_DONE>. Use this command only when you believe the task has been sufficiently explored.

"""

assistant_inception_prompt = """

As the {assistant_role_name}, you are here to respond to the {user_role_name}'s queries based on your extensive database of knowledge. You are not to initiate any task but to provide detailed, accurate, and helpful responses to the questions posed by the user.
Your responses should address the queries directly, offering thorough explanations and drawing on a broad range of sources when necessary. If a question falls outside your capability or scope, clearly explain why it cannot be addressed. Your ultimate goal is to aid the user in understanding the task comprehensively.

"""


explainer_inception_prompt = """

As the Explainer, your primary role is to assess the factual accuracy of the information provided by the {assistant_role_name} in response to queries from the {user_role_name}. Your feedback should not only verify the correctness of the responses but also delve into the problem-solving or thought process used by the {assistant_role_name} to arrive at these answers.

Provide a clear explanation of how the {assistant_role_name} reached its conclusions, highlighting any potential issues or strengths in the logic or methodology. Your assessment should be thorough, focusing on both the factual accuracy and the reasoning behind the answers.

This detailed analysis will help ensure that the {assistant_role_name}'s knowledge is both reliable and well-explained.

"""


#Create a helper helper to get system messages for AI assistant and AI user from role names and the task

def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    # Assistant system message
    assistant_sys_template = SystemMessagePromptTemplate.from_template(
        template=assistant_inception_prompt
    )
    assistant_sys_msg = assistant_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    # User system message
    user_sys_template = SystemMessagePromptTemplate.from_template(
        template=user_inception_prompt
    )
    user_sys_msg = user_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    # explainer system message
    explainer = SystemMessagePromptTemplate.from_template(
        template=explainer_inception_prompt
    )
    explainer = explainer_sys_template.format_messages(
        assistant_role_name=assistant_role_name,
        user_role_name=user_role_name,
        task=task,
    )[0]

    return assistant_sys_msg, user_sys_msg, explainer_sys_msg

explainer_sys_template = SystemMessagePromptTemplate.from_template(
    template=explainer_inception_prompt
)
explainer_sys_msg = explainer_sys_template.format_messages(
    assistant_role_name=assistant_role_name,
    user_role_name=user_role_name,
    task=task,
)[0]

explainer_agent = ExplainerAgent(explainer_sys_msg, ChatOpenAI(model="gpt-4", temperature=0.2))
explainer_agent2 = ExplainerAgent(explainer_sys_msg, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2))

#Create AI assistant agent and AI user agent from obtained system messages

assistant_sys_msg, user_sys_msg, explainer_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, task)

assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model= "gpt-4", temperature=0.2))
user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model="gpt-4", temperature=0.2))

#assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model= "gpt-3.5-turbo", temperature=0.6))
#user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2))


# Reset agents
assistant_agent.reset()
user_agent.reset()

# Initialize chats
user_msg = HumanMessage(
    content=(
        f"{user_sys_msg.content}. "
        "Now start to give me introductions one by one. "
        "Only reply with Instruction and Input."
    )
)

assistant_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
assistant_msg = assistant_agent.step(user_msg)

#Start role-playing session to solve the task!

print(f"Original task prompt:\n{task}\n")

chat_turn_limit, n = 10, 0
conversation_history = []


# Initial user message explicitly stating the task
initial_user_msg = HumanMessage(content=f"{task}")
conversation_history.append(initial_user_msg)
print(f"AI User ({user_role_name}):\n\n{initial_user_msg.content}\n\n")

# First response from the assistant to the task
initial_assistant_msg = assistant_agent.step(initial_user_msg)
conversation_history.append(initial_assistant_msg)
print(f"AI Assistant ({assistant_role_name}):\n\n{initial_assistant_msg.content}\n\n")

while n < chat_turn_limit:  
    n += 1
    
    explainer_feedback = explainer_agent.evaluate_conversation(conversation_history)
    print(f"OpenAI (4,0) Explainer Feedback on Conversation: {explainer_feedback} \n \n")

    explainer_feedback = explainer_agent2.evaluate_conversation(conversation_history)
    print(f"OpenAI (3,5) Explainer Feedback on Conversation: {explainer_feedback} \n \n")

    response = llama2_model.text_completion(
    [format_llama_prompt(explainer_inception_prompt, conversation_history)],
    max_gen_len=500,  # Max generation length
    temperature=0.8, # Temperature for generation
    top_p=0.95        # Top-p sampling parameter
)

    #Print the response
    print(" \n Llama2 Explainer Feedback on Conversation \n \n " + response[0]['generation'])
    
    conversation_history = []

    user_ai_msg = user_agent.step(assistant_msg)
    user_msg = HumanMessage(content=user_ai_msg.content)
    conversation_history.append(user_msg)
    print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")

    assistant_ai_msg = assistant_agent.step(user_msg)
    assistant_msg = HumanMessage(content=assistant_ai_msg.content)
    conversation_history.append(assistant_msg)
    print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
    

    if "<CAMEL_TASK_DONE>" in user_msg.content:
        print("Conversation ended by user signal.")

