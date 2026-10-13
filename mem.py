from mem0 import Memory # setting up client by importing Memory
from openai import OpenAI
import json

# creating a configuration
config = {
    "version": "v1.1",
    "embedder": {
        "provider": "ollama",
        "config": {  
            "model": "bge-m3:latest",
            "ollama_base_url": "http://127.0.0.1:11434"
            }
    },

    "llm": { # llm to extract out the memories
        "provider": "ollama",
        "config": {
            "model": "qwen2.5:7b",
            "ollama_base_url": "http://127.0.0.1:11434"

        }

    },

    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "127.0.0.1",
            "port": 6333,
            "collection_name": "mem0_bge_m3",
            "embedding_model_dims": 1024
        }

    }
}



# using all the above configurations creating a memory client
mem_client = Memory.from_config(config)

# setup llm client for chatting (pointing to ollama)
llm_client = OpenAI(
    base_url = "http://127.0.0.1:11434/v1", # Ollama provides an OpenAI-compatible API on the /v1 endpoint
    api_key = "ollama"
)


# continueing the chat
while True:
    user_query = input(">> ") # taking input from user (chat and save memory)

    # retrieve elevant memory for this user
    search_memory = mem_client.search(
        query=user_query,
        filters={"user_id": "Dheeru"}
        )

    # Extract memories safely
    results = search_memory.get("results", [])

    memories = [
    {
        "id": mem.get("id"),
        "memory": mem.get("memory")
    }
    for mem in results
    if mem.get("memory")
]

    print("Found Memories", memories)

    # generating a system prompt
    SYSTEM_PROMPT = f""" 
       You are a helpful AI assistant with persistent memory.
       Use the following saved memories about the user when relevant:
       {json.dumps(memories)}
       
       Instructions:
        - Use relevant memories to personalize your answers.
        - If the answer is not available in memory, answer normally.
        - Never invent facts about the user.
        - Treat the user's current message as the primary request.

    """

    # generate response using local LLM
    response = llm_client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content":  SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
)

    

    ai_response = response.choices[0].message.content

    print("AI:", ai_response)

    # saving the interaction to memory
    mem_client.add(
    user_id = "Dheeru",
    messages=[
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": ai_response}
    ]
)


    print("Memory has been saved...")