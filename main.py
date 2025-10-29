
from retriever import Retriever
from augmentation import Augmentation
from generation import Generation

TALK_ID = "sandeco-chat-001"

retriever = Retriever(collection_name="synthetic_dataset_papers")
augmentation = Augmentation(talk_id=TALK_ID)
generation = Generation(model="gemini-2.5-flash")

while True:
    user_query = input("👤 Você: ")
    
    if user_query.lower() == 'sair':
        print("👋 Até a próxima!")
        break

    chunks =   retriever.search(user_query, n_results=10, show_metadata=False)
    prompt =   augmentation.generate_prompt(user_query, chunks)        
    response = generation.generate(prompt)

    augmentation.add_memory(response)

    print(response)

