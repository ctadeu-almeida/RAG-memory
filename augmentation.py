# augmentation.py

from memory import Memory as MemoryRAG

class Augmentation:
    def __init__(self, talk_id):
        self.talk_id = talk_id
        self.memory_rag = MemoryRAG()
        self.prompt = ""

    def generate_prompt(self, query_text, chunks):


        separador = "\n\n------------------------\n\n" 

        # Junta os chunks com o separador e adiciona o cabeçalho
        chunks_formatados = f"Conhecimento\n------------------------\n\n{separador.join(chunks)}"

        self.prompt = f"""Responda em pt-br e em markdown, a query do usuário delimitada por <query> 
        usando apenas o conhecimento dos chunks delimitados por <chunks> 
        e tenha em mente o historico das conversas anteriores delimitado por <historico>. 
        Combine as informações para responder a query de forma unificada. A prioridade
        das informações são: query=1, chunks=2, historico=3.

        Se por acaso o conhecimento não for suficiente para responder a query, 
        responda apenas que não temos conhecimento suficiente para responder 
        a Pergunta.

        <chunks>
        {chunks_formatados}
        </chunks>        
        
        <query>
        {query_text}
        </query>

        <historico>
        {self.memory_rag.get_conversation(self.talk_id)}
        </historico>

        """

        return self.prompt
    
    def clear_memory(self):
        self.memory_rag.delete_conversation(self.talk_id)

    def add_memory(self, llm_response):
        try:
            self.memory_rag.add_memory(self.talk_id, "user", self.prompt)
            self.memory_rag.add_memory(self.talk_id, "system", llm_response)
            return True
        except Exception as e:
            print(f"Erro ao adicionar memória: {str(e)}")
            return False

    def get_conversation(self):
        return self.memory_rag.get_conversation(self.talk_id)