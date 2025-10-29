import redis
import json
import time
from typing import List, Dict, Any, Optional

class Memory:
    """
    Uma classe para gerenciar o histórico de conversas de chat no Redis.

    Utiliza um método 'add_memory' que cria ou atualiza uma conversa (upsert),
    resetando a expiração a cada nova mensagem.
    """
    
    DEFAULT_EXPIRATION_SECONDS = 24 * 60 * 60 # 24 horas

    def __init__(self, expiration_seconds: int = DEFAULT_EXPIRATION_SECONDS):
        """
        Inicializa o gerenciador de chat.

        Args:
            redis_client: Uma instância já conectada do cliente Redis.
            expiration_seconds: O tempo em segundos para as chaves expirarem.
        """
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        ping_result = redis_client.ping()
        if ping_result:
            print("🚀 Conectado ao Redis!\n")
        else:
            print("❌ Falha na conexão com o Redis.")
            raise Exception("Falha na conexão com o Redis.")

        self.redis = redis_client
        self.expiration = expiration_seconds
        print(f"ChatManager inicializado. As conversas expirarão em {self.expiration} segundos.")

    def _get_key(self, talk_id: str) -> str:
        """Método auxiliar para gerar a chave padronizada do Redis."""
        return f"conversation:{talk_id}"

    def add_memory(self, talk_id: str, role: str, message: str) -> None:
        """
        Adiciona uma nova mensagem à memória de conversa do usuário.

        Este método realiza uma operação de "upsert":
        - Se a conversa não existe, ela é criada com esta mensagem.
        - Se a conversa já existe, a mensagem é adicionada ao histórico.
        
        A expiração da chave é sempre resetada para o tempo definido no construtor.
        """
        key = self._get_key(talk_id)
        
        # 1. Tenta obter a conversa existente
        existing_history_json = self.redis.get(key)
        
        # 2. Se não existir, começa com uma lista vazia. Se existir, decodifica.
        if existing_history_json:
            history = json.loads(existing_history_json)
        else:
            history = []
            print(f"✅ Nova conversa sendo criada para o usuário '{talk_id}'.")
            
        # 3. Adiciona a nova mensagem
        history.insert(0, {"role": role, "content": message})
        
        # 4. Serializa e salva de volta no Redis, resetando a expiração
        updated_history_json = json.dumps(history)
        self.redis.set(key, updated_history_json, ex=self.expiration)
        
        print(f"💬 Memória de '{talk_id}' atualizada com a mensagem de '{role}'.")

    def get_conversation(self, talk_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Recupera o histórico completo de uma conversa.
        """
        key = self._get_key(talk_id)
        history_json = self.redis.get(key)
        
        if history_json:
            return json.loads(history_json)
        
        return None

    def delete_conversation(self, talk_id: str) -> None:
        """
        Deleta o histórico de uma conversa do Redis.
        """
        key = self._get_key(talk_id)
        if self.redis.delete(key) > 0:
            print(f"🗑️ Conversa para o usuário '{talk_id}' foi deletada.")
        else:
            print(f"ℹ️  Nenhuma conversa para o usuário '{talk_id}' foi encontrada para deletar.")


# --- Script Principal para Demonstrar o Uso da Classe Refatorada ---
if __name__ == "__main__":
    try:


        chat_manager = Memory()
        
        TALK_ID = "sandeco-upsert-test"
        
        # Limpa o ambiente de teste
        chat_manager.delete_conversation(TALK_ID)
        print("-" * 30)

        # Agora, todas as adições são feitas com o mesmo método 'add_memory'
        # A primeira chamada CRIA a conversa
        chat_manager.add_memory(TALK_ID, "user", "Qual o primeiro livro de Isaac Asimov?")
        
        time.sleep(1)
        # As chamadas seguintes ATUALIZAM a conversa
        chat_manager.add_memory(TALK_ID, "system", "O primeiro romance publicado por Isaac Asimov foi 'Pebble in the Sky' (1950).")
        
        time.sleep(1)
        chat_manager.add_memory(TALK_ID, "user", "Obrigado!")

        # Recuperamos e exibimos a conversa final
        conversa_final = chat_manager.get_conversation(TALK_ID)
        if conversa_final:
            key = chat_manager._get_key(TALK_ID)
            ttl = chat_manager.redis.ttl(key)
            print(f"\n--- Conversa Final Recuperada (expira em {ttl}s) ---")
            for msg in conversa_final:
                print(f"  [{msg['role'].upper()}]: {msg['content']}")
            print("-" * 50)

    except redis.exceptions.ConnectionError as e:
        print(f"Falha na conexão com o Redis: {e}")