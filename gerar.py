from semantic_encoder import SemanticEncoder

encoder = SemanticEncoder(
    docs_dir="docs", #diretório dos documentos
    chunk_size=5000, #tamanho do chunk
    overlap_size=500, #tamanho da sobreposição
)

# Construir base vetorial
stats = encoder.build(collection_name="synthetic_dataset_papers")

# Imprimir estatísticas
print(stats)



