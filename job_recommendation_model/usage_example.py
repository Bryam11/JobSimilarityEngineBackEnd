# Ejemplo de uso del modelo de recomendaci�n de empleos
import pickle

# Cargar el modelo
with open('job_recommendation_system.pkl', 'rb') as f:
    model = pickle.load(f)

# Hacer una recomendaci�n
query = "Data Scientist with Python experience"
recommendations = model.recommend_jobs(query, top_n=5, method='hybrid')

# Mostrar resultados
print(f"Recomendaciones para: '{query}'")
for rec in recommendations:
    print(f"{rec['rank']}. {rec['title']} - {rec['company']}")
    print(f"   Score: {rec['similarity_score']:.3f}")
    print(f"   Location: {rec['location']}")
    print()

# Informaci�n del modelo
print("Informaci�n del modelo:")
model_info = model.get_model_info()
for key, value in model_info.items():
    print(f"  {key}: {value}")
