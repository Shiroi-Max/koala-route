"""
Módulo de evaluación de escenarios para sistemas RAG.

Este módulo define la clase `Evaluator`, encargada de medir la calidad de las respuestas generadas
por el sistema, tanto a nivel de recuperación como de generación. Es utilizado para comparar
automáticamente las salidas del sistema con respuestas de referencia en escenarios de prueba definidos.

Funcionalidades:
- `recall_at_k`: Evalúa la calidad de la recuperación mediante la métrica de Recall adaptativo,
  ajustando dinámicamente el valor de `k` según el número de documentos recuperados.
- `semantic_similarity`: Calcula la similitud semántica entre la respuesta generada y una respuesta
  de referencia utilizando embeddings y similitud coseno.
- `evaluate_scenario`: Evalúa un escenario completo, devolviendo un diccionario con las métricas
  obtenidas.

Requisitos:
- El escenario debe incluir los campos `generated_response` y `reference_response` para evaluar
  la coherencia semántica.
- Si `evaluate` es `True`, también se requieren `expected_relevant_docs` y `retrieved_docs`.

Dependencias:
- `embeddings`: Objeto de embeddings compartido, inicializado desde `modules.vector`.
- `K_EVAL_THRESHOLD`: Proporción configurable de documentos usados como top-K en el cálculo de recall.
"""


from math import ceil
from typing import Dict, List, Set

from sklearn.metrics.pairwise import cosine_similarity

from config.config import K_EVAL_THRESHOLD
from modules.vector import embeddings


class Evaluator:
    def __init__(self):
        """
        Inicializa el evaluador utilizando el objeto de embeddings definido en `modules.vector`.
        Este objeto permite convertir texto en vectores para calcular similitud semántica.
        """
        self.embeddings = embeddings

    def recall_at_k(self, relevant_docs: Set[str], retrieved_docs: List[str]) -> float:
        """
        Calcula el recall adaptativo entre los documentos relevantes esperados y los recuperados.

        El valor de k se ajusta dinámicamente según la proporción `K_EVAL_THRESHOLD` y la longitud
        de la lista de documentos recuperados.

        Args:
            relevant_docs (Set[str]): Conjunto de identificadores de documentos relevantes esperados.
            retrieved_docs (List[str]): Lista de identificadores de documentos recuperados por el sistema.

        Returns:
            float: Valor de recall@k, entre 0.0 y 1.0. Si no hay documentos relevantes, retorna 0.0.
        """
        k = max(1, ceil(K_EVAL_THRESHOLD * len(retrieved_docs)))
        top_k = retrieved_docs[:k]
        relevant_retrieved = sum(1 for doc in top_k if doc in relevant_docs)
        total_relevant = len(relevant_docs)
        return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0

    def semantic_similarity(self, generated: str, reference: str) -> float:
        """
        Calcula la similitud semántica entre una respuesta generada y una respuesta de referencia,
        usando embeddings y similitud coseno.

        Args:
            generated (str): Respuesta generada por el sistema.
            reference (str): Respuesta de referencia esperada.

        Returns:
            float: Valor de similitud coseno entre los embeddings, entre -1.0 y 1.0.
        """
        emb_generated = self.embeddings.embed_query(generated)
        emb_reference = self.embeddings.embed_query(reference)
        return cosine_similarity([emb_generated], [emb_reference])[0][0]

    def evaluate_scenario(self, scenario: Dict) -> Dict:
        """
        Evalúa un escenario de prueba comparando los resultados generados con los esperados.

        Si `evaluate` es True, se evalúa tanto la recuperación como la generación.
        Si es False, se omite la métrica de recuperación y generación.

        Args:
            scenario (Dict): Diccionario con la información del escenario, incluyendo campos como:
                - name
                - generated_response
                - reference_response
                - retrieved_docs
                - expected_relevant_docs
                - evaluate (bool)

        Returns:
            Dict: Diccionario con las métricas obtenidas para el escenario, incluyendo:
                - "Escenario": nombre del escenario.
                - "Recall adaptativo": (opcional) resultado de la recuperación.
                - "Coherencia Semántica": similitud entre la respuesta generada y la referencia.
        """
        results = {"Escenario": scenario["name"]}

        if scenario.get("evaluate", True):
            recall = self.recall_at_k(
                set(scenario["expected_relevant_docs"]), scenario["retrieved_docs"]
            )
            results["Recall adaptativo"] = round(recall, 2)

            coherence = self.semantic_similarity(
                scenario["generated_response"], scenario["reference_response"]
            )
            results["Coherencia Semántica"] = round(coherence, 2)

        return results
