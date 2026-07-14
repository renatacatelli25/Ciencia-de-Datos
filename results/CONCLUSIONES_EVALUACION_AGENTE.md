# Conclusiones de la evaluación del agente RAG

**Experimento:** `agent_eval_20260712_170647`  
**Modo:** `api` (OpenAI: embeddings + gpt-4o-mini)  
**Dataset:** 14 preguntas del golden set (`utils/golden_set_rag.json`)  
**Fecha de referencia:** julio 2026

---

## 1. Resumen ejecutivo

El agente **recupera documentos relevantes en la mayoría de los casos**, pero **falla al convertir esa recuperación en una respuesta final correcta** en casi la mitad de las consultas.

| Dimensión | Score | Interpretación en una línea |
|---|---:|---|
| **Retrieval relevance** | 92.9% | Lo que se recupera suele tener relación con la pregunta |
| **Recall@5** | 78.6% | En ~8 de cada 10 preguntas, el producto gold aparece entre los candidatos |
| **MRR** | 0.64 | Cuando acierta, el producto correcto no siempre queda primero |
| **Product hit final** | 57.1% | Solo en ~6 de 14 preguntas el pipeline entrega el producto esperado |
| **Correctness** | 42.9% | Menos de la mitad de las respuestas coinciden con la referencia |
| **Fallback rate** | 28.6% | 4 de 14 consultas terminan en *"No encontré productos relevantes"* |

**Conclusión principal:** el cuello de botella **no está en la búsqueda semántica inicial**, sino en las etapas posteriores — **filtros, grader y generación** — que descartan candidatos válidos o recomiendan productos distintos al gold.

---

## 2. Metodología (breve)

Se evaluó el pipeline completo:

`extract_filters → retrieve (Chroma) → rerank (CrossEncoder) → grade (LLM) → generate (LLM)`

Métricas usadas (framework inspirado en LangSmith):

- **Correctness** — respuesta vs. ground truth
- **Relevance** — respuesta vs. pregunta
- **Groundedness** — respuesta vs. contexto recuperado
- **Retrieval relevance** — documentos recuperados vs. pregunta
- **Recall@5 / MRR / product_hit** — métricas determinísticas con IDs gold

---

## 3. Análisis por capa (funnel)

```
Retrieve (recall@5 = 78.6%)
    ↓ pierde ~14 pp
Rerank (product_hit_reranked = 64.3%)
    ↓ pierde ~7 pp
Grade + Generate (product_hit_final = 57.1%)
    ↓ pierde ~14 pp
Correctness final = 42.9%
```

### 3.1 Retrieval — fortaleza relativa

- **92.9%** de retrieval relevance (LLM-judge) indica que los documentos traídos por Chroma **sí contienen información relacionada** con la pregunta.
- **Recall@5 = 78.6%** confirma que el producto correcto **entra al embudo** en la mayoría de los casos.
- **MRR = 0.64** muestra que, cuando el producto gold aparece, **a menudo no queda en la primera posición**, lo que complica el rerank y el grade.

**Lectura:** el índice `chroma_db_productos_openai` y la búsqueda semántica cumplen un rol aceptable como primera etapa.

### 3.2 Rerank — mejora moderada pero insuficiente

- **Product hit reranked (64.3%)** > recall bruto en algunos casos, pero sigue lejos del 100%.
- El CrossEncoder **reordena**, pero no garantiza que el producto gold quede en el top final que ve el grader.

### 3.3 Grade + Generate — principal punto débil

- **Fallback rate = 28.6%:** en 4 preguntas el agente **no devolvió ningún producto**, a pesar de que en al menos 2 de esas el retrieval sí trajo documentos relevantes (según el evaluador de retrieval).
- **Groundedness = 57.1%:** cuando responde, casi la mitad de las veces **introduce afirmaciones no respaldadas** por el contexto (ej.: fechas sin info nutricional, filtros de rating mal aplicados).
- **Correctness = 42.9%** < **Relevance = 64.3%:** el agente **a menudo responde algo pertinente a la pregunta**, pero **no el producto correcto** ni los hechos de la referencia.

---

## 4. Resultados por pregunta

### ✅ Aciertos claros (6/14 — correctness = 1)

| Pregunta | Qué funcionó |
|---|---|
| Dark chocolate smooth | Recuperó, rankeó y recomendó el producto gold |
| Honey mustard mayo | Respuesta precisa y anclada al contexto |
| Dog food sin subproductos avícolas | Identificó el producto y sus atributos clave |
| Cornbread tipo cake | Recomendación directa y correcta |
| Dal Makhani spice mix | Match exacto con el mix gold |
| Beef jerky honey chipotle | Producto y atributos correctos |

Estas consultas comparten un patrón: **pregunta específica + producto con señal fuerte en la descripción**.

### ⚠️ Respuesta útil pero producto incorrecto (3/14)

| Pregunta | Problema |
|---|---|
| Coconut oil for sweet dishes | Recomendó Nutiva en lugar de Nature's Way; respuesta relevante pero incorrecta |
| Healthier puffed chips | Recomendó Sweet Potato Chips en lugar de Popchips |
| Dates quality/packaging | Acertó el producto gold, pero el LLM negó info nutricional que el judge considera presente → groundedness = 0 |

Aquí el agente **no falla en retrieval** (recall = 1), pero **elige otro producto** o **redacta detalles imprecisos**.

### ❌ Fallos totales — fallback (4/14)

| Pregunta | Qué pasó |
|---|---|
| Oatmeal cookies taste/texture | Fallback; recall = 0 — no entró el producto gold al pipeline |
| Dog treats for coughing | Fallback; recall = 0 — Greenies no matcheó la consulta sobre tos |
| Manuka honey for tea | **Recall = 1, MRR = 1**, pero el grader rechazó todo → fallback injustificado |
| Wrong brand/vendor complaints | Retrieval relevance = 1, pero grader no dejó pasar candidatos → fallback |

Estos casos muestran **dos tipos de error distintos**:
1. **Retrieval miss** (galletas, dog treats): la pregunta no alinea con las descripciones indexadas.
2. **Grader demasiado estricto** (manuka, wrong brand): el producto gold estaba disponible pero se descartó.

### ⚠️ Falso negativo por filtros (1/14)

| Pregunta | Problema |
|---|---|
| Coconut oil organic for popcorn | El agente aplicó filtros (`organic`, `rating_min=4`) demasiado restrictivos, descartó Nutiva (producto gold) y respondió que no había opciones válidas — a pesar de tener el producto en `graded_ids` |

---

## 5. Conclusiones

### 5.1 Sobre la calidad global

1. **El agente es parcialmente funcional:** en consultas concretas y bien cubiertas por las descripciones (condimentos, mixes, carnes secas, chocolates), alcanza **respuestas correctas y fundamentadas**.
2. **No es confiable como sistema general:** con **42.9% de correctness** y **28.6% de fallback**, un usuario real recibiría respuestas erróneas o vacías con frecuencia inaceptable para producción.
3. **Hay una brecha retrieve → respuesta:** el retrieval es bueno (78.6% recall), pero la calidad final cae a 57.1% product hit y 42.9% correctness. **El valor se pierde después del retrieve.**

### 5.2 Sobre cada componente

| Componente | Veredicto |
|---|---|
| **Índice Chroma + embeddings OpenAI** | Aceptable; base sólida para iterar |
| **CrossEncoder rerank** | Ayuda, pero no cierra la brecha solo |
| **Grader LLM** | Demasiado conservador; genera falsos negativos (manuka, wrong brand) |
| **Extract filters** | Puede bloquear respuestas válidas (filtro de rating en aceite de coco) |
| **Generate LLM** | Mejor cuando el grader pasa el producto correcto; tiende a recomendar alternativas plausibles pero incorrectas |

### 5.3 Sobre las métricas LLM vs. determinísticas

- **Retrieval relevance (92.9%)** es más optimista que **recall@5 (78.6%)** porque el judge semántico es más permisivo que el match exacto de product ID.
- **Relevance (64.3%) > Correctness (42.9%)** confirma que el agente **suena útil** aunque **no acierte el producto** — riesgo de alucinación aparentemente razonable.
- **Groundedness (57.1%)** advierte que incluso las respuestas "relevantes" **no siempre están ancladas** al contexto recuperado.

---

## 6. Recomendaciones prioritarias

### Corto plazo (impacto alto, esfuerzo bajo)

1. **Relajar o auditar el grader:** si recall = 1 y el grader deja 0 candidatos, forzar fallback solo tras revisar el top-1 del rerank.
2. **Suavizar filtros de rating/marca:** no descartar productos con `avg_score = 0.0` cuando el dataset no tiene ratings confiables.
3. **Evitar fallback cuando hay candidatos rerankeados:** usar al menos el top-1 del CrossEncoder si el grader falla.

### Mediano plazo

4. **Mejorar recall en consultas “atributo + producto”** (galletas de avena, dog treats para tos): enriquecer descripciones o ajustar chunking/indexación.
5. **Reportar product_id en la respuesta final** para facilitar verificación automática.
6. **Separar métricas por tipo de pregunta** (sabor, uso, queja, filtro explícito) para ver dónde falla cada familia.

### Para el informe del TP

7. Presentar el **funnel retrieve → rerank → grade → generate** como figura central.
8. Destacar que **RAG no termina en el retrieval:** un recall del 78% puede convertirse en 43% de correctness si el pipeline downstream filtra de más.
9. Usar los **6 aciertos** como evidencia de viabilidad y los **4 fallbacks** como evidencia de trabajo pendiente.

---

## 7. Tabla resumen final

| Métrica | Valor | % equivalente |
|---|---:|---:|
| Retrieval relevance | 0.929 | 92.9% |
| Recall@5 | 0.786 | 78.6% |
| MRR | 0.639 | — |
| Product hit reranked | 0.643 | 64.3% |
| Product hit final | 0.571 | 57.1% |
| Relevance | 0.643 | 64.3% |
| Groundedness | 0.571 | 57.1% |
| Correctness | 0.429 | 42.9% |
| Fallback rate | 0.286 | 28.6% |
| **N preguntas** | **14** | — |

---

## 8. Frase de cierre sugerida para la presentación

> *"El agente demuestra que la arquitectura RAG con rerank y grading puede recomendar productos correctamente en consultas específicas (43% de correctness global, 93% de retrieval relevance), pero pierde valor en el grader y en la generación: casi 3 de cada 10 preguntas devuelven fallback y, cuando responde, a menudo elige un producto alternativo plausible pero incorrecto. El próximo paso no es reindexar, sino afilar el pipeline downstream."*
