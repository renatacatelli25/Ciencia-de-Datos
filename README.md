# Ciencia-de-Datos

Proyecto de ciencia de datos con agente de búsqueda de productos sobre reviews de Amazon.

## Cómo clonar y ejecutar

### 1. Clonar con Git LFS

`Reviews.csv` (~287 MB) se versiona con Git LFS. Necesitás tener [Git LFS](https://git-lfs.com/) instalado:

```bash
git lfs install
git clone <url-del-repo>
cd Ciencia-de-Datos
git lfs pull
```

Si ya clonaste el repo sin LFS:

```bash
git lfs install
git lfs pull
```

### 2. Entorno Python

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 3. Variables de entorno

Copiá el ejemplo y completá tu API key de OpenAI:

```bash
copy .env.example .env
```

### 4. Artefactos incluidos (no hace falta regenerarlos)

Al clonar deberías tener:

| Archivo / carpeta | Para qué sirve |
|---|---|
| `chroma_db_productos_openai/` | Índice vectorial del agente |
| `descripciones_checkpoint.csv` | Descripciones de productos ya generadas |
| `product_names_checkpoint.csv` | Nombres inferidos de productos |
| `src/data/Reviews.csv` | Dataset completo (Git LFS) |

Con el checkpoint de descripciones, **`agent.py` no necesita `Reviews.csv`** para cargar el catálogo. El CSV sigue siendo necesario si querés re-ejecutar el notebook desde cero.

### 5. Ejecutar el agente

```bash
python -m src.agent
```

---

#### Ramas sugeridas

1. Preprocessing
2. Entrenamiento del modelo
3. Predicciones
4. Otros...

Formato de commits consistente para que todos hablemos el mismo idioma.
