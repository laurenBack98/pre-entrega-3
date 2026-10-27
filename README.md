# RAG de políticas de Rapanui

Aplicación de ejemplo de generación aumentada por recuperación (RAG). Indexa información sobre Rapanui en Chroma y usa un modelo local servido por Ollama para responder preguntas con el contexto recuperado.

## Requisitos

- Python 3.10 o superior
- [Ollama](https://ollama.com/) instalado y en ejecución
- El modelo `qwen2.5:7b` descargado en Ollama (o el modelo que configures)

## Instalación

Desde la carpeta del proyecto, crea y activa un entorno virtual en PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Descarga el modelo por defecto:

```powershell
ollama pull qwen2.5:7b
```

## Configuración

La aplicación lee estas variables opcionales de un archivo `.env` en la carpeta del proyecto:

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `LLM_MODEL` | `qwen2.5:7b` | Modelo disponible en Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Dirección del servidor Ollama |

Por ejemplo:

```dotenv
LLM_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434
```

## Ejecución

Con Ollama en ejecución y el entorno virtual activado:

```powershell
python main.py
```

El programa ejecuta dos consultas de ejemplo: una sobre información incluida en el contexto y otra sobre información ausente, para mostrar el comportamiento de abstención.

## Datos y persistencia

Los textos que se indexan están definidos en `ingesta.py`. Al construir la base vectorial, el programa los escribe en `data/`, los divide en fragmentos y crea el almacén persistente en `vectorstore/`. En ejecuciones posteriores reutiliza ese almacén si ya contiene archivos.

Las representaciones vectoriales se generan con `sentence-transformers/all-MiniLM-L6-v2` en CPU. La primera ejecución puede tardar más porque descarga el modelo de embeddings y crea el índice.
