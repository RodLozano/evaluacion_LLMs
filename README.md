# Asistente Turístico con LLM (RAG + Multiturno + Function Calling)

Proyecto final de la asignatura **Large Language Models**  
Máster en Inteligencia Artificial, Cloud Computing y DevOps.

Este repositorio contiene un **prototipo reproducible de asistente turístico** construido sobre un **LLM comercial (OpenAI)** que integra:

- **RAG (Retrieval-Augmented Generation)** sobre una guía turística en PDF.
- **Diálogo multiturno** con gestión explícita de memoria y control de longitud del contexto.
- **Function calling obligatorio** mediante la herramienta `get_weather(fecha)`, con manejo de errores y logging.
- **Evaluación básica** mediante prompts reproducibles y visualización de métricas.

Todo el flujo está integrado y orquestado desde **un único notebook principal**.

---
---

## Estructura del repositorio

``` text
.
├─ notebooks/
│  └─ main.ipynb              # Notebook principal con todo el flujo del proyecto
├─ data/
│  └─ guia_turistica.pdf      # Guía turística proporcionada por el profesorado
├─ src/
│  ├─ rag.py                  # Chunking, embeddings, vector store y retrieval
│  ├─ llm.py                  # Cliente del LLM y configuración de parámetros
│  ├─ memory.py               # Gestión del historial y control de longitud de contexto
│  ├─ tools.py                # Definición de get_weather (schema, ejecución y errores)
│  └─ eval.py                 # Prompts de evaluación, métricas y visualizaciones
├─ logs/
│  └─ tool_calls.log          # Registro de llamadas a la función get_weather
├─ requirements.txt           # Dependencias del proyecto
├─ .env.example               # Ejemplo de variables de entorno (API keys)
├─ .gitignore
└─ README.md
```

---

## ⚙️ Requisitos

- Python **3.10+**
- Acceso a la **API de OpenAI** (facturación activa)
- Entorno probado en Windows con **VS Code + Jupyter**

---

Entregables requeridos:
- **Repositorio** con estructura clara, README detallado, `requirements.txt` o `pyproject.toml` y `.gitignore`.
- **Notebook principal** con celdas ordenadas (carga → indexación → conversación → pruebas → análisis).
- **Informe final** (diseño, decisiones técnicas, resultados, limitaciones, mejoras). :contentReference[oaicite:2]{index=2}

---

## Requisitos funcionales (mínimos)

### 1) Conexión con un LLM comercial
- Uso de **variables de entorno** o gestor seguro para la API key.
- Parámetros del modelo visibles: `temperature`, `top_p`, `max_tokens`, etc. :contentReference[oaicite:3]{index=3}

### 2) RAG (Retrieval-Augmented Generation)
- Dividir la guía en **chunks** y generar **embeddings**.
- Guardar un índice en un **vector store** (FAISS, Chroma u otro).
- Responder **citando la fuente** (por ejemplo: chunk id / página / fragmento recuperado). :contentReference[oaicite:4]{index=4}

### 3) Diálogo multiturno
- Mantener **historial** de conversación (memoria).
- Controlar longitud para no superar límites de tokens (resumen, ventana deslizante, etc.). :contentReference[oaicite:5]{index=5}

### 4) Function call obligatoria: `get_weather(fecha)`
- Definir la herramienta con **JSON Schema** o **Pydantic**.
- Ejecutar la llamada (real o simulada) y manejar errores sencillos.
- Registrar intentos en un **log** (p. ej., `logs/tool_calls.log`). :contentReference[oaicite:6]{index=6}

