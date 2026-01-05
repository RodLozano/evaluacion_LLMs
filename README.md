# evaluacion_LLMs

# Asistente Turístico con LLM (RAG + Multiturno + Function Calling)

Proyecto final de la asignatura **Large Language Models** (Máster en IA, Cloud Computing y DevOps).  
El objetivo es construir un **asistente turístico reproducible** usando un **LLM comercial vía API** (p. ej. OpenAI GPT-4o o Google Gemini), integrando:

- **RAG** sobre una guía turística proporcionada por el profesorado (chunking + embeddings + vector store).
- **Diálogo multiturno** que mantenga el contexto de conversación y gestione el límite de tokens.
- **Function calling** obligatorio: `get_weather(fecha)` con **manejo básico de errores** y **registro en log**.

> Todo debe quedar integrado en un **único notebook principal**, con flujo claro y documentación. :contentReference[oaicite:1]{index=1}

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

