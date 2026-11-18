# faq-rag-backend-api

## Présentation

**faq-rag-backend-api** fusionne les projets `faq-backend-api` et `rag-backend-api` en un seul chatbot. Le projet est construit sur l'architecture hexagonale du RAG. Chaque question passe d'abord par la FAQ et ne bascule sur le RAG que si la FAQ ne permet pas de répondre de façon fiable.

### Flux de réponse

```
question
  │
  ├─► FAQ : bi-encoder (e5-small, ChromaDB en mémoire, 1 candidat par thème)
  │         └─► cross-encoder (rerank, score sigmoïde 0..1)
  │
  ├─ score ≥ FAQ_ACCEPT_THRESHOLD ............................ réponse FAQ   (source = "faq")
  ├─ FAQ_JUDGE_THRESHOLD ≤ score < FAQ_ACCEPT_THRESHOLD
  │     └─► LLM juge (Gemini) : la réponse FAQ répond-elle à la question ?
  │           ├─ oui ............................................ réponse FAQ   (source = "faq_judged")
  │           └─ non / erreur du juge ─┐
  └─ score < FAQ_JUDGE_THRESHOLD, ou aucun candidat ─┴─► RAG  (source = "rag")
        recherche ChromaDB (MMR) + génération Gemini ;
        les réponses FAQ candidates sont ajoutées au prompt
        (section « RÉPONSES FAQ POTENTIELLEMENT LIÉES »)
```

Points de comportement :
- Le juge n'évalue que le meilleur candidat après reranking.
- Si le juge échoue (erreur d'API ou JSON invalide), la réponse FAQ est considérée comme non validée et on passe au RAG.
- Les règles de l'ancien projet FAQ (« score ≥ 0.96 » et « écart relatif < 0.10 → demande de précision ») ont été supprimées.
- Si la recherche RAG ne trouve aucun extrait mais qu'il existe des candidats FAQ, le LLM est quand même appelé avec ces candidats. Sinon, l'API renvoie `DEFAULT_ANSWER`.

### Organisation du code (ajouts par rapport au RAG)

| Couche | Fichiers |
|---|---|
| Domaine | [faq_models.py](src/domain/model/faq_models.py) (`FaqCandidate`, `JudgeVerdict`, `decide_faq_route`) |
| Ports sortants | [faq_search_port.py](src/application/ports/outbound/faq_search_port.py), [reranker_port.py](src/application/ports/outbound/reranker_port.py), [answer_judge_port.py](src/application/ports/outbound/answer_judge_port.py) |
| Application | [faq_lookup.py](src/application/services/faq_lookup.py) (étape FAQ), [ask_question.py](src/application/use_cases/ask_question.py) (orchestration FAQ → RAG) |
| Adaptateurs | [chroma_faq_search_adapter.py](src/adapters/outbound/faq/chroma_faq_search_adapter.py), [cross_encoder_reranker_adapter.py](src/adapters/outbound/reranker/cross_encoder_reranker_adapter.py), [gemini_judge_adapter.py](src/adapters/outbound/llm/gemini_judge_adapter.py) (prompt : [prompt_judge.py](src/config/prompt_judge.py)) |
| Configuration | [settings.py](src/config/settings.py) (`FaqSettings`), [data/faq.json](data/faq.json) |

### Stack

- FastAPI, ChromaDB, Sentence-Transformers (bi-encoder et cross-encoder), Google Gemini (génération et juge), Docker.
- Observabilité avec Langfuse, ou logs locaux en repli.
- Évaluation RAGAS sur les réponses issues du RAG.
- Tests unitaires et de contrat (`pytest`), avec des fakes pour tous les ports.

## Hugging Face

> Les liens ci-dessous sont ceux du projet d'origine `rag-backend-api`. Aucun Space n'a encore été créé pour `faq-rag-backend-api` : il faudra en créer un et mettre ces liens à jour.

- Space (dépôt Git `origin`) : https://huggingface.co/spaces/Loren/rag-backend-api
- Le Space est de type `sdk: docker` (voir le front-matter de [README.md](README.md)) : il construit le [Dockerfile](Dockerfile) et lance `uvicorn` à chaque `git push` vers `origin`.
- Les variables d'environnement et les secrets (`GOOGLE_API_KEY`, `API_HF_TOKEN`, clés Langfuse, etc.) se définissent dans *Settings > Variables and secrets* du Space (les clés en **secrets**).
- La base ChromaDB n'est pas dans le repo : elle est téléchargée au démarrage depuis le dataset HF `{HF_ORG_NAME}/{HF_DATASET_NAME}` (par défaut `guild-open-tech/chroma-data`) avec le token `API_HF_TOKEN`.

## URLs

| Swagger (OpenAPI) | Santé |
|---|---|
| https://loren-rag-backend-api.hf.space/docs | https://loren-rag-backend-api.hf.space/health |

La racine `https://loren-rag-backend-api.hf.space/` affiche un message d'accueil JSON avec les liens vers `/docs` et `/health`.

Schéma OpenAPI brut : `/openapi.json`. Redoc : `/redoc`.

Endpoints :

- `GET /` : message d'accueil et liens utiles.
- `POST /api/v1/ask` : pose une question.
  - Paramètres RAG : `question`, `top_k`, `use_mmr`, `lambda_mult`, `similarity_threshold`, `trace_name`, `user_id`.
  - Paramètres FAQ, optionnels (sinon valeurs par défaut du serveur) : `faq_top_k`, `faq_similarity_threshold`, `faq_accept_threshold`, `faq_judge_threshold`. Si `faq_judge_threshold` est supérieur à `faq_accept_threshold`, l'API renvoie 400.
  - Réponse : `answer`, `source` (`faq` / `faq_judged` / `rag`), `chunks` (vide si la réponse vient de la FAQ), `faq_best_score`, `faq_candidates`, `judge` (`is_answering`, `reason`, ou `null` si le juge n'a pas été appelé), `trace_id`.
- `POST /api/v1/feedback` : envoie un feedback sur une réponse (`trace_id`, `score_value`, `feedback_type`, `comment`).
- `GET /health` : état du service et nombre de formulations FAQ indexées (`faq_count`).


## Variables d'environnement

### LLM (Gemini)

| Variable | Défaut | Usage |
|---|---|---|
| `GOOGLE_API_KEY` | (aucun) | **Obligatoire.** Clé API Google lue par le SDK `google-genai` pour appeler Gemini. |
| `MODEL_NAME` | `gemini-2.5-flash` | Modèle de génération. Le coût Langfuse n'est calculé que pour les modèles connus de `MODEL_PRICING_PER_1K` dans [ask_question.py](src/application/use_cases/ask_question.py). |
| `LLM_TEMPERATURE` | `0.1` | Température de génération. |
| `LLM_MAX_OUTPUT_TOKENS` | `512` | Nombre maximal de tokens de la réponse. |
| `LLM_THINKING_BUDGET` | `200` | Budget de tokens de « réflexion » de Gemini. |
| `LLM_SEED` | `42` | Seed pour des réponses reproductibles. |
| `DEFAULT_ANSWER` | `Je ne dispose pas d'informations sur ce sujet.` | Réponse renvoyée quand aucun extrait ne dépasse le seuil de similarité (le LLM n'est alors pas appelé). |

### Embeddings

| Variable | Défaut | Usage |
|---|---|---|
| `EMBED_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embedding des questions. Doit être le même que celui utilisé pour indexer la collection Chroma. Absent de `.env.example` mais pris en compte. Réutilisé par RAGAS. |

### Base vectorielle (ChromaDB / dataset Hugging Face)

| Variable | Défaut | Usage |
|---|---|---|
| `API_HF_TOKEN` | (aucun) | **Obligatoire.** Token Hugging Face pour télécharger la base Chroma depuis le dataset. |
| `HF_ORG_NAME` | `guild-open-tech` | Organisation / utilisateur propriétaire du dataset HF. |
| `HF_DATASET_NAME` | `chroma-data` | Nom du dataset HF contenant la base Chroma. |
| `COLLECTION_NAME` | `miss_terry` | Collection Chroma interrogée. |
| `HF_CACHE_DIR` | `/tmp` | Dossier de cache HF (`HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` en dérivent). **En local sous Windows, `/tmp` n'existe pas : définir par exemple `HF_CACHE_DIR=./.cache/hf`.** |

### Observabilité (Langfuse)

Langfuse est activé uniquement si `LANGFUSE_PUBLIC_KEY` **et** `LANGFUSE_SECRET_KEY` sont renseignées. Sinon l'API utilise l'observabilité locale par logs.

| Variable | Défaut | Usage |
|---|---|---|
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` (`.env.example`) | URL de l'instance Langfuse (lue par le SDK). |
| `LANGFUSE_PUBLIC_KEY` | (vide) | Clé publique du projet Langfuse. |
| `LANGFUSE_SECRET_KEY` | (vide) | Clé secrète du projet Langfuse. |

### FAQ

| Variable | Défaut | Usage |
|---|---|---|
| `FAQ_JSON_PATH` | `data/faq.json` | Fichier FAQ (format `intents`). Un chemin relatif est résolu depuis la racine du projet. L'index est reconstruit en mémoire à chaque démarrage. |
| `FAQ_EMBED_MODEL_NAME` | `intfloat/multilingual-e5-small` | Bi-encoder de la FAQ. Les préfixes `query:`/`passage:` sont ajoutés automatiquement pour les modèles e5. |
| `FAQ_TOP_K` | `3` | Nombre maximal de candidats FAQ (un par thème). |
| `FAQ_SIMILARITY_THRESHOLD` | `0.85` | Seuil de similarité cosinus pour retenir un candidat FAQ. |
| `CROSS_ENCODER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder de rerank. Ce modèle est anglophone : un modèle multilingue (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `BAAI/bge-reranker-v2-m3`, etc.) imposera de recalibrer les seuils. |
| `FAQ_ACCEPT_THRESHOLD` | `0.90` | Seuil haut : à partir de ce score, la réponse FAQ est renvoyée directement. |
| `FAQ_JUDGE_THRESHOLD` | `0.60` | Seuil bas : entre ce seuil et le seuil haut, le LLM juge est consulté. En dessous, on passe au RAG. |

### LLM juge (Gemini)

| Variable | Défaut | Usage |
|---|---|---|
| `JUDGE_MODEL_NAME` | `gemini-2.5-flash` | Modèle du juge. Il utilise la même clé `GOOGLE_API_KEY` que la génération. |
| `JUDGE_TEMPERATURE` | `0.0` | Température du juge. |
| `JUDGE_THINKING_BUDGET` | `0` | Budget de « réflexion » du juge. |
| `JUDGE_MAX_OUTPUT_TOKENS` | `256` | Tokens maximum pour la réponse JSON `{is_answering, reason}`. |

### Évaluation RAGAS

| Variable | Défaut | Usage |
|---|---|---|
| `ENABLE_RAGAS_EVAL` | `true` | `true` active l'évaluation RAGAS des réponses ; toute autre valeur la désactive. |
| `RAGAS_LLM_MODEL_NAME` | `gpt-4o-mini` | Modèle LLM juge utilisé par RAGAS. |
| `OPENAI_API_KEY` | (vide) | Clé OpenAI nécessaire au LLM juge. Sans elle, l'initialisation de RAGAS échoue, un warning est loggé (`ragas.adapter.init_failed`) et l'API continue sans évaluation. |

Les embeddings utilisés par RAGAS sont ceux de `EMBED_MODEL_NAME`.

### Logs et divers

| Variable | Défaut | Usage |
|---|---|---|
| `LOG_FILE` | `./logs/app.log` | Fichier de logs, rotation quotidienne à minuit, 15 jours conservés. Le dossier est créé au démarrage s'il n'existe pas. |
| `API_URL` | (aucun) | Optionnelle, uniquement recopiée dans les métadonnées Langfuse de l'étape de retrieval. |

## Observabilité (Langfuse)

Les traces `faq-rag-pipeline` contiennent les étapes suivantes :
- `faq-retrieval`, puis `faq-rerank` (avec le score `faq_best_ce_score`) ;
- `faq-judge`, seulement si le juge est consulté (avec le score `judge_verdict`) ;
- `retrieval` et `answer-generation`, seulement sur le chemin RAG (latence, tokens, coût estimé).

La trace porte aussi les scores `answer_source` et `total_latency`. Les feedbacks envoyés via `POST /api/v1/feedback` sont rattachés au `trace_id` renvoyé par `/api/v1/ask`. Sans clés Langfuse, les mêmes événements sont écrits dans `logs/app.log`.

## Évaluation RAGAS

Après chaque réponse issue du RAG (pas pour les réponses FAQ), RAGAS calcule des métriques de qualité (LLM juge `RAGAS_LLM_MODEL_NAME` et embeddings `EMBED_MODEL_NAME`) publiées comme scores Langfuse. Désactivable avec `ENABLE_RAGAS_EVAL=false`.

---
