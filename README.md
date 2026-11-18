---
title: FAQ RAG Backend API
emoji: 💬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# faq-rag-backend-api

## Présentation

**faq-rag-backend-api** est l'API backend d'un chatbot de questions-réponses. Elle combine deux sources de réponse :

1. **Une FAQ** : un ensemble de thèmes, chacun avec plusieurs formulations de question et une réponse validée. Quand la question de l'utilisateur correspond clairement à un thème, la réponse de la FAQ est renvoyée telle quelle.
2. **Un RAG** (*Retrieval-Augmented Generation*) : quand la FAQ ne permet pas de répondre de façon fiable, l'API recherche les extraits pertinents dans une base documentaire vectorielle (ChromaDB) et fait générer une réponse par un LLM (Google Gemini) à partir de ces extraits.

Chaque question passe d'abord par la FAQ. Le RAG n'est utilisé qu'en repli.

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

Règles de décision :
- Le juge n'évalue que le meilleur candidat après reranking.
- Si le juge échoue (erreur d'API ou JSON invalide), la réponse FAQ est considérée comme non validée et la question passe au RAG.
- Si la recherche RAG ne trouve aucun extrait mais qu'il existe des candidats FAQ, le LLM est quand même appelé avec ces candidats. Sinon, l'API renvoie `DEFAULT_ANSWER` sans appeler le LLM.

### Stack

- **API** : FastAPI, Uvicorn, Docker.
- **FAQ** : Sentence-Transformers (bi-encoder `intfloat/multilingual-e5-small` et cross-encoder), index ChromaDB en mémoire reconstruit à chaque démarrage.
- **RAG** : base ChromaDB persistante téléchargée depuis un dataset Hugging Face, embeddings `paraphrase-multilingual-MiniLM-L12-v2`, recherche MMR via LangChain.
- **LLM** : Google Gemini, pour la génération et pour le juge FAQ.
- **Observabilité** : Langfuse, ou logs locaux si Langfuse n'est pas configuré.
- **Évaluation** : RAGAS sur les réponses générées par le RAG.
- **Tests** : `pytest`, tests unitaires et de contrat HTTP, avec des fakes pour tous les ports.

## Architecture

Le code suit une **architecture hexagonale** (ports et adaptateurs) : le domaine et la logique applicative ne dépendent d'aucune technologie, les modèles, bases et services externes sont branchés via des adaptateurs.

```
src/
├── domain/          Modèles métier et règles (routage FAQ, erreurs)
├── application/     Cas d'usage, services et ports (interfaces)
├── adapters/
│   ├── inbound/     API HTTP FastAPI
│   └── outbound/    Implémentations des ports (Chroma, Gemini, Langfuse, RAGAS…)
├── bootstrap/       Assemblage des dépendances (container)
└── config/          Paramètres, prompts, réponse par défaut
data/                Fichiers FAQ (JSON)
tests/               Tests unitaires et de contrat
```

| Couche | Fichiers principaux |
|---|---|
| Domaine | [faq_models.py](src/domain/model/faq_models.py) (`FaqCandidate`, `JudgeVerdict`, `decide_faq_route`), [rag_models.py](src/domain/model/rag_models.py), [feedback_models.py](src/domain/model/feedback_models.py), [evaluation_models.py](src/domain/model/evaluation_models.py) |
| Cas d'usage | [ask_question.py](src/application/use_cases/ask_question.py) (orchestration FAQ → RAG), [submit_feedback.py](src/application/use_cases/submit_feedback.py) |
| Services | [faq_lookup.py](src/application/services/faq_lookup.py) (recherche, rerank et juge FAQ) |
| Ports sortants | [ports/outbound/](src/application/ports/outbound/) : FAQ, reranker, juge, retrieval, embeddings, LLM, prompt, observabilité, évaluation |
| Adaptateur entrant | [app.py](src/adapters/inbound/api/app.py) (FastAPI), [schemas.py](src/adapters/inbound/api/schemas.py) |
| Adaptateurs sortants | [chroma_faq_search_adapter.py](src/adapters/outbound/faq/chroma_faq_search_adapter.py), [cross_encoder_reranker_adapter.py](src/adapters/outbound/reranker/cross_encoder_reranker_adapter.py), [gemini_judge_adapter.py](src/adapters/outbound/llm/gemini_judge_adapter.py), [gemini_llm_adapter.py](src/adapters/outbound/llm/gemini_llm_adapter.py), [chroma_retrieval_adapter.py](src/adapters/outbound/vectorstore/chroma_retrieval_adapter.py), [langfuse_observability_adapter.py](src/adapters/outbound/observability/langfuse_observability_adapter.py), [ragas_evaluation_adapter.py](src/adapters/outbound/evaluation/ragas_evaluation_adapter.py) |
| Assemblage | [container.py](src/bootstrap/container.py) |
| Configuration | [settings.py](src/config/settings.py) (`FaqSettings`), [prompt_gemini.py](src/config/prompt_gemini.py), [prompt_judge.py](src/config/prompt_judge.py), [default_answer.py](src/config/default_answer.py) |

### Données

- **FAQ** : fichier JSON au format `intents` dans [data/](data/). Chaque intent contient une `category` (thème), une `canonical_question`, des `utterances` (formulations alternatives) et une `response`. Toutes les formulations sont indexées ; la recherche garde au plus un candidat par thème.
- **Base documentaire RAG** : base ChromaDB hébergée dans un dataset Hugging Face (`{HF_ORG_NAME}/{HF_DATASET_NAME}`, par défaut `guild-open-tech/chroma-data`). Elle n'est pas dans le dépôt : elle est téléchargée au démarrage avec le token `API_HF_TOKEN`. Le modèle `EMBED_MODEL_NAME` doit être celui qui a servi à indexer la collection.

## API

Une fois le service lancé :

| Page | Chemin |
|---|---|
| Accueil (JSON avec liens) | `/` |
| Swagger (OpenAPI) | `/docs` |
| Redoc | `/redoc` |
| Schéma OpenAPI brut | `/openapi.json` |
| Santé | `/health` |

Endpoints :

- `GET /` : message d'accueil et liens utiles.
- `POST /api/v1/ask` : pose une question.
  - Paramètres RAG : `question`, `top_k`, `use_mmr`, `lambda_mult`, `similarity_threshold`, `trace_name`, `user_id`.
  - Paramètres FAQ, optionnels (sinon valeurs par défaut du serveur) : `faq_top_k`, `faq_similarity_threshold`, `faq_accept_threshold`, `faq_judge_threshold`. Si `faq_judge_threshold` est supérieur à `faq_accept_threshold`, l'API renvoie 400.
  - Réponse : `answer`, `source` (`faq` / `faq_judged` / `rag`), `chunks` (vide si la réponse vient de la FAQ), `faq_best_score`, `faq_candidates`, `judge` (`is_answering`, `reason`, ou `null` si le juge n'a pas été appelé), `trace_id`.
- `POST /api/v1/feedback` : envoie un feedback sur une réponse (`trace_id`, `score_value`, `feedback_type`, `comment`).
- `GET /health` : état du service et nombre de formulations FAQ indexées (`faq_count`).

Exemple :

```bash
curl -X POST http://localhost:7860/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "C'\''est quoi l'\''association ?"}'
```

## Lancement en local

Prérequis : Python 3.11, une clé `GOOGLE_API_KEY` et un token `API_HF_TOKEN` donnant accès au dataset Chroma.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
cp .env.example .env            # puis renseigner les clés
uvicorn src.adapters.inbound.api.app:app --host 0.0.0.0 --port 7860
```

Sous Windows, `/tmp` n'existe pas : garder `HF_CACHE_DIR=./.cache/hf` (valeur de `.env.example`).

Au premier démarrage, les modèles Sentence-Transformers et la base Chroma sont téléchargés, ce qui peut prendre quelques minutes.

Avec Docker :

```bash
docker build -t faq-rag-backend-api .
docker run --env-file .env -p 7860:7860 faq-rag-backend-api
```

### Tests

```bash
pytest
```

Les tests utilisent des fakes pour tous les ports : ils ne téléchargent aucun modèle et n'appellent aucun service externe.

## Déploiement sur Hugging Face Spaces

Le service se déploie sur un **Space Hugging Face de type Docker**. L'offre gratuite *CPU basic* (2 vCPU, 16 Go de RAM) suffit pour charger les modèles Sentence-Transformers, la base Chroma et RAGAS.

### Création du Space

1. Sur Hugging Face : *New Space*, choisir le SDK **Docker** (modèle *Blank*) et le matériel *CPU basic*.
2. Ajouter le Space comme remote Git du projet, puis pousser :
   ```bash
   git remote add space https://huggingface.co/spaces/<utilisateur>/<nom-du-space>
   git push space main
   ```
3. À chaque `git push`, le Space construit le [Dockerfile](Dockerfile) et lance `uvicorn` sur le port **7860**.

Le Space lit sa configuration dans l'en-tête YAML placé tout en haut de ce `README.md` (`sdk: docker`, `app_port: 7860`). Il faut le conserver.

### Variables d'environnement et secrets

Les renseigner dans *Settings > Variables and secrets* du Space :

- en **secrets** : `GOOGLE_API_KEY`, `API_HF_TOKEN`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `OPENAI_API_KEY` ;
- en **variables** : les autres paramètres, si les valeurs par défaut ne conviennent pas (`LANGFUSE_HOST`, seuils FAQ, modèles…).

Sans `OPENAI_API_KEY`, définir `ENABLE_RAGAS_EVAL=false`. Garder `HF_CACHE_DIR` à sa valeur par défaut `/tmp` dans le conteneur.

Le fichier `.env` est exclu de l'image Docker par [.dockerignore](.dockerignore) : les secrets doivent être définis dans le Space.

### Points d'attention

- **Stockage éphémère** : à chaque redémarrage ou redéploiement, les modèles et la base Chroma sont téléchargés à nouveau, et le contenu de `logs/` est perdu. Les traces durables passent par Langfuse.
- **Mise en veille** : un Space gratuit s'endort après une période d'inactivité. La requête suivante le réveille, avec un démarrage de plusieurs minutes (téléchargements et chargement des modèles).
- **URL publique** : `https://<utilisateur>-<nom-du-space>.hf.space`. Swagger est alors disponible sur `https://<utilisateur>-<nom-du-space>.hf.space/docs`.

## Variables d'environnement

Le fichier [.env.example](.env.example) liste les variables avec des valeurs prêtes à l'emploi. Les défauts ci-dessous sont ceux appliqués par le code quand la variable est absente.

### LLM (Gemini)

| Variable | Défaut | Usage |
|---|---|---|
| `GOOGLE_API_KEY` | (aucun) | **Obligatoire.** Clé API Google lue par le SDK `google-genai` pour appeler Gemini. |
| `MODEL_NAME` | `gemini-2.5-flash` | Modèle de génération. Le coût Langfuse n'est calculé que pour les modèles connus de `MODEL_PRICING_PER_1K` dans [ask_question.py](src/application/use_cases/ask_question.py). |
| `LLM_TEMPERATURE` | `0.1` | Température de génération. |
| `LLM_MAX_OUTPUT_TOKENS` | `512` | Nombre maximal de tokens de la réponse. |
| `LLM_THINKING_BUDGET` | `200` | Budget de tokens de « réflexion » de Gemini. |
| `LLM_SEED` | `42` | Seed pour des réponses reproductibles. |
| `DEFAULT_ANSWER` | `Je ne dispose pas d'informations sur ce sujet.` | Réponse renvoyée quand ni la FAQ ni le RAG ne fournissent de contenu (le LLM n'est alors pas appelé). |

### Embeddings (RAG)

| Variable | Défaut | Usage |
|---|---|---|
| `EMBED_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modèle d'embedding des questions pour le RAG. Doit être le même que celui utilisé pour indexer la collection Chroma. Réutilisé par RAGAS. |

### Base vectorielle (ChromaDB / dataset Hugging Face)

| Variable | Défaut | Usage |
|---|---|---|
| `API_HF_TOKEN` | (aucun) | **Obligatoire.** Token Hugging Face pour télécharger la base Chroma depuis le dataset. |
| `HF_ORG_NAME` | `guild-open-tech` | Organisation / utilisateur propriétaire du dataset HF. |
| `HF_DATASET_NAME` | `chroma-data` | Nom du dataset HF contenant la base Chroma. |
| `COLLECTION_NAME` | `miss_terry` | Collection Chroma interrogée. |
| `HF_CACHE_DIR` | `/tmp` | Dossier de cache HF (`HF_HOME`, `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE` en dérivent). En local sous Windows, utiliser par exemple `./.cache/hf`. |

### FAQ

| Variable | Défaut | Usage |
|---|---|---|
| `FAQ_JSON_PATH` | `data/260817_FAQ_Miss_Terry.json` (`.env.example` : `data/faq.json`) | Fichier FAQ (format `intents`). Un chemin relatif est résolu depuis la racine du projet. L'index est reconstruit en mémoire à chaque démarrage. |
| `FAQ_EMBED_MODEL_NAME` | `intfloat/multilingual-e5-small` | Bi-encoder de la FAQ. Les préfixes `query:`/`passage:` sont ajoutés automatiquement pour les modèles e5. |
| `FAQ_TOP_K` | `3` | Nombre maximal de candidats FAQ (un par thème). |
| `FAQ_SIMILARITY_THRESHOLD` | `0.85` | Seuil de similarité cosinus pour retenir un candidat FAQ. |
| `CROSS_ENCODER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Cross-encoder de rerank. Ce modèle est anglophone : passer à un modèle multilingue (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, `BAAI/bge-reranker-v2-m3`, etc.) impose de recalibrer les seuils. |
| `FAQ_ACCEPT_THRESHOLD` | `0.96` (`.env.example` : `0.90`) | Seuil haut : à partir de ce score, la réponse FAQ est renvoyée directement. |
| `FAQ_JUDGE_THRESHOLD` | `0.60` | Seuil bas : entre ce seuil et le seuil haut, le LLM juge est consulté. En dessous, la question passe au RAG. |

### LLM juge (Gemini)

| Variable | Défaut | Usage |
|---|---|---|
| `JUDGE_MODEL_NAME` | `gemini-2.5-flash` | Modèle du juge. Il utilise la même clé `GOOGLE_API_KEY` que la génération. |
| `JUDGE_TEMPERATURE` | `0.0` | Température du juge. |
| `JUDGE_THINKING_BUDGET` | `0` | Budget de « réflexion » du juge. |
| `JUDGE_MAX_OUTPUT_TOKENS` | `256` | Tokens maximum pour la réponse JSON `{is_answering, reason}`. |

### Observabilité (Langfuse)

Langfuse est activé uniquement si `LANGFUSE_PUBLIC_KEY` **et** `LANGFUSE_SECRET_KEY` sont renseignées. Sinon l'API utilise l'observabilité locale par logs.

| Variable | Défaut | Usage |
|---|---|---|
| `LANGFUSE_HOST` | `https://cloud.langfuse.com` (`.env.example`) | URL de l'instance Langfuse (lue par le SDK). |
| `LANGFUSE_PUBLIC_KEY` | (vide) | Clé publique du projet Langfuse. |
| `LANGFUSE_SECRET_KEY` | (vide) | Clé secrète du projet Langfuse. |

### Évaluation RAGAS

| Variable | Défaut | Usage |
|---|---|---|
| `ENABLE_RAGAS_EVAL` | `true` | `true` active l'évaluation RAGAS des réponses RAG ; toute autre valeur la désactive. |
| `RAGAS_LLM_MODEL_NAME` | `gpt-4o-mini` | Modèle LLM juge utilisé par RAGAS. |
| `OPENAI_API_KEY` | (vide) | Clé OpenAI nécessaire au LLM juge de RAGAS. Sans elle, l'initialisation de RAGAS échoue, un warning est loggé (`ragas.adapter.init_failed`) et l'API continue sans évaluation. |

### Logs et divers

| Variable | Défaut | Usage |
|---|---|---|
| `LOG_FILE` | `./logs/app.log` | Fichier de logs, rotation quotidienne à minuit, 15 jours conservés. Le dossier est créé au démarrage s'il n'existe pas. |
| `API_URL` | (aucun) | Optionnelle, uniquement recopiée dans les métadonnées Langfuse de l'étape de retrieval. |

## Observabilité (Langfuse)

Chaque question produit une trace `faq-rag-pipeline` avec les étapes suivantes :
- `faq-retrieval`, puis `faq-rerank` (avec le score `faq_best_ce_score`) ;
- `faq-judge`, seulement si le juge est consulté (avec le score `judge_verdict`) ;
- `retrieval` et `answer-generation`, seulement sur le chemin RAG (latence, tokens, coût estimé).

La trace porte aussi les scores `answer_source` et `total_latency`. Les feedbacks envoyés via `POST /api/v1/feedback` sont rattachés au `trace_id` renvoyé par `/api/v1/ask`. Sans clés Langfuse, les mêmes événements sont écrits dans le fichier de logs.

## Évaluation RAGAS

Après chaque réponse issue du RAG (pas pour les réponses FAQ), RAGAS calcule des métriques de qualité (LLM juge `RAGAS_LLM_MODEL_NAME` et embeddings `EMBED_MODEL_NAME`), publiées comme scores Langfuse. Désactivable avec `ENABLE_RAGAS_EVAL=false`.
