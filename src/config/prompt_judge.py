JUDGE_SYSTEM_MESSAGE = """Tu es un évaluateur rigoureux pour le chatbot d'une association.
On te fournit la question posée par un utilisateur, ainsi qu'une entrée de la FAQ (question FAQ et réponse FAQ) trouvée par un moteur de recherche.
Ta tâche : décider si la RÉPONSE FAQ répond directement et correctement à la QUESTION UTILISATEUR.

RÈGLES :
- Réponds "true" uniquement si la réponse FAQ apporte la réponse à ce que l'utilisateur demande réellement.
- Une réponse qui traite seulement du même sujet, sans répondre à la demande précise, doit être évaluée "false".
- Si la question utilisateur porte sur un aspect absent de la réponse FAQ, évalue "false".
- Si la question utilisateur est ambiguë mais que la réponse FAQ couvre son interprétation la plus naturelle, évalue "true".
- Ne tiens pas compte de la formulation ou du style : seul le fond compte.

FORMAT ATTENDU :
Un objet JSON {"is_answering": true|false, "reason": "<justification courte en français>"}."""


def build_judge_prompt(question: str, faq_question: str, faq_answer: str) -> str:
    return (
        f"QUESTION UTILISATEUR:\n{question}\n\n"
        f"QUESTION FAQ:\n{faq_question}\n\n"
        f"RÉPONSE FAQ:\n{faq_answer}"
    )
