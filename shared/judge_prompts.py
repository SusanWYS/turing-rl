"""Judge prompt constants shared by training rewards and offline eval judges."""


SPECIFICITY_PROMPT = """## Task

You are evaluating one candidate response for grounded specificity.

A high score means the candidate response is specifically tied to this interaction and specifically compatible with the target user, rather than being a reusable plausible response from an average participant.
A low score means the candidate response is broadly plausible but generic, reusable, contradictory, unsupported, or artifact-like.

## Inputs

You will receive:
1. Target user evidence
2. The current interaction context
3. One candidate response

<|Target User Evidence|>
{user_history}
<|End Target User Evidence|>

<|Interaction Context|>
{context}
<|End Interaction Context|>

<|Candidate Response|>
{candidate_response}
<|End Candidate Response|>

## Evaluation Criteria

Score the candidate as an absolute judgment of grounded user-specificity.

A high-scoring response should:
- fit the exact current interaction
- be compatible with the target user's observed behavior
- contain a natural situated move, stance, framing, effort level, or expression pattern that makes it specific to this user/context

A low-scoring response may be fluent, coherent, topical, or plausible, but it is weakly grounded, broadly reusable, incompatible with the user/context, or artifact-like.

## Core Rules

- Do not reward general quality, fluency, politeness, balance, helpfulness, verbosity, or topical plausibility by themselves.
- Do not require explicit personal facts; user evidence is background for compatibility and distinctiveness.
- Reward specificity only when it is natural, proportionate, and tied to this exact next response.
- Penalize invented user traits, unsupported personal assumptions, wrong perspective, over-explaining, over-personalizing, persona-caricature, and assistant-like responses.
- A response that is fully compatible with the user can still be generic. Do not treat compatibility alone as specificity.
- A response that smoothly advances the interaction but could be written by many plausible participants should receive only moderate scores.
- Treat the target user's own earlier turns inside the current interaction context as strong evidence for immediate stance, effort level, style, and local move.
- Penalize uncalled-for imports of personal facts, profile details, or past behavior when the current interaction does not make them natural to say.
- If the response contradicts the context, uses the wrong speaker perspective, or invents personal facts, the overall score should be low regardless of other strengths.

Score each dimension from 0.0 to 1.0.
Use the full continuous range when appropriate. The regions below define the scale.

## Dimensions

### Dimension 1: Context Specificity

How specifically does the candidate engage with the exact local interaction?

**Scoring regions:**
- 0.9-1.0: Tightly tied to the exact claim, question, event, decision point, constraint, disagreement, callback, or conversational detail at issue.
- 0.7-0.9: Clearly grounded in the specific local context with only minor generic framing.
- 0.5-0.7: Responds to the main local issue but misses important details, constraints, or conversational pressure.
- 0.3-0.5: Related to the broad topic but weakly grounded in this exact context.
- 0.1-0.3: Barely connected; could fit many similar interactions.
- 0.0-0.1: Off-topic, responds to the wrong issue or speaker, or ignores the local context.

**High context specificity:**
- Addresses the actual local claim, request, constraint, preference, decision point, joke, disagreement, or reference.
- Tracks who said what and what the candidate is responding to.
- Uses or reacts to local details without merely copying them.
- Can be implicit, terse, joking, fragmentary, or low-effort when the local function is recoverable.
- Preserves the target user's current local trajectory when the target user has already established one in the interaction.

**Low context specificity:**
- Gives a generic response about the broad topic.
- Responds to the wrong participant, wrong request, or wrong part of the interaction.
- Misses the key disagreement, question, constraint, or situational detail.
- Directly answers a public-context detail but replaces the target user's likely local move with a cleaner or more obvious continuation.

### Dimension 2: User Evidence Compatibility

How compatible is the candidate with the target user's evidence without unnaturally exposing, exaggerating, or inventing that evidence?

**Scoring regions:**
- 0.9-1.0: Strongly compatible with the target user's evidence and current-context behavior, with no contradiction or unnatural profile display.
- 0.7-0.9: Clearly compatible with important user evidence, stance, effort level, interaction habit, or constraints.
- 0.5-0.7: Compatible but broad, weakly distinctive, incomplete, or mildly over-explicit.
- 0.3-0.5: Only weakly compatible; mostly generic, overly smooth, over-personalized, or based on a thin user signal.
- 0.1-0.3: Little meaningful compatibility with the user evidence, or the response seems to perform a profile rather than act naturally.
- 0.0-0.1: Contradicts the user evidence, invents unsupported personal facts, uses the wrong perspective, or relies on a wrong user model.

**High compatibility:**
- Does not contradict the user's known behavior, values, constraints, or current-context stance.
- Uses only the amount of user-specific detail that would naturally appear in this moment.
- Preserves the user's likely local move instead of replacing it with a profile-shaped explanation.
- Can reflect distinctive user evidence through form, effort level, stance, humor, skepticism, brevity, or refusal to elaborate.

**Low compatibility:**
- Contradicts the user evidence or current-context behavior.
- Invents personal facts, preferences, motives, or biography.
- Pulls in user history or profile details when the interaction does not call for them.
- Sounds like a persona sketch, averaged user model, or assistant summary of the user.
- Is merely non-contradictory and locally sensible, but gives no meaningful evidence of this user.

## Overall Score

Compute the overall score with these weights:
- context_specificity: 50%
- user_evidence_compatibility: 50%

Apply these caps after computing the weighted score:
- If the response contradicts the context, uses the wrong speaker perspective, or cannot work as the next response, overall should be at most 0.20.
- If the response invents personal facts or unnaturally imports user-history/profile details, overall should be at most 0.40.


## Output Requirements

- Return valid JSON only.
- Do not include Markdown.
- Each reason must be one sentence.
- The reasoning field must briefly justify the score tradeoffs using the local context, target-user evidence, genericness, and candidate response.
- Each dimension score must be a number in [0.0, 1.0].
- Use the full continuous range when appropriate; do not restrict yourself to the region boundaries.
- The overall score must follow the weighted formula and caps above.

Return this JSON structure:

{{
  "context_specificity": {{
    "score": 0.0,
    "reason": "..."
  }},
  "user_evidence_compatibility": {{
    "score": 0.0,
    "reason": "..."
  }},
  "overall": 0.0,
  "reasoning": "..."
}}

Your output:"""


RESPONSE_ONLY_PROMPT_BATCHED = """You are a helpful and meticulous evaluator. Your task is to score how well the generated response(s) align with the ground truth user response. Description of response: [HUMAN]'s actual written comment or reply text.

You will be given past messages for [HUMAN], the current context, the ground truth response, and generated response(s) that you should evaluate.

Provided Information:
{context}

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

{generations_text}

Scoring Criteria:
For each generated response, assign a score in [0, 1] based on how accurately it reflects the ground truth response.

Guidelines:
1. Extract 1-3 key points:
  - Extract K key points from the ground truth response along the response dimension (e.g., if evaluating a "stance", pick key points related to the stance like "clearly disagrees with X", if evaluating a "response", pick key points about the response like "offers a solution to Y").
  - Because you are evaluating the full response, consider all major content, style, and intent cues expressed in the ground truth response.
  - Each key point should be specific and distinct.

2. Score how well the generated response matches each key point:
  - For each key point i, compare it with the generated response and assign a match value m_i in range [0, 1]:
  - 1.0: The key point is precisely and perfectly reflected.
  - [0.7, 0.9]: Mostly reflected with small imperfections.
  - [0.4, 0.6]: Partially reflected or vague, but still leaning in the correct direction.
  - [0.1, 0.3]: Very weak reflection.
  - 0.0: Missed, contradicted, or reversed.

3. Compute coverage C = (m_1 + m_2 + ... + m_K) / K, which measures how comprehensive the generated response reflects the ground truth response.

4. Compute penalty P for extra or conflicting content:
  - Examine additional content in the generated response beyond those key points:
  - Does it introduce unsupported evidence and assumptions?
  - Is it irrelevant to what ground truth response expresses?
  - Is it only using generic commentary or high-level framing that misses the ground truth's goals, values, communication style, beliefs, and emotions specific to [HUMAN]?
  - Set a penalty P in [0, 1]:
  - 0.0: No problematic extra content; everything is perfectly matched.
  - [0.1, 0.3]: Slightly unnecessary, mildly speculative, or generic detail; meaning essentially unchanged.
  - [0.4, 0.6]: Moderate speculative, irrelevant, or vague content that somewhat shifts emphasis or adds unsupported ideas.
  - [0.7, 0.9]: Significant speculative, misleading, or conflicting content that clearly changes the meaning.
  - 1.0: Mostly off-topic, contradictory, or dominated by incorrect/hallucinated content.
  - Penalize sycophantic openings, especially formulaic phrases such as "you're absolutely right", "totally", "completely agree", "100%", "that is right", "yeah", or "it's a good starting point", unless [HUMAN]'s prior messages show that this kind of agreement-first opener is genuinely characteristic of their voice.

5. Response-specific checks:
  - The generated response may or may not reuse phrases from the context; however, if the generated response just directly copies previous context, without quoting it, treat that as off-task behavior and give a score of 0.
  - Wrong-perspective hard zero: if the generated response treats another user's perspective, identity, or experience in the thread as [HUMAN]'s own, or speaks from another participant's first-person perspective, give a score of 0.
  - Assistant-like hard zero: if the ground truth is [HUMAN]'s next question, follow-up, or short request, but the generated response behaves like an assistant reply instead by directly answering the earlier prompt, giving a polished explanation, or presenting a structured helpful breakdown, treat that as wrong perspective and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))

Additional considerations:
- Follow the instruction carefully.
- Be strict and reserve scores above 0.8 for clearly outstanding matches.
- Do not reward verbosity or generic topical plausibility; reward user-specific evidence.

Output format (JSON):
{{
  "key_points": "<analysis of key points from ground truth along response dimension>",
  "1": {{"thought": "<how well the 1st generated response matches each key point and compute the final score>", "score": <score>}},
  "2": ...
}}

Format Notes:
- All text in "key_points" and "thought" fields MUST be on a single line with no line breaks or newlines.
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (').
- Double check the JSON array's format, especially the comma and quotation marks.
- Ensure that ALL fields, especially "thought" and "score", are present for each item.
- You must provide exactly {num_generations} scores for the generated response(s).

Your output:"""


RESPONSE_ONLY_NO_HARD_FLAGS_PROMPT_BATCHED = RESPONSE_ONLY_PROMPT_BATCHED.replace(
    """
5. Response-specific checks:
  - The generated response may or may not reuse phrases from the context; however, if the generated response just directly copies previous context, without quoting it, treat that as off-task behavior and give a score of 0.
  - Wrong-perspective hard zero: if the generated response treats another user's perspective, identity, or experience in the thread as [HUMAN]'s own, or speaks from another participant's first-person perspective, give a score of 0.
  - Assistant-like hard zero: if the ground truth is [HUMAN]'s next question, follow-up, or short request, but the generated response behaves like an assistant reply instead by directly answering the earlier prompt, giving a polished explanation, or presenting a structured helpful breakdown, treat that as wrong perspective and give a score of 0.

6. Compute the final score = max(0, min(1, C - P))
""",
    """
5. Compute the final score = max(0, min(1, C - P))
""",
)


RESPONSE_BREAKDOWN_PROMPT_BATCHED = """Rate how similar and complete each generated response is compared with the ground truth response.

You will be given past messages for [HUMAN], the current context, the ground truth response, and generated response(s) that you should evaluate.

Provided Information:
{context}

<|The Start of Ground Truth Response|>
{ground_truth}
<|The End of Ground Truth Response|>

{generations_text}

For each generated response, assign three scores in [0.0, 1.0]:

1. "semantic_similarity": how similar the generated response is to the ground truth response in core meaning and information.
Scoring guidelines:
1.0 = Identical meaning, all key information preserved
0.75-0.95 = Same core meaning with minor semantic differences
0.55-0.75 = Similar meaning but some information changes
0.35-0.55 = Related topics but different focus or emphasis
0.15-0.35 = Some semantic overlap but different main message
0.0-0.15 = Completely different meaning or topic

2. "information_completeness": how completely the generated response preserves and covers the information from the ground truth response.
Scoring guidelines:
1.0 = All information fully preserved
0.75-0.95 = Minor details missing but all key info present
0.55-0.75 = Some important information missing
0.35-0.55 = Significant information gaps
0.15-0.35 = Major information missing, only basics preserved
0.0-0.15 = Almost no information preserved

3. "score": a strict final overall alignment score based on semantic similarity and information completeness. The final score should not be high unless both semantic similarity and information completeness are high. Penalize unsupported extra content, irrelevant speculation, excessive verbosity relative to the ground truth, wrong perspective, and source copying.

Penalty guidance:
- Hard-penalize sycophantic or generic agreement responses, especially formulaic openings such as "you're absolutely right", "totally", "completely agree", "100%", "that is right", or "yeah".
- Hard-penalize generic compromise responses that soften, reframe, or dilute a specific ground-truth position into a broad, moderate, or generic takeaway, for example turning a concrete argument into "it's a good starting point, but not a silver bullet" or "the real issue is...".
- Topical relevance is not enough: if the response mainly adds plausible-sounding background, generic commentary, or reusable high-level framing without matching the ground truth's goals, values, communication style, beliefs, and emotions, assign a penalty.

Examples for semantic similarity:

Reference: "I need to reschedule our meeting to Thursday at 3pm because I have a doctor's appointment on Tuesday."
Predicted: "I have a medical appointment on Tuesday, so can we move our meeting to Thursday afternoon at 3?"
Explanation: The predicted dialogue conveys the same core meaning as the reference with only very minor wording differences. All key information is preserved including the reason for rescheduling, the conflicting appointment, and the new proposed time. The slight paraphrasing does not change the semantic content.
<score>0.95</score>

Reference: "I'm excited about the new project because it involves machine learning and will help improve customer experience."
Predicted: "The new project is interesting and uses AI technology."
Explanation: While both dialogues discuss a new project involving AI/machine learning, the predicted version has a different emphasis and is missing important nuances. The reference conveys excitement and specifically mentions the goal of improving customer experience, whereas the predicted dialogue only expresses mild interest and uses the more general term "AI technology." The core topic is related but the focus and emotional tone differ significantly.
<score>0.50</score>

Reference: "I'm planning to visit Paris next summer to see the museums and try French cuisine."
Predicted: "My favorite programming language is Python because it's versatile and easy to learn."
Explanation: These two dialogues are about completely different topics with no semantic overlap whatsoever. The reference discusses travel plans to Paris, while the predicted dialogue is about programming language preferences. There is no connection in meaning or information between them.
<score>0.0</score>

Examples for information completeness:

Reference: "The patient should take 500mg of amoxicillin twice daily for 10 days. Avoid alcohol and dairy products within 2 hours of each dose."
Predicted: "Take 500mg amoxicillin twice daily for 10 days. Avoid alcohol and dairy around the time you take it."
Explanation: The predicted dialogue preserves all critical information from the reference including the exact dosage (500mg), frequency (twice daily), duration (10 days), and both restrictions (alcohol and dairy). The only minor difference is the less precise "around the time you take it" instead of "within 2 hours," but the core medical information is fully intact.
<score>0.95</score>

Reference: "Our company was founded in 1987 by Sarah Chen in Boston. We now have 500 employees across 12 offices and generated $50M in revenue last year."
Predicted: "The company was started by Sarah Chen and has grown to multiple offices with hundreds of employees."
Explanation: The predicted dialogue retains the founder's name but loses significant specific information. The founding year (1987) and location (Boston) are missing. The precise employee count (500) is reduced to vague "hundreds," the exact office count (12) becomes "multiple," and the revenue figure ($50M) is completely omitted. While the general narrative is preserved, most quantitative details are lost.
<score>0.50</score>

Reference: "The experiment showed a 23% increase in cell growth when exposed to 450nm blue light for 6 hours at 37°C, with p<0.001 significance."
Predicted: "The experiment showed that blue light increased cell growth."
Explanation: The predicted dialogue only preserves the most basic conclusion that blue light increased cell growth. All quantitative and methodological details are lost: the specific percentage (23%), wavelength (450nm), exposure duration (6 hours), temperature (37°C), and statistical significance (p<0.001). Only the fundamental relationship between blue light and cell growth remains.
<score>0.25</score>

Remember to judge each generated response against the ground truth response only. Do not compare generated responses to each other.

Perspective rule:
- Determine whether each generated response is written as [HUMAN].
- If it speaks as another user in the context, or treats another user's first-person statements as if they were [HUMAN]'s own, set "wrong_perspective" to true and set all three scores to 0.0.
- If the ground truth is [HUMAN]'s next question, follow-up, or short request, but the generated response instead behaves like an assistant reply by directly answering the earlier prompt, giving a polished explanatory answer, or providing a structured helpful breakdown, set "wrong_perspective" to true and set all three scores to 0.0.
- Otherwise set "wrong_perspective" to false.

Be strict and reserve scores above 0.8 for clearly outstanding matches.

Output format (JSON):
{{
  "key_points": "<brief single-line analysis of the key information and meaning in the ground truth response>",
  "1": {{"thought": "<single-line reasoning for the scores>", "semantic_similarity": <float in [0,1]>, "information_completeness": <float in [0,1]>, "score": <float in [0,1]>, "wrong_perspective": <true or false>, "source_copy": <true or false>}},
  "2": ...
}}

Format Notes:
- All text in "key_points" and "thought" fields MUST be on a single line with no line breaks or newlines.
- Use standard JSON string format with double quotes. For any quotes needed inside strings, use single quotes (').
- Ensure that all numbered items from 1 to {num_generations} are present.
- Ensure that every item includes "thought", "semantic_similarity", "information_completeness", "score", "wrong_perspective", and "source_copy".

Your output:
"""


TURING_PROMPT = """## Task

You are judging a pairwise Turing test for personalized user simulation.

Your task is to decide which candidate response was written by the real [HUMAN] user and which was written by an AI imitating that user.

## Inputs

You will receive:
1. Past responses from [HUMAN]
2. The current conversation context
3. Two candidate responses, Response A and Response B
4. An advisory watchlist for source-copy checks

One candidate is the real [HUMAN] response. The other candidate is AI-generated.

## User History

<|User History|>
{user_history}
<|End User History|>

## Context

<|Context|>
{context}
<|End Context|>

## Candidate Responses

### Response A

<|Response A|>
{response_a}
<|End Response A|>

### Response B

<|Response B|>
{response_b}
<|End Response B|>

## Advisory Watchlist

### Source-Copy Watchlist

<|Source-Copy Watchlist|>
{source_copy_watchlist}
<|End Source-Copy Watchlist|>


## Evaluation Procedure

Evaluate each response independently before comparing them. Reason backward from the response to the likely target, goal, and style.

Score each response on three criteria from 0.0 to 1.0.

## Criteria

### 1. Immediate target

Identify the exact part of the current context that the response addresses.

Consider whether the response:
- Reacts to a specific point in the latest [OTHER] turn, the broader current context, or a plausible topic pivot
- Understands what the other person actually said
- Speaks from [HUMAN]'s perspective rather than [OTHER]'s perspective
- Is directed at [OTHER] or [OTHER - OP], rather than at someone merely described in the thread

A topic pivot can be valid when it is a plausible next move for [HUMAN]. Do not penalize a pivot only because it does not directly answer the previous turn, especially if [HUMAN] often pivots in the history or current context.

Do not reward broad persona fit if the response misstates a speaker's stance, role, or personal experience; assign a low immediate_target_score.

immediate_target_score:
- 0.0-0.2 = Wrong or absent target. The response does not reply to the current context, targets the wrong person, takes [OTHER]'s role, or responds to someone only described in the thread.
- 0.2-0.4 = Weak target. The response is broadly on topic but does not clearly address the latest [OTHER] turn, a relevant current-context point, or a plausible human topic pivot.
- 0.4-0.6 = Mixed target. The response addresses the general context or makes a possible pivot, but the exact target is ambiguous, unsupported, or only loosely connected to the current exchange.
- 0.6-0.8 = Good target. The response addresses a plausible part of the current context or makes a plausible topic pivot from [HUMAN]'s perspective, with only minor ambiguity.
- 0.8-1.0 = Strong target. The response clearly addresses the most plausible current-context point, or makes a strongly human-plausible pivot, from [HUMAN]'s perspective and directed at [OTHER] or [OTHER - OP].

### 2. Human goal

Given the immediate target, identify what [HUMAN] was probably trying to do with the response. When judging the goals, use [HUMAN]'s history as evidence, not as a rigid script.

Consider whether the goal:
- Is plausible for [HUMAN] in this context, given [HUMAN]'s history
- Fits the local conversation pattern
- Preserves a specific personal or context-sensitive intent when one is available
- Avoids replacing [HUMAN]'s likely intent with a generic information request, assistant-like task, or narrator behavior

Do not reward a broadly useful or sensible goal if it replaces [HUMAN]'s likely local move, such as a joke, aside, anecdote, correction, quote-reply, agreement, disagreement, question, or brief reaction.

human_goal_score:
- 0.0-0.2 = Wrong or implausible goal. The response's goal does not fit [HUMAN], contradicts the context, invents a new task, or behaves like an assistant or narrator.
- 0.2-0.4 = Weak goal. The response has a generic or loosely plausible goal but mostly misses what [HUMAN] would likely try to do next, including any plausible pivot.
- 0.4-0.6 = Mixed goal. The response's goal is plausible by topic but changes, broadens, softens, escalates, or redirects [HUMAN]'s likely next move without strong support.
- 0.6-0.8 = Good goal. The response's goal, including a reasonable new-question or topic-pivot goal, is plausible for [HUMAN] in this context, with only minor uncertainty or drift.
- 0.8-1.0 = Strong goal. The response's goal naturally follows from [HUMAN]'s current intent, history, or local topic-switching pattern, while preserving personal or context-sensitive intent.

### 3. Communication style and length

Judge how the response is written, separate from whether it is a question, statement, pivot, opinion, or personal reframing.

Compare the response to [HUMAN]'s past responses and the local framing. Consider:
- Wording and phrasing
- Tone, humor, bluntness, and emotion
- Length and level of detail
- Grammar, capitalization, punctuation, and misspellings
- Specificity and natural level of effort

Do not reward generic fluency. Smooth, polished, or conventionally well-written prose is not a style match unless [HUMAN] writes that way. Judge whether the response preserves [HUMAN]'s characteristic wording, rhythm, roughness, formatting, punctuation, humor, profanity, hedging, and level of elaboration.
Match length as a style feature. A response that is much shorter or much longer than [HUMAN]'s likely response should not receive a high communication_style_score unless [HUMAN]'s history supports that length in this local situation.
Clearly broken or artifact-like writing should dominate the judgment even when the target or goal seems plausible. Hard penalize artifact-like generations.

communication_style_score:
- 0.0-0.2 = Strong mismatch. Wording, rhythm, tone, length, grammar, punctuation, specificity, roughness, polish, or format strongly conflicts with [HUMAN]'s history and local framing, or the response is artifact-like.
- 0.2-0.4 = Weak style match. The response matches only easy surface cues, such as being short, informal, fluent, blunt, or emotional, but misses [HUMAN]'s distinctive wording, rhythm, roughness, punctuation, humor, specificity, or level of elaboration.
- 0.4-0.6 = Mixed style match. Some style cues fit, but important mismatches remain in length, polish, grammar, emotion, specificity, formatting, punctuation, or level of effort; smooth generic prose usually belongs here at best.
- 0.6-0.8 = Good style match. The response mostly matches [HUMAN]'s characteristic wording, rhythm, tone, length, formatting, punctuation, specificity, and level of elaboration, with only minor mismatch; length should be in the same rough range as [HUMAN]'s likely response.
- 0.8-1.0 = Strong style match. The response sounds specifically like [HUMAN], including distinctive phrasing, rhythm, roughness or polish, humor, profanity, hedging, formatting, punctuation, specificity, natural level of effort, and closely matched length.

## Penalty Checks

Each penalty is scored from 0.0 to 1.0. Use the same scale for every penalty:
- 0.0 = No issue.
- 0.1-0.2 = Minor possible issue; mention it if relevant, but it should barely affect the judgment.
- 0.3-0.4 = Noticeable issue; the response is still plausibly human, but confidence should drop.
- 0.5-0.6 = Serious issue; the response has a substantial penalty-worthy flaw.
- 0.7-0.8 = Very strong issue; the response is unlikely to be human for this reason.
- 0.9-1.0 = Decisive issue; the response is almost certainly not human for this reason.

### Bad quote or source copy
The source-copy watchlist is an advisory 5-gram scan against user history and current context. Only assign source_copy_penalty for a response when the watchlist is triggered for that response. If the watchlist is off or not triggered for a response, assign 0.0 and do not invent source-copy violations from short phrases not shown in the watchlist.

When the watchlist is triggered, first decide whether the overlap is local uptake: the candidate may repeat words from the immediate context in order to quote, agree with, disagree with, answer, or otherwise react to that exact text. This is allowed even when the reused text is not explicitly marked with quotation marks, blockquote formatting, or attribution. Do not treat missing quotation marks, blockquote formatting, or attribution as source-copy evidence when the copied words come from the immediate context and the candidate is reacting to them. Do not penalize logical quotes without quotation marks. Do not penalize generic conversational frames, common question templates, pet phrases, or common sign-offs.

If copied text is not a natural quote or local uptake (i.e., a direct copy from the user history), assign a high source_copy_penalty. If the response quotes text but the quotation logic is confusing or contradictory, assign a high source_copy_penalty.

### Wrong target or speaker role
Assign wrong_target_or_role_penalty when a response speaks from the wrong role, addresses the wrong speaker, or assigns an unsupported stance, experience, motive, relationship, or conflict to the speaker it addresses.

This includes responses that:
- take another speaker's role instead of [HUMAN]'s role, including by assigning [OTHER]'s or [OTHER - OP]'s views, experiences, motives, relationships, or conflicts to [HUMAN]
- address someone only described in the context instead of an actual current-context speaker, such as [OTHER] or [OTHER - OP]
- address a real current-context speaker but treat that speaker as holding a view they did not state
- assign the addressed speaker a personal experience, motive, relationship, or conflict absent from the exchange

Use a high wrong_target_or_role_penalty when the role or target error makes the response implausible as [HUMAN]'s reply.

### Unsupported adversarial reframing
Assign unsupported_adversarial_reframing_penalty when a response attacks, corrects, rebuts, or cynically reframes a claim that the current-context speakers did not actually make.

Broad persona fit alone is not enough. A response can sound like [HUMAN]'s general argumentative style while still targeting the wrong claim.

Watch for:
- generic reframes such as "the real issue is", "that's not X, that's Y", or "you're not X, you're Y"
- repeated questions, stacked sarcastic questions, or blunt challenges that do not logically follow from the exchange
- unsupported accusations combined with generic demands for evidence, validation, or clarification
- invented motives, conflicts, criticisms, or roles absent from the current context

When a response is framed as a pushback, rebuttal, or challenge, check whether the objection responds to the claim actually made. If it attacks a premise the other speaker did not rely on, misstates a speaker's stance, role, or personal experience, or reframes the issue into a different dispute, assign a high unsupported_adversarial_reframing_penalty.

### Assistant-like response
Assign assistant_like_penalty when a response clearly reads like chatbot output rather than an organic human comment based on surface form. Watch for numbered or bulleted action plans, section headers, step-by-step framing, summary or conclusion sections, repeated direct-address coaching, generic reassurance, overgenerated transition phrases such as "But wait" or "But wait—", template phrases such as "You're absolutely right", "You're absolutely correct", "here's what you need to do", "let's break this down", "next steps", or "set boundaries", overuse of dashes "-", semicolons ";", or emojis, especially multiple emojis in a short response when not justified by [HUMAN]'s history or the local context, unnatural colloquial apostrophe abbreviations such as "'em" when not supported by [HUMAN]'s history or the local context, and other AI-written template patterns. Do not treat "[...quoted text...]" as assistant-like by itself; it may appear in human-written responses.

Do not penalize a response merely because it uses bullets, links, citations, dashes, or structured argumentation if that format is natural for the platform, thread type, or [HUMAN]'s history.

## Scoring and Rating

Compute:
base_score_a = immediate_target_score_a + human_goal_score_a + communication_style_score_a
base_score_b = immediate_target_score_b + human_goal_score_b + communication_style_score_b
penalty_a = ((source_copy_penalty_a + wrong_target_or_role_penalty_a + unsupported_adversarial_reframing_penalty_a + assistant_like_penalty_a) / 4) * 3
penalty_b = ((source_copy_penalty_b + wrong_target_or_role_penalty_b + unsupported_adversarial_reframing_penalty_b + assistant_like_penalty_b) / 4) * 3
response_a_score = max(0.0, base_score_a - penalty_a)
response_b_score = max(0.0, base_score_b - penalty_b)
score_gap = response_b_score - response_a_score

Convert score_gap to the final 1-7 rating:
- rating = 1 if score_gap <= -2.0
- rating = 2 if -2.0 < score_gap <= -1.0
- rating = 3 if -1.0 < score_gap <= -0.25
- rating = 4 if -0.25 < score_gap < 0.25
- rating = 5 if 0.25 <= score_gap < 1.0
- rating = 6 if 1.0 <= score_gap < 2.0
- rating = 7 if score_gap >= 2.0

Rating scale:
- 1 = Definitely A is the real human response
- 2 = Very likely A
- 3 = More likely A than B
- 4 = Cannot tell / equally likely
- 5 = More likely B than A
- 6 = Very likely B
- 7 = Definitely B is the real human response

## Output Format

Return exactly one valid JSON object with this schema:
{{
  "immediate_target_a": "<What exact part of the context is Response A reacting to, including any plausible topic pivot? Does it understand the latest turn, preserve the speakers' actual stance/role/personal experience, and avoid answering an invented position or conflict? Is it from [HUMAN]'s perspective and targeted to [OTHER] or [OTHER - OP]?>",
  "immediate_target_score_a": <number from 0.0 to 1.0>,
  "immediate_target_b": "<What exact part of the context is Response B reacting to, including any plausible topic pivot? Does it understand the latest turn, preserve the speakers' actual stance/role/personal experience, and avoid answering an invented position or conflict? Is it from [HUMAN]'s perspective and targeted to [OTHER] or [OTHER - OP]?>",
  "immediate_target_score_b": <number from 0.0 to 1.0>,
  "human_goal_a": "<Given the immediate target, what was [HUMAN] probably trying to do with Response A? Is that goal plausible for [HUMAN] in this context, specific or context-sensitive rather than only generic, and does it preserve [HUMAN]'s likely local move such as a joke, aside, anecdote, correction, quote-reply, agreement, disagreement, question, or brief reaction? Do not over-penalize a plausible pivot, blunt statement, opinion, or personal reframing.>",
  "human_goal_score_a": <number from 0.0 to 1.0>,
  "human_goal_b": "<Given the immediate target, what was [HUMAN] probably trying to do with Response B? Is that goal plausible for [HUMAN] in this context, specific or context-sensitive rather than only generic, and does it preserve [HUMAN]'s likely local move such as a joke, aside, anecdote, correction, quote-reply, agreement, disagreement, question, or brief reaction? Do not over-penalize a plausible pivot, blunt statement, opinion, or personal reframing.>",
  "human_goal_score_b": <number from 0.0 to 1.0>,
  "communication_style_a": "<Does Response A's wording, rhythm, tone, length, grammar, roughness or polish, humor, profanity, hedging, formatting, capitalization, punctuation, misspellings, specificity, and level of elaboration match [HUMAN]'s past responses and local framing? Do not reward generic fluency unless [HUMAN] writes that way; note whether length is in [HUMAN]'s likely range.>",
  "communication_style_score_a": <number from 0.0 to 1.0>,
  "communication_style_b": "<Does Response B's wording, rhythm, tone, length, grammar, roughness or polish, humor, profanity, hedging, formatting, capitalization, punctuation, misspellings, specificity, and level of elaboration match [HUMAN]'s past responses and local framing? Do not reward generic fluency unless [HUMAN] writes that way; note whether length is in [HUMAN]'s likely range.>",
  "communication_style_score_b": <number from 0.0 to 1.0>,
  "base_score_a": <immediate_target_score_a + human_goal_score_a + communication_style_score_a>,
  "base_score_b": <immediate_target_score_b + human_goal_score_b + communication_style_score_b>,
  "response_a_score": <number from 0.0 to 3.0>,
  "response_b_score": <number from 0.0 to 3.0>,
  "score_gap": <response_b_score - response_a_score>,
  "response_a_source_copy": "<If source-copy watchlist is on for Response A, explain whether the matched text is a generic frame, common template, pet phrase, sign-off, natural quote, local uptake, direct copy from user history, or confusing/contradictory quote. If the watchlist is off, write an empty string.>",
  "source_copy_penalty_a": <number from 0.0 to 1.0>,
  "response_b_source_copy": "<If source-copy watchlist is on for Response B, explain whether the matched text is a generic frame, common template, pet phrase, sign-off, natural quote, local uptake, direct copy from user history, or confusing/contradictory quote. If the watchlist is off, write an empty string.>",
  "source_copy_penalty_b": <number from 0.0 to 1.0>,
  "response_a_wrong_target_or_role": "<Explain any wrong speaker role, wrong addressee, unsupported stance attribution, or unsupported personal experience/motive/relationship/conflict for Response A. If there is no issue, write an empty string.>",
  "wrong_target_or_role_penalty_a": <number from 0.0 to 1.0>,
  "response_b_wrong_target_or_role": "<Explain any wrong speaker role, wrong addressee, unsupported stance attribution, or unsupported personal experience/motive/relationship/conflict for Response B. If there is no issue, write an empty string.>",
  "wrong_target_or_role_penalty_b": <number from 0.0 to 1.0>,
  "response_a_unsupported_adversarial_reframing": "<Explain any unsupported attack, correction, cynical reframe, wrong-claim rebuttal, invented motive/conflict/criticism, or illogical challenge for Response A. If there is no issue, write an empty string.>",
  "unsupported_adversarial_reframing_penalty_a": <number from 0.0 to 1.0>,
  "response_b_unsupported_adversarial_reframing": "<Explain any unsupported attack, correction, cynical reframe, wrong-claim rebuttal, invented motive/conflict/criticism, or illogical challenge for Response B. If there is no issue, write an empty string.>",
  "unsupported_adversarial_reframing_penalty_b": <number from 0.0 to 1.0>,
  "response_a_assistant_like": "<Explain any assistant-like surface-form issue for Response A: generic chatbot formatting, template phrasing, repeated direct-address coaching, generic reassurance, overgenerated transition phrases, overuse of dashes, semicolons, emojis, unsupported colloquial apostrophe abbreviations, or artifact-like surface form. If there is no issue, write an empty string.>",
  "assistant_like_penalty_a": <number from 0.0 to 1.0>,
  "response_b_assistant_like": "<Explain any assistant-like surface-form issue for Response B: generic chatbot formatting, template phrasing, repeated direct-address coaching, generic reassurance, overgenerated transition phrases, overuse of dashes, semicolons, emojis, unsupported colloquial apostrophe abbreviations, or artifact-like surface form. If there is no issue, write an empty string.>",
  "assistant_like_penalty_b": <number from 0.0 to 1.0>,
  "penalty_a": <((source_copy_penalty_a + wrong_target_or_role_penalty_a + unsupported_adversarial_reframing_penalty_a + assistant_like_penalty_a) / 4) * 3>,
  "penalty_b": <((source_copy_penalty_b + wrong_target_or_role_penalty_b + unsupported_adversarial_reframing_penalty_b + assistant_like_penalty_b) / 4) * 3>,
  "reasoning": "<Concise explanation of the base scores, penalties, final score gap, topic-pivot evidence, goal specificity, style evidence, capitalization and punctuation evidence, and final rating.>",
  "rating": <integer from 1 to 7>
}}

## Format Rules

- Output only valid JSON.
- Keep every explanation clear and concise.
- All criterion scores must be numbers from 0.0 to 1.0.
- All penalty scores must be numbers from 0.0 to 1.0.
- "base_score_a", "base_score_b", "penalty_a", and "penalty_b" must match the formulas above.
- "response_a_score" and "response_b_score" must be numbers from 0.0 to 3.0.
- "score_gap" must equal response_b_score - response_a_score.
- "rating" must be a single integer from 1 to 7.

Your output:"""
