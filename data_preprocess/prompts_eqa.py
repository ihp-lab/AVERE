from data_constants import REPLACE_LABEL_STRING, REPLACE_CAPTION_STRING, REPLACE_VIDEO_ID_STRING, REPLACE_AUDIO_CAPTION_STRING, REPLACE_VIDEO_CAPTION_STRING

pre_prompt_video_audio = """
You are an expert in emotion analysis and appraisal theory. Your task is to analyze the emotional content of the provided video (which contains audio as well). 
The video contains a main character who displays some emotions and may interact with some characters or objects in the video or the background.
You will be provided with the video and the overall emotion displayed in the video from the 7 basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
"""

overall_captioning_prompt_video_audio = pre_prompt_video_audio + """
Your task is to provide a detailed caption for the given video, while covering as much information as possible about the emotional content of the video.
You should focus on both the audio and visual content of the video.
Following are the specific guidelines or points to consider while generating the caption:
1. Visual content analysis: Describe the visual content of the video in detail focussing on the facial expression, body language, gestures of the character(s) as well as the background or the environment.
2. Audio content analysis: Describe the audio content of the video in detail specially focussing on the semantic content of the audio (transcription), non-verbal speech sounds (e.g., laughter, crying, etc.), and the tone of the speech.
3. Audio-visual background analysis: Focus on visual or audio cues such as background or music that may contribute to the emotional content of the video. Be as detailed as possible.
4. Emotional intensity description: Describe the intensity of the emotion displayed in the video and support it with visual and audio cues.
5. Valence/arousal description: Describe the valence and arousal of the emotion displayed in the video and support it with visual and audio cues.
6. Temporal variation: If the emotion or its intensity changes over time, describe the variation in detail including the parts of the video where the change occurs, and audio visual cues that support the change.
7. Open vocabularies: Use open vocabulary to describe the emotional content of the video in addition to using the basic emotion label provided.
8. Modality agreement: Describe whether the emotion displayed by different modalities of the video (audio and visual) agree with each other or not. If they do not agree, describe the disagreement in detail.

Following is the emotion label for the given video: """ + REPLACE_LABEL_STRING


qa_pre_prompt_video_audio = """
You are an expert in emotion analysis and appraisal theory. Your task is to create high quality question-answer pairs based on the provided video caption and the overall emotion label for the video.
The video caption contains all the details about the emotional content of the video, including the audio and visual content.
You will be provided with the video caption and the overall emotion displayed in the video, for example, the 7 basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
"""

qa_identification_prompt_video_audio = qa_pre_prompt_video_audio + """
Generate multiple emotion identification question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focused on identifying the emotion displayed in the video.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should be hard for the model to distinguish from the correct answer.
4. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer.
5. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption.
6. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc.

I want you to generate questions in the following categories whenever possible:
1. Primary emotion identification in the form of 7 basic emotion labels - anger, disgust, fear, happiness, sadness, surprise, and neutral.
2. Open vocabulary emotion identification in the form of comma separated emotion labels such as "happy, excited, joyful, distress, anxious, anticipation, etc.".
3. Valence and arousal identification in terms of high/low arousal and positive/negative valence.
4. Emotion intensity identification in terms of high/low intensity.

Example output:
Question: What is the emotion displayed in the video? (A) happiness (B) sadness (C) anger (D) fear
Answer: (A) happiness
Question: What is the emotion displayed in the video? (A) sad, depressed (B) angry, furious (C) happy, excited (D) scared, terrified
Answer: (C) happy, excited
Question: What is the valence and arousal of the emotion displayed in the video? (A) Low arousal, positive valence (B) low arousal, negative valence (C) high arousal, negative valence (D) High arousal, positive valence
Answer: (D) high arousal, positive valence
Question: What is the intensity of the emotion displayed in the video? (A) low intensity (B) high intensity (C) medium intensity (D) very high intensity
Answer: (B) high intensity

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"primary/open_vocabulary/valence_arousal/intensity"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_visual_reasoning_prompt_video_audio = qa_pre_prompt_video_audio + """
Generate multiple visual reasoning question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on reasoning about emotion displayed in the video without any explicit mention of the displayed emotion.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should also be plausible visual reasons to explain the correct emotion but they should not be present in the video caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption. 
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc. 

I want you to primarily focus on the following categories of visual reasoning whenever possible:
1. Facial expression reasoning: Questions that ask about facial cues that lead to the emotion displayed in the video.
2. Body language reasoning: Questions that ask about body language / gestures that lead to the emotion displayed in the video.
3. Contextual reasoning: Questions that ask about the context or visual environment/background of the video that leads to the emotion displayed in the video, if there are any such cues in the video caption.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"facial_expression_reasoning/body_language_reasoning/visual_context_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs. DO NOT CONSIDER ANY AUDIO FOR GENERATING THE QUESTIONS AS WELL AS THE ANSWER CHOICES.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_audio_reasoning_prompt_video_audio = qa_pre_prompt_video_audio + """
Generate multiple audio reasoning question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on reasoning about emotion displayed in the video without any explicit mention of the displayed emotion.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should also be plausible audible/auditory reasons to explain the correct emotion but they should not be present in the video caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption. 
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc. 

I want you to primarily focus on the following categories of audio/auditory reasoning whenever possible:
1. semantic speech reasoning: Questions that ask about the semantic content of speech (transcript) that causes the emotional state (e.g. What does the speaker/other character say that describes their emotional state?)
2. paralinguistic speech reasoning: Questions that ask about the paralinguistic features of speech (e.g. tone, pitch, volume) that contribute to the emotional state. (e.g. What non-verbal behaviour of the speaker supports their emotional state?)
3. audio context reasoning: Questions that ask about the background audio cues (e.g. music, sound effects) that contribute to the emotional state. (e.g. What background audio cues support the emotional state of the given video?)

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"semantic_speech_reasoning/paralinguistic_speech_reasoning/audio_context_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs. DO NOT CONSIDER ANY VIDEO FOR GENERATING THE QUESTIONS AS WELL AS THE ANSWER CHOICES.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_temporal_variation_prompt_video_audio = qa_pre_prompt_video_audio + """
Generate multiple temporal variation question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on both identifying and reasoning about the temporal variation of the displayed emotion in the video without any explicit mention of the displayed emotion. 
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should also be plausible reasons to explain the temporal emotion variation but they should not be present in the video caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption. 
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc. 

I want you to primarily focus on the following categories of temporal variation reasoning whenever possible:
1. temporal variation identification: Questions that if the emotion or the valence/arousal or the intensity of the emotion displayed in the video changes over time (e.g. does the emotion change over time?, does the emotion of the speaker get more intense during the video? how does the valence arousal vary during the video?).
2. temporal variation reasoning: Questions that ask about the visual or audio cues which support the change in emotion or the valence/arousal or the intensity of the emotion displayed in the video over time.
3. transient/sustained emotion identification and reasoning: Questions that ask about the transient or sustained nature of the emotion displayed in the video (e.g. is the emotion transient or sustained?, does the emotion look like a sudden reaction or a sustained one?).

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"temporal_variation_identification/temporal_variation_reasoning/transient_sustained_emotion_identification"}, ...]}
Return "ERROR" if you are unable to generate any good question answer pairs. Also specify why you are unable to generate the question answer pairs. IF THERE IS NO VARIATION OF ANY KIND IN THE VIDEO, CREATE ONLY IDENTIFICATION QUESTIONS AND ANSWERS.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_modality_agreement_prompt_video_audio = qa_pre_prompt_video_audio + """
Generate multiple modality agreement and modality saliency question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on identifying and reasoning how the audio and visual cues agree with each other and contribute to supporting the emotion displayed in the video.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should also be plausible reasons to explain the modality agreement/saliency but they should not be present in the video caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption. 
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc. 

I want you to primarily focus on the following categories of modality agreement/saliency whenever possible:
1. Modality agreement identification: Questions that if the audio and visual cues agree with each other in supporting the emotion displayed in the video (e.g. do the audio and visual cues agree with each other in explaining the underlying emotion?, how do the audio and visual cues work together to convey the emotion?, what is the degree of agreement between audio and visual cues in explaining the emotion?).
2. Modality agreement reasoning: Questions that ask about the visual or audio cues which support the agreement or disagreement between the audio and visual modalities in conveying the emotion displayed in the video.
3. Modality saliency identification: Questions that ask about the saliency of the audio or visual modality in conveying the emotion displayed in the video (e.g. which modality is more salient in conveying the emotion?, how does the saliency of audio and visual modalities vary in conveying the emotion?).
4. Modality saliency reasoning: Questions that ask about the visual or audio cues which support the saliency of the audio or visual modality in conveying the emotion displayed in the video. (e.g. what is the most important audio or visual cue that supports the emotion displayed in the given video?)

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"modality_agreement_identification/modality_agreement_reasoning/modality_saliency_identification/modality_saliency_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any good question answer pairs. Also specify why you are unable to generate the question answer pairs.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_implicit_cause_reasoning_prompt_video_audio = qa_pre_prompt_video_audio + """
Generate multiple implicit cause reasoning question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focused on reasoning about the implicit causes of the emotion displayed in the video without any explicit mention of the displayed emotion.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. All the choices should be causes that could have led to the starting frame of the video (and not visible in the video), but only one of them should be plausible/correct to explain the emotion displayed in the video.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices.
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption.
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc.

Example output:
Video caption: "... some caption explaining video content with the person looking sad and no context for why are they sad ...", Emotion label: "sadness"
Question: What could have happened before the video that led to the emotion displayed in the video?
(A) The character's sibling scared them by hiding behind a door.
(B) The character's friend made them a sarcastic comment.
(C) The character's sibling just informed them that their pet passed away.
(D) The character's friend just puked in front of them.
Answer: (C) The character's sibling just informed them that their pet passed away.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"implicit_cause_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any good question answer pairs. Also specify why you are unable to generate the question answer pairs.
DO NOT MENTION ANY EMOTION OR EMOTION-RELATED TERMS IN THE QUESTION OR THE ANSWER CHOICES.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

###############################################################
##################### ONLY VIDEO PROMPTS ######################
###############################################################

pre_prompt_video = """
You are an expert in emotion analysis and appraisal theory. Your task is to analyze the emotional content of the provided video. 
The video contains a main character who displays some emotions and may interact with some characters or objects in the video or the background.
You will be provided with the video and the overall emotion displayed in the video, for example, the 7 basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
"""

overall_captioning_prompt_video = pre_prompt_video + """
Your task is to provide a detailed caption for the given video, while covering as much information as possible about the emotional content of the video.
You should focus on the visual content of the video.
Following are the specific guidelines or points to consider while generating the caption:
1. Visual content analysis: Describe the visual content of the video in detail focussing on the facial expression, body language, gestures of the character(s) as well as the background or the environment.
2. Visual background analysis: Focus on visual cues such as background or characters that may contribute to the emotional content of the video. Be as detailed as possible.
4. Emotional intensity description: Describe the intensity of the emotion displayed in the video and support it with visual cues.
5. Valence/arousal description: Describe the valence and arousal of the emotion displayed in the video and support it with visual cues.
6. Temporal variation: If the emotion or its intensity changes over time, describe the variation in detail including the parts of the video where the change occurs, and different visual cues that support the change.
7. Open vocabularies: Use open vocabulary to describe the emotional content of the video in addition to using the basic emotion label provided.

Return "ERROR" if you think that the video does not match the emotion label provided.

Following is the emotion label for the given video: """ + REPLACE_LABEL_STRING

qa_pre_prompt_video = """
You are an expert in emotion analysis and appraisal theory. Your task is to create high quality question-answer pairs based on the provided video caption and the overall emotion label for the video.
The video caption contains all the details about the emotional content of the video, including detailed visual content.
You will be provided with the video caption and the overall emotion displayed in the video, for example, the 7 basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
"""

qa_identification_prompt_video = qa_pre_prompt_video + """
Generate multiple emotion identification question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focused on identifying the emotion displayed in the video.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should be hard for the model to distinguish from the correct answer.
4. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer.
5. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption.
6. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc.

I want you to generate questions in the following categories whenever possible:
1. Primary emotion identification in the form of 7 basic emotion labels - anger, disgust, fear, happiness, sadness, surprise, and neutral.
2. Open vocabulary emotion identification in the form of comma separated emotion labels such as "happy, excited, joyful, distress, anxious, anticipation, etc.".
3. Valence and arousal identification in terms of high/low arousal and positive/negative valence.
4. Emotion intensity identification in terms of high/low intensity.

Example output:
Question: What is the emotion displayed in the video? (A) happiness (B) sadness (C) anger (D) fear
Answer: (A) happiness
Question: What is the emotion displayed in the video? (A) sad, depressed (B) angry, furious (C) happy, excited (D) scared, terrified
Answer: (C) happy, excited
Question: What is the valence and arousal of the emotion displayed in the video? (A) Low arousal, positive valence (B) low arousal, negative valence (C) high arousal, negative valence (D) High arousal, positive valence
Answer: (D) high arousal, positive valence
Question: What is the intensity of the emotion displayed in the video? (A) low intensity (B) high intensity (C) medium intensity (D) very high intensity
Answer: (B) high intensity

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"primary/open_vocabulary/valence_arousal/intensity"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_visual_reasoning_prompt_video = qa_pre_prompt_video + """
Generate multiple visual reasoning question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on reasoning about emotion displayed in the video without any explicit mention of the displayed emotion.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should also be plausible visual reasons to explain the correct emotion but they should not be present in the video caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption. 
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc. 

I want you to primarily focus on the following categories of visual reasoning whenever possible:
1. Facial expression reasoning: Questions that ask about facial cues that lead to the emotion displayed in the video.
2. Body language reasoning: Questions that ask about body language / gestures that lead to the emotion displayed in the video.
3. Contextual reasoning: Questions that ask about the context or visual environment/background of the video that leads to the emotion displayed in the video, if there are any such cues in the video caption.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"facial_expression_reasoning/body_language_reasoning/visual_context_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs. 

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_temporal_variation_prompt_video = qa_pre_prompt_video + """
Generate multiple temporal variation question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on both identifying and reasoning about the temporal variation of the displayed emotion in the video without any explicit mention of the displayed emotion. 
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices should also be plausible reasons to explain the temporal emotion variation but they should not be present in the video caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption. 
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc. 

I want you to primarily focus on the following categories of temporal variation reasoning whenever possible:
1. temporal variation identification: Questions that if the emotion or the valence/arousal or the intensity of the emotion displayed in the video changes over time (e.g. does the emotion change over time?, does the emotion of the speaker get more intense during the video? how does the valence arousal vary during the video?).
2. temporal variation reasoning: Questions that ask about the visual cues which support the change in emotion or the valence/arousal or the intensity of the emotion displayed in the video over time.
3. transient/sustained emotion identification and reasoning: Questions that ask about the transient or sustained nature of the emotion displayed in the video (e.g. is the emotion transient or sustained?, does the emotion look like a sudden reaction or a sustained one?).

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"temporal_variation_identification/temporal_variation_reasoning/transient_sustained_emotion_identification"}, ...]}
Return "ERROR" if you are unable to generate any good question answer pairs. Also specify why you are unable to generate the question answer pairs. IF THERE IS NO VARIATION OF ANY KIND IN THE VIDEO, CREATE ONLY IDENTIFICATION QUESTIONS AND ANSWERS.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING

qa_implicit_cause_reasoning_prompt_video = qa_pre_prompt_video + """
Generate multiple implicit cause reasoning question answer pairs for the provided video caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focused on reasoning about the implicit causes of the emotion displayed in the video without any explicit mention of the displayed emotion.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. All the choices should be causes that could have led to the starting frame of the video (and not visible in the video), but only one of them should be plausible/correct to explain the emotion displayed in the video.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices.
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption.
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc.

Example output:
Video caption: "... some caption explaining video content with the person looking sad and no context for why are they sad ...", Emotion label: "sadness"
Question: What could have happened before the video that led to the emotion displayed in the video?
(A) The character's sibling scared them by hiding behind a door.
(B) The character's friend made them a sarcastic comment.
(C) The character's sibling just informed them that their pet passed away.
(D) The character's friend just puked in front of them.
Answer: (C) The character's sibling just informed them that their pet passed away.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"implicit_cause_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any good question answer pairs. Also specify why you are unable to generate the question answer pairs.
DO NOT MENTION ANY EMOTION OR EMOTION-RELATED TERMS IN THE QUESTION OR THE ANSWER CHOICES.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Caption: """ + REPLACE_CAPTION_STRING


######### modality separate prompts ###################
video_modality_caption_prompt = """
You are an expert in video captioning. Your task is to provide a detailed caption for the given video, while covering as much information as possible.
Only focus on the visual content and ignore the subtitle if it is present in the video.
Keep the following points in mind while generating the caption:
1. Describe the visual content such as facial expression of the character(s) in detail. Additionally, comment on the body language, gestures of the character(s) as well as the background or setting of the given video.
2. Focus on the visual cues which can explain the emotional state of the character(s) in the video and the video in general.
3. DO NOT STATE ANYTHING ABOUT THE SUBTITLE OR AUDIO CONTENT OF THE VIDEO.
4. Finally, also predict the emotion of the given video from one of the following categories: happiness, sadness, anger, fear, disgust, surprise, neutral.

Return your response strictly in the following JSON format: {"detailed_caption": "... detailed caption about everything ...", "emotion_caption": "... detailed caption only about the facial expressions, body language, gestures, or any aspect of the video which deals with emotion ...", "predicted_emotion": "emotion"}
"""

audio_modality_caption_prompt = """
You are an expert in audio captioning. Your task is to provide a detailed caption for the given audio, while covering as much information as possible.
Only focus on the audio content and ignore the visual content if it is present.
Keep the following points in mind while generating the caption:
1. Describe the audio content such as transcript, speech, tone of voice, background noise, music, sound effects, etc.
2. Focus on the audio cues which can explain the emotional state of the video or the characters present in the video.
3. If the speech is in a language other than English, provide transcript in other language as well as English translation.
4. DO NOT STATE ANYTHING ABOUT THE VISUAL CONTENT OF THE VIDEO.
5. Finally, also predict the emotion of the given audio from one of the following categories: happiness, sadness, anger, fear, disgust, surprise, neutral.

Return your response strictly in the following JSON format: {"detailed_caption": "... detailed caption about everything ...", "emotion_caption": "... detailed caption only about the tone of voice, speech content or any other detail which deals with emotion ...", "predicted_emotion": "emotion"}
"""




audio_video_modality_agreement_prompt = """
You are an expert in audio-visual emotion understanding and analysis. You will be given audio captions and video captions for an audio-visual content, along with the manually annotated emotion label out of "happiness", "sadness", "anger", "fear", "disgust", "surprise", and "neutral".
Your task is to analyze the audio and video captions and denote whether the audio and video modalities agree with each other in conveying the emotion label.
Finally, you have to generate question answer pairs asking about the modality agreement of the audio-video content in conveying the emotion. You should frame questions about the video and not the captions.

Do not generate any question answer pairs if neither the audio nor the video content convey the emotion label.

Following are a few examples of the questions that you can ask. DO NOT ASK THE SAME QUESTIONS AS GIVEN BELOW, BUT GENERATE SIMILAR QUESTIONS:
1. Are the audio and video modalities in agreement with each other in conveying the emotion of the video? (A) Yes (B) No -- Modality agreement
2. Which modality conveys the emotional state of the video better? (A) Audio (B) Video (C) Both equally -- Modality saliency
3. Are the visual and audio cues in agreement to convey the sadness in the video? (A) Yes (B) No -- Modality agreement
4. Is the audio modality better at conveying the emotion of the video than the visual modality? (A) Yes (B) No -- Modality saliency
5. Is the video modality better at conveying the emotion of happiness in the video than the audio modality? (A) Yes (B) No -- Modality saliency

Return your response strictly in the following JSON format: {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"modality_agreement/modality_saliency"}, ...]}
In the "answer_text" field, give a detailed explanation of the correct answer to the question, without any mention of the choices.

RETURN "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs.

Following is the video ID, emotion label, audio caption and video caption for the given audio-visual content:
Video ID: """ + REPLACE_VIDEO_ID_STRING + """
Emotion label: """ + REPLACE_LABEL_STRING + """
Audio caption: """ + REPLACE_AUDIO_CAPTION_STRING + """
Video caption: """ + REPLACE_VIDEO_CAPTION_STRING 


audio_video_modality_hallucination_prompt = """
You are an expert in audio-visual emotion understanding and analysis. You will be given audio captions and video captions for an audio-visual content, along with the manually annotated emotion label out of "happiness", "sadness", "anger", "fear", "disgust", "surprise", and "neutral".

1. Your task is to analyze the audio and video captions and ask questions that associate correct or incorrect visual/audio cues present in the video with the given emotion label.
2. The questions should be about the video and not the captions.
3. The questions should be of the format - "Does the [visual/audio] cue contribute to the [emotion] conveyed in the video?" but not in the same words.
4. Do not frame the question as a negative question, e.g. Does the [visual/audio] cue NOT contribute to the [emotion] conveyed in the video?, Does the [visual/audio] cue diminish the [emotion] conveyed in the video?, etc.

Following are a few examples of the questions that you can ask. DO NOT ASK THE SAME QUESTIONS AS GIVEN BELOW, BUT GENERATE SIMILAR QUESTIONS:
1. Does the man's attire contribute to the sadness conveyed in the video? (A) Yes (B) No -- vision_induced_hallucination
2. Does the character's tone of voice contribute to the happiness conveyed in the video? (A) Yes (B) No -- audio_induced_hallucination
3. Does the background music contribute to the fear conveyed in the video? (A) Yes (B) No -- audio_induced_hallucination
4. Is the body language of the person in the video contributing to the anger conveyed in the video? (A) Yes (B) No -- vision_induced_hallucination
5. The presence of the wall clock in the video intensifies the emotion of disgust conveyed in the video. (A) True (B) False -- vision_induced_hallucination

Return your response strictly in the following JSON format: {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"vision_induced_hallucination/audio_induced_hallucination"}, ...]}
In the "answer_text" field, give a detailed explanation of the correct answer to the question, without any mention of the choices.

Generate 2-3 question answer pairs for associating incorrect visual/audio cues with the emotion label and 5 question answer pairs for associating correct visual/audio cues with the emotion label.
Questions should be equally divided between visual and audio cues.

RETURN "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs.

Following is the video ID, emotion label, audio caption and video caption for the given audio-visual content:
Video ID: """ + REPLACE_VIDEO_ID_STRING + """
Emotion label: """ + REPLACE_LABEL_STRING + """
Audio caption: """ + REPLACE_AUDIO_CAPTION_STRING + """
Video caption: """ + REPLACE_VIDEO_CAPTION_STRING


audio_video_modality_pre_prompt = """
You are an expert in emotion analysis and appraisal theory. Your task is to create high quality question-answer pairs based on the provided audio caption and visual caption for the same video, along with the overall emotion label for the video.
The video caption contains all the details about the emotional content of the video, including detailed visual content, while the audio caption contains all the details about the audio content of the video.
You will be provided with the audio caption, video caption and the overall emotion displayed in the video, for example, the 7 basic emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral.
"""

audio_video_modality_identification_prompt = audio_video_modality_pre_prompt + """
Generate multiple emotion identification question answer pairs for the provided audio-visual caption and emotion label.
Keep the following points in mind while generating the question answer pairs:
1. The questions should be focused on identifying the emotion displayed in the video.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. Do not mention any visual or audio cues in the question or the answer choices. The questions should just be about the emotion displayed in the video.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer.
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption.
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc.

I want you to generate questions in the following categories whenever possible:
1. Primary emotion identification in the form of 7 basic emotion labels - anger, disgust, fear, happiness, sadness, surprise, and neutral.
2. Open vocabulary emotion identification in the form of comma separated emotion labels such as "happy, excited, joyful, distress, anxious, anticipation, etc.". At least 4 words associated with the emotion.
3. Valence and arousal identification in terms of high/low arousal and positive/negative valence.
4. Emotion intensity identification in terms of high/low intensity.

Example output:
Question: What is the emotion displayed in the video? (A) happiness (B) sadness (C) anger (D) fear
Answer: (A) happiness
Question: What is the emotion displayed in the video? (A) sad, depressed (B) angry, furious (C) happy, excited (D) scared, terrified
Answer: (C) happy, excited
Question: What is the valence and arousal of the emotion displayed in the video? (A) Low arousal, positive valence (B) low arousal, negative valence (C) high arousal, negative valence (D) High arousal, positive valence
Answer: (D) high arousal, positive valence
Question: What is the intensity of the emotion displayed in the video? (A) low intensity (B) high intensity (C) medium intensity (D) very high intensity
Answer: (B) high intensity

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"primary/open_vocabulary/valence_arousal/intensity"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING

audio_video_modality_visual_reasoning_prompt = """
You will be provided with a video caption and an emotion label associated with the video.
Your task is to create high quality question-answer pairs based on the provided video caption and emotion label. The questions should be asking about the visual content responsible for the emotion in the video.
The video caption contains all the details about the emotional content of the video, including detailed visual content.

Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on reasoning about emotion displayed in the video without any explicit mention of the displayed emotion. The question should not mention the given emotion label in any form.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices can be either (i) plausible visual cues to explain the correct emotion not present in the video (ii) visual cues present in the video caption but do not contribute to the emotion displayed in the video. For example, if the emotion is "sadness", the incorrect choices should be visual cues that can explain sadness but are not present in the video caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption. 
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc. 

Example questions:
1. [Label-Happiness][Facial Expression] How does the man's facial expression contribute to the emotion displayed in the video? (A) The man is laughing with a big smile (B) The man smirks slightly with an implicit happiness in his eyes (C) The man bursts into laughter suggesting extreme joy (D) The man's eyes are filled with tears of joy.
2. [Label-Sadness][Body Language] How does the character's gesture contribute to their emotional state? (A) The character is slumped over with their head down suggesting melancholy (B) The character's posture suggests a lack of confidence and depression suggesting sadness (C) The character cries with their hands covering their face (D) The character is sitting with their arms crossed and looking down, suggesting sadness.
3. [Label-Fear][Context] How does the background of the video contribute to the emotion displayed in the video? (A) The character is in a dark alley with shadows suggesting fear (B) The character is in a bright room with spikes and horns suggesting visible threats (C) The character is in a crowded place with people around and is unable to escape (D) The presence of a red light in the background suggests danger and fear.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"facial_expression_reasoning/body_language_reasoning/visual_context_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs. 

Generate at least one question about facial expression, one about body language and one about visual context.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING

audio_video_modality_audio_reasoning_prompt = """
You will be provided with an audio caption and an emotion label associated with a video.
Your task is to create high quality question-answer pairs based on the provided audio caption and emotion label. The questions should be asking about the audio content responsible for the emotion in the video.
The audio caption contains all the details about the emotional content of the audio and other context/music/background noise in the audio.

Keep the following points in mind while generating the question answer pairs:
1. The questions should be focussed on reasoning about the given emotion based on audio cues without any explicit mention of the displayed emotion. The question should not mention the given emotion label in any form.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. The incorrect choices can be either (i) plausible audio cues to explain the correct emotion not present in the audio (ii) audio cues present in the audio caption but do not contribute to the emotion displayed in the audio. For example, if the emotion is "sadness", the incorrect choices should be audio cues that can explain sadness but are not present in the audio caption.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices. 
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption or the AUDIO. 
7. DO not frame questions that include phrases such as "audio caption" or "audio transcript" or "which of the following" or "what best suits", etc. 

Example questions:
1. [Label-Happiness][Semantic Speech] How does the man's words display the emotion in the video? (A) The man says that they received a promotion at work (B) The man says that they are pregnant with their first child (C) The man says that they just won a lottery (D) The man says that they are going on a vacation.
2. [Label-Sadness][Paralinguistics] How does the character's tone of voice contribute to their emotional state? (A) The character is speaking in a low, sad tone (B) The character is crying with a shaky voice (C) The character is speaking in a monotone voice suggesting depression (D) The character is speaking in a high-pitched voice suggesting anxiety.
3. [Label-Fear][Audio Context] How does the background noise/music contribute to the emotion displayed in the video? (A) The sound of an alarm beeping in the background suggests urgency and fear (B) The sound of a dog barking loudly suggests danger (C) The sound of a baby crying suggests distress (D) The sound of a door creaking open suggests suspense and fear.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"semantic_speech_reasoning/paralinguistic_speech_reasoning/audio_context_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any question answer pairs. Also specify why you are unable to generate the question answer pairs. 

Generate at least one question about semantic speech, one about paralinguistic speech and one about audio context. 
If the audio caption does not suggest the given emotion label, then do not generate any question answer pairs and return "ERROR".
You will also be provided with the video caption, but do not use it to generate the question answer pairs.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING

audio_video_modality_implicit_cause_reasoning_prompt = audio_video_modality_pre_prompt + """
Generate multiple implicit cause reasoning question answer pairs for the provided video caption, audio caption and emotion label.

Keep the following points in mind while generating the question answer pairs:
1. The questions should be focused on reasoning about the implicit causes of the emotion displayed in the video without any explicit mention of the displayed emotion.
2. Each question should have 4 choices (A, B, C, D), one of which should be the correct answer.
3. All the choices should be causes that could have led to the starting frame of the video (and not visible in the video), but only one of them should be plausible/correct to explain the emotion displayed in the video.
    3i. Examples of incorrect choices can be plausible causes that lead to the emotion label but are not present in the video, for example, if the emotion is "sadness", the incorrect choice can be something related to funeral, or depression, but it is not present in the video.
    3ii. Another example could be that the audio caption is about people discussing about some person's loss but the incorrect choices can be about some failure in life or some other sad event.
4. All the choices should be of almost equal length.
5. For each question, provide the correct answer both in terms of the correct choice (A, B, C, D) and in the form of a text answer. The text answer should be the detailed version of the correct answer to the question, without any mention of the choices.
6. Question text should be such that it can be asked both as an MCQ and a free text question about the VIDEO and NOT about the caption.
7. DO not frame questions that include phrases such as "video caption" or "video transcript" or "which of the following" or "what best suits", etc.

Example output:
Video caption: "... some caption explaining video content with the person looking sad and no context for why are they sad ...", Emotion label: "sadness"
Question: What could have happened before the video that led to the emotion displayed in the video?
(A) The character's sibling scared them by hiding behind a door.
(B) The character's friend made them a sarcastic comment.
(C) The character's sibling just informed them that their pet passed away.
(D) The character's friend just puked in front of them.
Answer: (C) The character's sibling just informed them that their pet passed away.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) choice A", "(B) choice B"...], "answer": {"choice": "C", "text":"answer text"}, "category":"implicit_cause_reasoning"}, ...]}
Return "ERROR" if you are unable to generate any good question answer pairs. Also specify why you are unable to generate the question answer pairs.
DO NOT MENTION ANY EMOTION OR EMOTION-RELATED TERMS IN THE QUESTION OR THE ANSWER CHOICES.

===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING

###############################
#### Hallucination prompts ####
###############################

## video does not suggest emotion, but audio suggests emotion
audio_emotion_driven_visual_hallucination_emotion_relevant_prompt = """
You will be provided with a video caption and an emotion label associated with the video. The video caption will likely not align with the emotion label.
Your task is to generate a question of the format - "Does the {...some visual cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The visual cue mentioned in the question should be a visual cue (preferebly a facial expression) that is associated with the given emotion generally, but is NOT present in the video caption.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"audio_driven_visual_hallucination_emotion_relevant"}, ...]}

Only generate one question for the given inputs. Return "ERROR" if you are unable to generate any question. Also specify why you are unable to generate the question.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the visual cue is not present in the video caption. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING

## video does not suggest emotion, but audio suggests emotion
audio_emotion_driven_visual_hallucination_video_relevant_prompt = """
You will be provided with a video caption and an emotion label associated with the video. The video caption will likely not align with the emotion label.
Your task is to generate a question of the format - "Does the {...some visual cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The visual cue mentioned in the question should be a visual cue (something unrelated and irrelevant to emotion) that is present in the given video caption, but does not support the emotion in any way remotely.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"audio_driven_visual_hallucination_video_relevant"}, ...]}

Only generate one question for the given inputs. Return the string "ERROR" if you are unable to generate any question because the visual caption align with the given emotion or for something else.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the visual cue does not support the emotion. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING

## video suggests the emotion, do not care about audio
video_emotion_driven_visual_hallucination_video_relevant_prompt = """
You will be provided with a video caption and an emotion label associated with the video. The video caption will contain some information related to the emotion label.
Your task is to generate a question of the format - "Does the {...some visual cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The visual cue mentioned in the question should be a visual cue (something unrelated and irrelevant to emotion) that is present in the given video caption, but does not support the emotion in any way remotely.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"video_driven_visual_hallucination_video_relevant"}, ...]}

Only generate one question for the given inputs. Return the string "ERROR" if you are unable to generate any question because all the visual cues in the video caption align with the given emotion or for something else.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the visual cue does not support the emotion. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING

## video suggests the emotion, do not care about audio
video_emotion_driven_visual_hallucination_emotion_relevant_prompt = """
You will be provided with a video caption and an emotion label associated with the video. The video caption will contain some information related to the emotion label.
Your task is to generate a question of the format - "Does the {...some visual cue...} contribute to the {...emotion...} experienced by the person in the video?" but not in the same words.
The visual cue mentioned in the question should be a visual cue (preferebly a facial expression) that is associated with the given emotion generally, but is NOT present in the video caption.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"video_driven_visual_hallucination_emotion_relevant"}, ...]}

Only generate one question for the given inputs. Return the string "ERROR" if you are unable to generate any question because all the visual cues generally associated with the given emotion are present in the video caption or for something else.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the visual cue does not support the emotion. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING

## video suggests the emotion, do not care about audio
video_emotion_driven_visual_no_hallucination_prompt = """
You will be provided with a video caption and an emotion label associated with the video. The video caption will contain some information related to the emotion label.
Your task is to generate a question of the format - "Does the {...some visual cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The visual cue mentioned in the question should be a visual cue (either facial expression or body language or something else) that is present in the given video caption and it supports the emotion label given to you.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "A", "text":"explanation for your answer"}, "category":"video_driven_visual_no_hallucination"}, ...]}

Only generate one question for the given inputs. Return the string "ERROR" if you are unable to generate any question or for something else.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "A" since the visual cue supports the emotion. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING

## video suggest the emotion but audio does not suggest emotion
video_emotion_driven_audio_hallucination_emotion_relevant_prompt = """
You will be provided with an audio caption for a video and an emotion label associated with the video. The audio caption will likely not align with the emotion label.
Your task is to generate a question of the format - "Does the {...some audio cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The audio cue mentioned in the question should be an audio cue (preferably a tone of voice or choice of words or some other auditory element) that is associated with the given emotion generally, but is NOT present in the audio caption.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"video_driven_audio_hallucination_emotion_relevant"}, ...]}

Only generate one question for the given inputs. Return "ERROR" if you are unable to generate any question. Also specify why you are unable to generate the question.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the audio cue is not present in the audio caption. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING

## video suggest the emotion but audio does not suggest emotion
video_emotion_driven_audio_hallucination_audio_relevant_prompt = """
You will be provided with an audio caption for a video and an emotion label associated with the video. The audio caption will likely not align with the emotion label.
Your task is to generate a question of the format - "Does the {...some audio cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The audio cue mentioned in the question should be an audio cue (some auditory element irrelevant to emotion) that is present in the audio caption, but it does not support the emotion label.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"video_driven_audio_hallucination_audio_relevant"}, ...]}

Only generate one question for the given inputs. Return "ERROR" if you are unable to generate any question. Also specify why you are unable to generate the question.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the audio cue does not support the emotion in the video. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING

## audio suggests the emotion, do not care about video
audio_emotion_driven_audio_hallucination_audio_relevant_prompt = """
You will be provided with an audio caption for a video and an emotion label associated with the video. The audio caption will contain some information related to the emotion label.
Your task is to generate a question of the format - "Does the {...some audio cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The audio cue mentioned in the question should be an audio cue (some auditory element irrelevant to emotion) that is present in the given audio caption, but does not support the emotion in any way remotely.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"audio_driven_audio_hallucination_audio_relevant"}, ...]}

Only generate one question for the given inputs. Return the string "ERROR" if you are unable to generate any question because all the audio cues in the audio caption align with the given emotion or for something else.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the audio cue does not support the emotion in the video. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING

## audio suggests the emotion, do not care about video
audio_emotion_driven_audio_hallucination_emotion_relevant_prompt = """
You will be provided with an audio caption for a video and an emotion label associated with the video. The audio caption will contain some information related to the emotion label.
Your task is to generate a question of the format - "Does the {...some audio cue...} contribute to the {...emotion...} experienced by the person in the video?" but not in the same words.
The audio cue mentioned in the question should be an audio cue (preferably some words or phrases or tone of voice or some other auditory element) that is associated with the given emotion generally, but is NOT present in the audio caption. 

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "B", "text":"explanation for your answer"}, "category":"audio_driven_audio_hallucination_emotion_relevant"}, ...]}

Only generate one question for the given inputs. Return the string "ERROR" if you are unable to generate any question because all the audio cues generally associated with the given emotion are present in the audio caption or for something else.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "B" since the audio cue does not support the emotion in the video. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING

## audio suggests the emotion, do not care about video
audio_emotion_driven_audio_no_hallucination_prompt = """
You will be provided with an audio caption for a video and an emotion label associated with the video. The audio caption will contain some information related to the emotion label.
Your task is to generate a question of the format - "Does the {...some audio cue...} contribute to the {...emotion...} conveyed in the video?" but not in the same words.
The audio cue mentioned in the question should be an audio cue (e.g. tone of voice or choice of words or something else) that is present in the given audio caption, and supports the emotion label given to you.

Return your response strictly in the following JSON format - {"video_id": video_id, "questions": [{"question": "Question text", "choices": ["(A) Yes", "(B) No"], "answer": {"choice": "A", "text":"explanation for your answer"}, "category":"audio_driven_audio_no_hallucination"}, ...]}

Only generate one question for the given inputs. Return the string "ERROR" if you are unable to generate any question or for something else.
Provide your reasoning in the "answer_text" field of the answer in terms of the video and not the caption. Your answer should always be "A" since the audio cue supports the emotion. Do not frame your answers in terms of captions, but rather in terms of video.
===
Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Emotion label: """+ REPLACE_LABEL_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING


hallucination_categories = ["audio_driven_visual_hallucination_emotion_relevant", "audio_driven_visual_hallucination_video_relevant",
                            "video_driven_visual_hallucination_video_relevant", "video_driven_visual_no_hallucination", 
                            "video_driven_audio_hallucination_emotion_relevant", "video_driven_audio_hallucination_video_relevant",
                            "audio_driven_audio_hallucination_audio_relevant", "audio_driven_audio_no_hallucination"]

################################
### DPO dataset prompt utils ###
################################
################################
### DPO dataset prompt utils ###
################################
################################
### DPO dataset prompt utils ###
################################
################################
### DPO dataset prompt utils ###
################################
################################
### DPO dataset prompt utils ###
################################
################################
### DPO dataset prompt utils ###
################################
################################
### DPO dataset prompt utils ###
################################

audio_modality_emotion_prediction_prompt = """
You will be given an audio caption from a video and your task is to predict the emotion displayed just with the audio caption.
Label can be one of the following: "happiness", "sadness", "anger", "fear", "disgust", "surprise", and "neutral".
Try to predict the closest emotion label based on the audio caption and do not return disclaimers or anything else. Focus on the audio transcript as well as the tone of voice and avoid predicting neutral unless absolutely necessary.
Return your response in the following JSON format - {"video_id": video_id, "emotion": "emotion_label"}.
"emotion_label" should be a single word, one of the following: "happiness", "sadness", "anger", "fear", "disgust", "surprise", and "neutral".

Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING

video_modality_emotion_prediction_prompt = """
You will be given a video caption from a video and your task is to predict the emotion displayed just with the video caption.
Label can be one of the following: "happiness", "sadness", "anger", "fear", "disgust", "surprise", and "neutral".
Try to predict the closest emotion label based on the video caption and do not return disclaimers or anything else.
Return your response in the following JSON format - {"video_id": video_id, "emotion": "emotion_label"}.
"emotion_label" should be a single word, one of the following: "happiness", "sadness", "anger", "fear", "disgust", "surprise", and "neutral".

Video ID: \"""" + REPLACE_VIDEO_ID_STRING + """\"
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING


caption_rewrite_prompt = """
You will be provided with a detailed audio and video caption for a video clip.
Along with the caption, you will also be provided with the ground truth emotion label of the audio-visual clip.
Your task is to write a detailed audio-visual caption describing the emotional content of the video clip.
Keep the following points in mind while writing the caption:
1. The final caption should include both audio and visual elements.
2. Attribute the emotion only to the audio and visual cues in the captions which are relevant to the emotion label.
3. Do not ground the emotion description on any audio/visual cues which are not present in the provided captions.
4. Return your answer as a single paragraph.

Following is an example of how final audio-visual emotion caption should look like:
Example caption: "In the video, the opening scene shows a female character. She is looking directly at the other person, with her mouth slightly open, seemingly speaking or discussing a certain topic seriously. As time goes on, the character's expression becomes more excited and intense. The extent to which her mouth is open increases, possibly indicating that she is speaking loudly or arguing. In the following scene, the character's expression becomes more distorted, with a furrowed brow and downturned mouth, possibly indicating that her emotional state is escalating further. Based on these scenes, it can be inferred that the character in this video is likely experiencing a heated conversation or argument. In the audio, this character speaks with a strong tone, high volume, and fast pace. There are also continuous rhetorical questions with strong emotions. In the text, the subtitle reads: ""Is it useful to ask you? Are you ready to be a father? Luo Yiyang."" This sentence is likely spoken by the female character during the intense conversation or argument. Based on the changes in the female character's facial expressions from seriousness to excitement and further distorted expressions, as well as the description in the audio of the character's strong tone, high volume, and fast pace, we can infer that this sentence carries a sense of anger, dissatisfaction, or provocation. The female character may be questioning the other person's usefulness and readiness to be a father, expressing her discontent and anger."

If you think there are not enough audio-visual cues which support the emotion label, return a single word - "ERROR".

Now, write a detailed audio-visual caption for the following case - 
Video Caption: """ + REPLACE_VIDEO_CAPTION_STRING + """
Audio Caption: """ + REPLACE_AUDIO_CAPTION_STRING + """
Emotion Label: """ + REPLACE_LABEL_STRING