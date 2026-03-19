AUDIO_EXTENSIONS = ["wav", "flac"]
VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "flv"]
REPLACE_LABEL_STRING = "REPLACE_GT_LABEL"
REPLACE_CAPTION_STRING = "{VIDEO_CAPTION}"
REPLACE_VIDEO_ID_STRING = "{VIDEO_ID}"
REPLACE_AUDIO_CAPTION_STRING = "{ONLY_AUDIO_CAPTION}"
REPLACE_VIDEO_CAPTION_STRING = "{ONLY_VIDEO_CAPTION}"

# google cloud constants
BUCKET_NAME = "ihp-lab-emotion-reasoning"
BUCKET_AUDIO_FOLDER = "audio/main_data"
BUCKET_VIDEO_FOLDER = "video/main_data"


## instructions
EMOTION_PERCEPTION_INSTRUCTIONS = [
    "<audio>\nDescribe the emotion of the speaker in one word.",
    "<audio>\nUse one word to describe the speaker's emotion.",
    "<audio>\nIdentify the speaker's primary feeling using a single adjective.",
    "<audio>\nWhat single word best captures the speaker's sentiment?",
    "<audio>\nSummarize the speaker's emotional state with one term.",
    "<audio>\nExpress the speaker's mood in just one word.",
    "<audio>\nCharacterize the speaker's feeling in a single word.",
    "<audio>\nProvide a one-word description of the speaker's emotion.",
    "<audio>\nCondense the speaker's emotional state into a single word.",
    "<audio>\nIn one word, what is the speaker feeling?",
    "<audio>\nState the emotion conveyed by the speaker in one word.",
    "<audio>\nDescribe the speaker's affect using a single word.",
    "<audio>\nGive one word to represent the speaker's emotion.",
    "<audio>\nReduce the speaker's feeling to a single adjective.",
    "<audio>\nWhat is the most prominent emotion of the speaker, in one word?",
    "<audio>\nAnswer with a single word: What emotion is the speaker displaying?",
    "<audio>\nExtract the emotion from the speaker's voice and express it in one word.",
    "<audio>\nUsing one word, define the speaker's emotional disposition.",
    "<audio>\nRepresent the emotional state of the speaker with a single term.",
    "<audio>\nWhat single adjective best describes the speaker's feeling?"
]

EMOTION_INSTRUCTIONS = [
    "<audio>\nDescribe the emotion of the speaker in detail.",
    "<audio>\nWhat is the emotion of the speaker and why do you think so?",
    "<audio>\nAnalyze the speaker's emotional state.",
    "<audio>\nIdentify the primary emotion conveyed by the speaker.",
    "<audio>\nExplain the speaker's emotional expression.",
    "<audio>\nWhat emotions are present in the speaker's voice?",
    "<audio>\nDetail the nuances of the speaker's emotional delivery.",
    "<audio>\nHow would you characterize the speaker's emotional tone?",
    "<audio>\nDescribe the intensity of the speaker's emotion.",
    "<audio>\nWhat is the underlying emotion the speaker is experiencing?",
    "<audio>\nDiscuss the speaker's emotional vulnerability.",
    "<audio>\nHow does the speaker's voice reflect their emotional state?",
    "<audio>\nAnalyze the changes in the speaker's emotion throughout the audio.",
    "<audio>\nIdentify any conflicting emotions the speaker might be feeling.",
    "<audio>\nWhat is the overall emotional impact of the speaker's words?",
    "<audio>\nExplain how the speaker's delivery contributes to their emotional expression.",
    "<audio>\nDescribe the emotional arc of the speaker's message.",
    "<audio>\nHow effectively does the speaker convey their emotion?",
    "<audio>\nWhat techniques does the speaker use to express their emotion?",
    "<audio>\nSummarize the speaker's emotional journey."
]

VIDEO_EMOTION_PERCEPTION_INSTRUCTIONS = [
    "<video>\nDescribe the emotion of the person in the video in one word.",
    "<video>\nUse one word to describe the person's emotion in the video.",
    "<video>\nIdentify the person's primary feeling in the video using a single adjective.",
    "<video>\nWhat single word best captures the person's sentiment in the video?",
    "<video>\nSummarize the person's emotional state in the video with one term.",
    "<video>\nExpress the person's mood in the video in just one word.",
    "<video>\nCharacterize the person's feeling in the video in a single word.",
    "<video>\nProvide a one-word description of the person's emotion in the video.",
    "<video>\nCondense the person's emotional state in the video into a single word.",
    "<video>\nIn one word, what is the person in the video feeling?",
    "<video>\nState the emotion conveyed by the person in the video in one word.",
    "<video>\nDescribe the person's affect in the video using a single word.",
    "<video>\nGive one word to represent the person's emotion in the video.",
    "<video>\nReduce the person's feeling in the video to a single adjective.",
    "<video>\nWhat is the most prominent emotion of the person in the video, in one word?",
    "<video>\nAnswer with a single word: What emotion is the person in the video displaying?",
    "<video>\nExtract the emotion from the person's expression in the video and express it in one word.",
    "<video>\nUsing one word, define the person's emotional disposition in the video.",
    "<video>\nRepresent the emotional state of the person in the video with a single term.",
    "<video>\nWhat single adjective best describes the person's feeling in the video?"
]

VIDEO_EMOTION_INSTRUCTIONS = [
    "<video>\nDescribe the emotion of the person in the video in detail.",
    "<video>\nWhat is the emotion of the person in the video and why do you think so?",
    "<video>\nAnalyze the person's emotional state in the video.",
    "<video>\nIdentify the primary emotion conveyed by the person in the video.",
    "<video>\nExplain the person's emotional expression in the video.",
    "<video>\nWhat emotions are present in the person's behavior in the video?",
    "<video>\nDetail the nuances of the person's emotional delivery in the video.",
    "<video>\nHow would you characterize the person's emotional tone in the video?",
    "<video>\nDescribe the intensity of the person's emotion in the video.",
    "<video>\nWhat is the underlying emotion the person is experiencing in the video?",
    "<video>\nDiscuss the person's emotional vulnerability as seen in the video.",
    "<video>\nHow does the person's behavior reflect their emotional state in the video?",
    "<video>\nAnalyze the changes in the person's emotion throughout the video.",
    "<video>\nIdentify any conflicting emotions the person might be feeling in the video.",
    "<video>\nWhat is the overall emotional impact of the person's actions in the video?",
    "<video>\nExplain how the person's behavior contributes to their emotional expression in the video.",
    "<video>\nDescribe the emotional arc of the person's actions in the video.",
    "<video>\nHow effectively does the person convey their emotion in the video?",
    "<video>\nWhat techniques does the person use to express their emotion in the video?",
    "<video>\nSummarize the person's emotional journey in the video."
]

AGE_PERCEPTION_INSTRUCTIONS = [
    "<audio>\nEstimate the age of the speaker in the audio.",
    "<audio>\nWhat age best describes the speaker's voice?",
    "<audio>\nBased on the audio, give your best guess for the speaker's age.",
    "<audio>\nApproximate the speaker's age based on their vocal characteristics.",
    "<audio>\nWhat is the most likely age of the person speaking in this audio?",
    "<audio>\nBased on the speech patterns, estimate the age of the speaker.",
    "<audio>\nDetermine if the speaker is elderly, middle-aged, young adult, or child. Provide a numerical age estimation.",
    "<audio>\nAssign an age to the speaker.",
    "<audio>\nEstimate the speaker's age; is there any indication of maturity in the voice?",
    "<audio>\nEstimate age of the speaker in the given audio file.",
    "<audio>\nProvide a whole number estimate for the age of the speaker in this audio?",
    "<audio>\nEstimate the age of the speaker to the nearest whole number",
    "<audio>\nEstimated age of the speaker in the audio should be?",
    "<audio>\nAnalyze the voice and provide an estimated age to the nearest positive integer.",
    "<audio>\nWhat age can be attributed to the speaker in the given audio?",
    "<audio>\nIdentify the age of the spaeker in the audio."
]

GENDER_PERCEPTION_INSTRUCTIONS = [
    "<audio>\nDetermine the gender of the speaker in the given audio.",
    "<audio>\nDoes the audio sound like that of a male or a female?",
    "<audio>\nWhat is the biological gender of the speaker?",
    "<audio>\nIs the speaker a man or a woman based on the audio?",
    "<audio>\nIdentify the gender of the person speaking in the audio.",
    "<audio>\nAnalyze the audio and classify the speaker as either male or female.",
    "<audio>\nCan you tell if the voice in the audio belongs to a male or female?",
    "<audio>\nClassify the speaker\'s gender based on their vocal characteristics.",
    "<audio>\nWhat is the perceived gender of the speaker in this audio recording?",
    "<audio>\nIs the voice in the audio clip masculine or feminine?",
    "<audio>\nPredict the gender of the individual speaking in the provided audio.",
    "<audio>\nWhat gender is associated with the voice present in the audio file?",
    "<audio>\nBased on the voice, is the speaker likely male or female?",
    "<audio>\nProvide an assessment of the speaker's gender in the audio.",
    "<audio>\nFrom the audio, identify whether the voice is of a male or female individual.",
    "<audio>\nReport the speaker\'s identified gender from the given audio recording.",
    "<audio>\nWhat is the speaker's gender, as determined from the audio sample?",
    "<audio>\nAssess the audio recording to determine the speaker's gender.",
    "<audio>\nWhat gender would you assign to the voice in the supplied audio file?",
    "<audio>\nCharacterize the speaker's gender according to the audio sample."
]

GEMINI_PRICE = {
    "gemini-2.0-flash-001": {
        "input_audio_per_sec": 0.0000125,
        "input_video_per_sec": 0.00001935,
        "input_text_per_million_char": 0.01875,
        "output_text_per_million_char": 0.075
    },
    "gemini-2.0-flash-lite": {
        "input_audio_per_sec": 0.00000938,
        "input_video_per_sec": 0.000009675,
        "input_text_per_million_char": 0.009375,
        "output_text_per_million_char": 0.0375
    }
}