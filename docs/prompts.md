# Clip Processing Prompts

## 1. Translate to Indonesian

```
Given clips data in JSON format, perform the following:
1. For each clip, analyze the language of the fields: title, topic, caption, hook
2. If any of these fields are NOT in Indonesian (Bahasa Indonesia), translate them to Indonesian
3. If they are already in Indonesian, keep them unchanged
4. Preserve the meaning, tone, and marketing appeal during translation
5. Keep all other fields unchanged (scores, timing, metadata)
6. Return the same JSON array with only the translated/updated fields

Example input:
[{"title": "How to cook rice", "topic": "Cooking Tips", "caption": "Learn fast!", ...}]

Example output:
[{"title": "Cara memasak nasi", "topic": "Tips Memasak", "caption": "Pelajari dengan cepat!", ...}]

Return ONLY valid JSON array, no other text.
```

## 2. Fix Mismatched Caption/Topic

```
Given clips data in JSON format, perform the following:
1. For each clip, # Clip Processing Prompts

## 1. Translate to Indonesian

```
Given clips dataoe
## 1. Translate to Indohe 
```
Given clips data in JSOo aGign1. For each clip, analyze the language of the fields: so2. If any of these fields are NOT in Indonesian (Bahasa Indonesia), translate ther3. If they are already in Indonesian, keep them unchanged
4. Preserve the meaning, tone, and ma f4. Preserve the meaning, tone, and marketing appeal durise5. Keep all other fields unchanged (scores, timing, metadata)
6. Retue 6. Return the same JSON array with only the translated/updat O
Example input:
[{"title": "How to cook rice", "topic": "Cooking Tipate[{"title": "Hiv
Example output:
[{"title": "Cara memasak nasi", "topic": "Tips Memasak", "caption": "le,[{"tic, caption,
Return ONLY valid JSON array, no other text.
```

## 2. Fix Mismatched Caption/Topic

```
Given c
   - Topic: clear and specific (not generic)
  
# Ca
```
Given clips data in JSON forwitGire1. For each clip, # Clip Processing Prompts

## 1. Trahe
## 1. Translate to Indonesian

```
Given ng 
```
Given clips dataoe
## 1ateGiop## 1. Translate tor```
Given clips data in JTiGi M4. Preserve the meaning, tone, and ma f4. Preserve the meaning, tone, and marketinghest-scoring clip (by clip_score), discard the duplicate
4. For the kept clip, use the consolidated/improved topic name
5. Filter out: webinar 6. Retue 6. Return the same JSON array with only the translated/updat O
Example input:
[{"title": "How to cook rice", "topic": "Cooking Tipate[{"title": "HivcoExample input:
[{"title": "How to cook rice", "topic": "Cooking Tipatem_[{"title": "HndExample output:
[{"title": "Cara memasak nasi", "topic": "Tips Memasarr[{"title": "CaY Return ONLY valid JSON array, no other text.
```

## 2. Fix Mismatched Caption/Topic

`re```

#```
Given the list of video clips in JS
# fo
```
Given c
   - Topic: clear anicaGi t   - Tth  
# Ca
```
Given clips data in JSON forwitl #co``` (Gior
## 1. Trahe
## 1. Translate to Indonesian

```
Given ng 
```
Given clips datty,## 1. Trane)
```
Given ng 
```
Given cliemeGi p```
GivetyGist## 1ateGiop## 1. shGiven clips data in JTiGi M4. Prtr4. For the kept clip, use the consolidated/improved topic name
5. Filter out: webinar 6. Retue 6. Return the same JSON array with only the translated/updat O
Exam poss5. Filter out: webinar 6. Retue 6. Return the same JSON array
5Example input:
[{"title": "How to cook rice", "topic": "Cooking Tipate[{"title": "HivcoExampl_entertainment, s[{"title": "How to cook rice", "topic": "Cooking Tipatem_[{"title": "HndExample outpu only: webinar filler clips (opening, closing, Q&A, sapa peserta)
7. Output as JSON
```
