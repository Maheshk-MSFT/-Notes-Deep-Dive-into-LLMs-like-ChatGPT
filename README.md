# -Notes-Deep-Dive-into-LLMs-like-ChatGPT
Self reference - Deep Dive into LLMs like ChatGPT - Andrej video https://www.youtube.com/watch?v=7xTGNNLPyMI

  ![image](https://github.com/user-attachments/assets/f3a1a429-b8a6-46e1-8fee-d22efda57202)

  **Fineweb pretraining flow: **
(a) Common Crawl 2.6 billion webpages ->  FineWeb 44 TB Dataset- huggingface -> block list -> extract raw html - only text -> language filter (only english) -> PI removal -> copyrighted info -> terabytes of text -> neural network (0's and 1's) -> bytes (unique symbols) -> tokenisation -> prediction -> { base model }

(b) Base Model -> Post-Training -> dataset conversations->  helpful, truthful and harmless response -> 

(c) SFT model (Supervised Finetuning) -> Post training -> Reinforcement Learning 

(d) RLHF (Reinforcement learning from Human Feedback) - this is for unverfiable domains say joke etc where we need humans to score

(e) Multimodel - Everything goes into token  

_probability distribution
_SFT models vs deep thinking models (research/thinking models) - mostly paid
_all the thinking models are in preview - all using Reinforcement model

References: 
1) lmareana.ai
2) AI news
3) together.ai
4) Hyperbolic - basemodels
5) LMStudio 
------------------------------------------------------------------------------------------------------------------------------------------------------
  ![image](https://github.com/user-attachments/assets/f84300cc-09d2-4be2-919e-5d04585b7c99)
  
  ![image](https://github.com/user-attachments/assets/b3828e3b-1047-4bcc-9e58-6c741bc8b7f8)

  ![image](https://github.com/user-attachments/assets/9ad89e13-662d-4b74-b88d-984db175ec83)

  ![image](https://github.com/user-attachments/assets/d0543b00-8cc9-4ce7-aac7-4921969acc01)

  ![image](https://github.com/user-attachments/assets/2e141c46-a0dc-4209-9336-3446409a840c)

  ![image](https://github.com/user-attachments/assets/14839d72-bd39-4ed1-b03d-e2c8185bc7ce)

  ![image](https://github.com/user-attachments/assets/303e3a60-9b3d-44d1-b9ef-8ccbdfcb98f2)

  ![image](https://github.com/user-attachments/assets/a833958a-5c50-458e-86de-8b87b8062b23)
  
![image](https://github.com/user-attachments/assets/bdbd339a-c463-4a65-8ac6-43c28aaf07fe)

![image](https://github.com/user-attachments/assets/b10f66f3-4426-49f2-9d2b-9a7f0709ea5f)

![image](https://github.com/user-attachments/assets/cfa1e5ab-fca7-4355-b080-d47ff1aa0c9e)

![image](https://github.com/user-attachments/assets/68c5671d-19f2-4ed4-842e-73bf83eb3f34)

![image](https://github.com/user-attachments/assets/9c2ff9ea-3aa2-4955-bde4-6ca3b23b6f25)

![image](https://github.com/user-attachments/assets/8747d720-0e98-49f7-9cd1-325e91ca220e)

![image](https://github.com/user-attachments/assets/7f30826b-1c8a-4513-b85d-903b50f635f0)

![image](https://github.com/user-attachments/assets/9ac228b0-1e65-4262-89bb-ef8129914751)

![image](https://github.com/user-attachments/assets/731af686-9859-48b8-9413-13538e5e3ffe)

![image](https://github.com/user-attachments/assets/96e9b754-eaac-42c4-a6ec-77fc873a3cca)

![image](https://github.com/user-attachments/assets/87bc3bd2-d803-430b-8f4e-2f3fb33751d2)

![image](https://github.com/user-attachments/assets/7b5d18d7-f039-4e61-bb52-65ad815160b1)

![image](https://github.com/user-attachments/assets/4cc8fcdc-4793-4e92-8745-bee133e16a43)

![image](https://github.com/user-attachments/assets/3b3bdd2c-8c61-4503-ab2e-1b75926ca99f)

![image](https://github.com/user-attachments/assets/98c19af1-f1e7-4978-8cef-3ab4715202b3)

![image](https://github.com/user-attachments/assets/46483778-3b54-47cf-bf9b-e908502ae7aa)

![image](https://github.com/user-attachments/assets/ad1c66ed-f766-4a2c-9a7d-868638413420)

![image](https://github.com/user-attachments/assets/f756ea31-e699-402e-9f00-d1a592cf04cb)

![image](https://github.com/user-attachments/assets/a18851a7-eb54-4406-80cd-04dcbe010f3c)

![image](https://github.com/user-attachments/assets/a17f71bb-c3fc-49e5-8b64-37c375d19ff2)

![image](https://github.com/user-attachments/assets/ccd87898-e483-41d2-892f-4c14e894ed87)

![image](https://github.com/user-attachments/assets/0ea70555-e221-4c33-834a-3713d6bce543)

![image](https://github.com/user-attachments/assets/01660de8-cc59-453e-b1aa-2e25e45c9d75)

![image](https://github.com/user-attachments/assets/d1c0261b-d799-4b83-b16e-b15116aa1ee1)

![image](https://github.com/user-attachments/assets/9c5b06cf-7581-43ae-836c-b3de1b621676)

![image](https://github.com/user-attachments/assets/5a73a55d-f69c-4ecf-9994-c447dc2f948c)

![image](https://github.com/user-attachments/assets/1f0d4fb7-c579-4e16-bd05-1f7cf632c635)

```
**Chapters**
00:00:00 introduction
00:01:00 pretraining data (internet)
00:07:47 tokenization
00:14:27 neural network I/O
00:20:11 neural network internals
00:26:01 inference
00:31:09 GPT-2: training and inference
00:42:52 Llama 3.1 base model inference
00:59:23 pretraining to post-training
01:01:06 post-training data (conversations)
01:20:32 hallucinations, tool use, knowledge/working memory
01:41:46 knowledge of self
01:46:56 models need tokens to think
02:01:11 tokenization revisited: models struggle with spelling
02:04:53 jagged intelligence
02:07:28 supervised finetuning to reinforcement learning
02:14:42 reinforcement learning
02:27:47 DeepSeek-R1
02:42:07 AlphaGo
02:48:26 reinforcement learning from human feedback (RLHF)
03:09:39 preview of things to come
03:15:15 keeping track of LLMs
03:18:34 where to find LLMs
03:21:46 grand summary
```

```
Links
ChatGPT https://chatgpt.com/
FineWeb (pretraining dataset): https://huggingface.co/spaces/Hugging...
Tiktokenizer: https://tiktokenizer.vercel.app/
Transformer Neural Net 3D visualizer: https://bbycroft.net/llm
llm.c Let's Reproduce GPT-2 https://github.com/karpathy/llm.c/dis...
Llama 3 paper from Meta: https://arxiv.org/abs/2407.21783
Hyperbolic, for inference of base model: https://app.hyperbolic.xyz/
InstructGPT paper on SFT: https://arxiv.org/abs/2203.02155
HuggingFace inference playground: https://huggingface.co/spaces/hugging...
DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
TogetherAI Playground for open model inference: https://api.together.xyz/playground
AlphaGo paper (PDF): https://discovery.ucl.ac.uk/id/eprint...
AlphaGo Move 37 video:    • Lee Sedol vs AlphaGo  Move 37 reactions an...  
LM Arena for model rankings: https://lmarena.ai/
AI News Newsletter: https://buttondown.com/ainews
LMStudio for local inference https://lmstudio.ai/

```
