# LSML2 Final Project

### Design principles
I would like to create simple question answering bot, using the Cornell movie dialogs corpus, and using Seq2Seq with attention model. Trained model consisting of Encoder and Decoder models will be trained on dataset, model weights will be saved according with word2id dictionaries.
### Model description
Input: string variable length 
Output: string variable length Model will be the sequence-to-sequence (seq2seq) with attention, based on paper https://arxiv.org/abs/1409.3215 
The encoder model consists of an embedding layer and lstm, the task of which is to transform a question encoded through the Word2id dictionary into a vector representation. Then, based on the obtained vector representation, the Decoder model consisting of an embedding layer, a GRUcell, an attention mechanism and an output fully connected layer turns this into a answer, which is decoded through the id2word dictionary. 
Embedding size was choosen as 256 
Hidden size was equal 512 
As loss function nn.CrossEntropyLoss() was used. 
I couldn't come up with any metric to evaluate the generated answer, so I just printed out the answer and saw if it made any sense, when I liked the response, I trained the network a little more and then stopped. I know its a weird approach, but in short time I didnt find anything better ;) And actually it work well.
For the deployment I choose async architecture based on Flask, Celery and Redis as query DB. 
Service will be divided into parts:
1. backend - which will perform the model inference according with the text processing utils. (on port 8080) 
2. frontend - simple streamlit app with text input field and button (on port 8501) 
3. Celery worker. 
4. Redis DB(port 6379) Each part should be placed in their own docker container and all containers will be linked through docker-compose.
### Running instructions
using terminal type: docker-compose up -d --build docker-compose up open the browser network URL: http://192.168.0.137:8501 or http://localhost:8501 you can type your question in text input field and press Enter or click Ask button, and service will give you answer.
### Dataset description
Dataset Cornell movie dialogs corpus consists of 220,579 conversational exchanges between 10,292 pairs of movie characters, involves 9,035 characters from 617 movies, and in total 304,713 utterances.
### Model training code
Model training code provided in folder Train_model
