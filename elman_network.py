import torch
import torch.nn as nn
import torch.optim as optim
import spacy


###### DATA PROCESSING STARTS #########


def load_corpus():
	#this loads the data from sample_corpus.txt
	with open('sample_corpus.txt','r',encoding="utf8") as f:
		corpus = f.read().replace('\n',' ')
	return corpus

def remove_infrequent_words(sents):
	word_counts = {}
	for s in sents:
		for w in s:
			if w in word_counts:
				word_counts[w] += 1
			else:
				word_counts[w] = 1

	threshold = 2
	filtered_sents = []
	for s in sents:
		new_s = []
		for w in s:
			if word_counts[w] < threshold:
				new_s.append('<UNKNOWN>')
			else:
				new_s.append(w)
		filtered_sents.append(new_s)
	return filtered_sents

def segment_and_tokenize(corpus):
	#make sure to run: 
	# pip install -U pip setuptools wheel
	# pip install -U spacy
	# python -m spacy download en_core_web_sm
	#in the command line before using this!

	#corpus is assumed to be a string, containing the entire corpus
	nlp = spacy.load('en_core_web_sm')
	tokens = nlp(corpus)
	sents = [[t.text for t in s] for s in tokens.sents if len([t.text for t in s])>1]
	sents = remove_infrequent_words(sents)
	sents = [['<START>']+s+['<END>'] for s in sents]
	return sents

def make_word_to_ix(sents):
	word_to_ix = {}
	num_unique_words = 0
	for sent in sents:
		for word in sent:
			if word not in word_to_ix:
				word_to_ix[word] = num_unique_words
				num_unique_words += 1


	return word_to_ix

def sent_to_onehot_vecs(sent,word_to_ix):
	#note: this is not how you would do this in practice! 

	vecs = []
	for i in range(len(sent)):
		word = sent[i]
		word_index = word_to_ix[word]

		vec = torch.zeros(len(word_to_ix), dtype=torch.float32,requires_grad=False)
		vec[word_index] = 1
		vecs.append(vec)

	return vecs

def vectorize_sents(sents,word_to_ix):
	one_hot_vecs = []
	for s in sents:
		one_hot_vecs.append(sent_to_onehot_vecs(s,word_to_ix))
	return one_hot_vecs

def get_data():
	corpus = load_corpus()
	sents = segment_and_tokenize(corpus)
	word_to_ix = make_word_to_ix(sents)

	vectorized_sents = vectorize_sents(sents,word_to_ix)

	vocab_size = len(word_to_ix)

	return vectorized_sents, vocab_size




###### DATA PROCESSING ENDS #########




###### RNN DEFINITION STARTS #########

class ElmanNetwork(nn.Module):

	def __init__(self, embedding_dim, vocab_size, hidden_state_dim):
		super().__init__()

		self.W_e = nn.Parameter(torch.rand((embedding_dim, vocab_size )))
		self.W_x = nn.Parameter(torch.rand((hidden_state_dim, embedding_dim )))
		self.W_h = nn.Parameter(torch.rand((hidden_state_dim, hidden_state_dim )))
		self.W_p = nn.Parameter(torch.rand((vocab_size, hidden_state_dim )))
		self.b = nn.Parameter(torch.rand((hidden_state_dim )))

	def initialize_hidden_state(self,shape):
		return torch.zeros(shape,dtype=torch.float32,requires_grad=False)


	def elman_unit(self,word_embedding,h_previous):
		return torch.sigmoid(torch.matmul(self.W_x,word_embedding)+torch.matmul(self.W_h,h_previous)+self.b)

	def embed_word(self,word):
		#word is a one-hot vector
		return torch.matmul(self.W_e,word)


	def single_layer_perceptron(self,h):
		s = torch.matmul(self.W_p,h)
		softmax = nn.Softmax(dim=0)
		return softmax(s)


	def forward(self,sent):
		h_previous = self.initialize_hidden_state(self.W_h.size(1))

		predictions = []
		for i in range(len(sent)-1):
			current_word = sent[i]

			current_word_embedding = self.embed_word(current_word)

			h_current = self.elman_unit(current_word_embedding,h_previous)

			prediction = self.single_layer_perceptron(h_current)
			predictions.append(prediction)

			h_previous = h_current

		return predictions


###### RNN DEFINITION ENDS #########


#### LOSS FUNCTION BEGINS #######

def word_loss(word_probs, word):
	#outcome is a one-hot vector
	prob_of_word = torch.dot(word_probs,word)
	return -1*torch.log(prob_of_word)

def sent_loss(predictions, sent):
	L = torch.tensor(0,dtype=torch.float32)

	num_words = len(predictions)

	for i in range(num_words):
		word_probs = predictions[i]
		observed_word = sent[i+1]
		L = L + word_loss(word_probs,observed_word)

	return L / num_words


##### LOSS FUNCTION ENDS #######


def train():
	
	vectorized_sents, vocab_size = get_data()
	
	num_epochs = 100

	hidden_state_dim = 20
	embedding_dim = 20
	learning_rate = 0.001

	elman_network = ElmanNetwork(embedding_dim,vocab_size,hidden_state_dim)

	optimizer = optim.SGD(elman_network.parameters(), lr=learning_rate)

	for i in range(num_epochs):

		total_loss = 0
		for s in vectorized_sents:
			optimizer.zero_grad()
			predictions = elman_network(s)
			loss = sent_loss(predictions,s)
			total_loss += loss.detach().numpy()

			loss.backward() 
			optimizer.step() 
		print(total_loss / len(vectorized_sents))


if __name__=='__main__':
	train()



