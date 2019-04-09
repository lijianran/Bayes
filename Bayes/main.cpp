
#include"Bayes.h"

int main()
{
	CNaiveBayes nb;
	nb.create_vocab_list();
	nb.get_train_matrix();
	nb.train_NB0();
	string doc1_to_classify[] = { "love","my","dalmation","null" };
	nb.classify_NB(doc1_to_classify);

	return 0;
}