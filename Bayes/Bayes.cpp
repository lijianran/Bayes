
#include"Bayes.h"


CNaiveBayes::CNaiveBayes()
{
	vector<string> vec;
	for (int i = 0; i < 6; i++)
	{
		vec.clear();
		for (int j = 0; Docs_list[i][j] != "null"; j++)
		{
			vec.push_back(Docs_list[i][j]);
		}
		list_of_Docs.push_back(vec);
	}
	for (int i = 0; i < sizeof(class_vec) / sizeof(class_vec[0]); i++)
	{
		list_classes.push_back(class_vec[i]);
	}
}

CNaiveBayes::~CNaiveBayes()
{
}

void CNaiveBayes::create_vocab_list()
{
	vector<vector<string>>::iterator it = list_of_Docs.begin();
	int index = 1;
	while (it != list_of_Docs.end())
	{
		vector<string> vec = *it;
		vector<string>::iterator tmp_it = vec.begin();
		while (tmp_it != vec.end())
		{
			if (my_vocab_list[*tmp_it] == 0)
			{
				my_vocab_list[*tmp_it] = index++;
			}
			tmp_it++;
		}
		it++;
	}
}

void CNaiveBayes::set_words_to_vec(int idx)
{
	int len = my_vocab_list.size() + 1;//长度加1，留了一个常数b（ax+b=y）
	return_vec = new int[len]();
	fill(return_vec, return_vec + len, 0);
	for (int i = 0; Docs_list[idx][i] != "null"; i++)
	{
		int pos = my_vocab_list[Docs_list[idx][i]];
		if (pos != 0)
		{
			return_vec[pos] = 1;
		}
	}
}

void CNaiveBayes::print_matrix()
{
	cout << "print the matrix :" << endl;
	int len = my_vocab_list.size();
	for (int i = 0; i < len; i++)
	{
		cout << return_vec[i] << " ";
		if ((i + 1) % 8 == 0 && i != 0)
			cout << endl;
	}
	cout << endl;
}

void CNaiveBayes::get_train_matrix()
{
	train_mat.clear();
	for (int i = 0; i < 6; i++)
	{
		set_words_to_vec(i);
		vector<int> vec(return_vec, return_vec + (my_vocab_list.size() + 1));
		train_mat.push_back(vec);
	}
}

void CNaiveBayes::train_NB0()
{
	int num_train_docs = train_mat.size();
	cout << "num_train_docs = " << num_train_docs << endl;
	int sum = accumulate(list_classes.begin(), list_classes.end(), 0);
	p_abusive = (float)sum / (float)num_train_docs;
	cout << "p_abusive = " << p_abusive << endl;

	p0vect.resize(train_mat[0].size(), 1);
	p1vect.resize(train_mat[0].size(), 1);
	float p0Denom = 2.0;
	float p1Denom = 2.0;

	for (int i = 0; i < list_classes.size(); i++)
	{
		if (list_classes[i] == 1)
		{
			for (int j = 0; j < p1vect.size(); j++)
			{
				p1vect[j] += train_mat[i][j];
				if (train_mat[i][j] == 1)
					p1Denom++;
			}
		}
		else
		{
			for (int j = 0; j < p0vect.size(); j++)
			{
				p0vect[j] += train_mat[i][j];
				if (train_mat[i][j] == 1)
					p0Denom++;
			}
		}
	}

	for (int i = 0; i < p1vect.size(); i++)
	{
		p0vect[i] = log(p0vect[i] / p0Denom);
		p1vect[i] = log(p1vect[i] / p1Denom);
	}

	cout << "print the p0vect values :\n";
	for (int i = 0; i < p0vect.size(); i++)
	{
		cout << setw(11) << p0vect[i];
		if ((i + 1) % 8 == 0 && i != 0)
			cout << endl;
	}

	cout << "\nprint the p1vect values :\n";
	for (int i = 0; i < p1vect.size(); i++)
	{
		cout << setw(11) << p1vect[i];
		if ((i + 1) % 8 == 0 && i != 0)
			cout << endl;
	}
	cout << endl;
}

int CNaiveBayes::classify_NB(string *doc_to_classify)
{
	return_vec = new int[my_vocab_list.size() + 1]();
	for (int i = 0; doc_to_classify[i] != "null"; i++)
	{
		int pos = my_vocab_list[doc_to_classify[i]];
		if (pos != 0)
		{
			return_vec[pos] = 1;
		}
	}
	float p1 = inner_product(p1vect.begin(), p1vect.end(), return_vec, 0) + log(p_abusive);
	float p0 = inner_product(p0vect.begin(), p0vect.end(), return_vec, 0) + log(1 - p_abusive);
	cout << "p1 = " << p1 << endl;
	cout << "p0 = " << p0 << endl;

	if (p1 > p0)
	{
		cout << " p1 > p0:" << endl;
		cout << "The Doc is abusive!" << endl;
		return 1;
	}
	else
	{
		cout << " p1 < p0:" << endl;
		cout << "The Doc is normal!" << endl;
		return 0;
	}
}


