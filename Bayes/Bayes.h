#pragma once
#ifndef BAYES_HEADER
#define BAYES_HEADER
#include<iostream>
#include<vector>
#include<cstring>
#include<map>
#include<algorithm>
#include<numeric>
#include<iomanip> 
using namespace std;
static string Docs_list[6][10] = {//�ĵ��б�
{ "my","dog","has","flea","problems","help","please","null" },
{ "maybe","not","take","him","to","dog","park","stupid","null" },
{ "my","dalmation","is","so","cute","I","love","him","null" },
{ "stop","posting","stupid","worthless","garbage","null" },
{ "mr","licks","ate","my","steak","how","to","stop","him","null" },
{ "quit","buying","worthless","dog","food","stupid","null" }
};
static int class_vec[6] = { 0,1,0,1,0,1 };//��Ӧ�ĵ������1�����������ĵ���0���������ĵ�

class CNaiveBayes      //���ر�Ҷ˹��
{
private:
	vector<vector<string>> list_of_Docs;   //�ĵ�ת������б�
	vector<int> list_classes;   //�ĵ����
	map<string, int>  my_vocab_list;    //�ĵ�ȥ�غ�ĵ��ʣ���ͼ�洢
	int *return_vec;   //��Ŀ������ָ�봴����
	vector<vector<int>> train_mat;    //��ά����ѵ���Ĳ��Ծ���

	vector<float> p0vect;   //ÿ�����ʳ����������ĵ��еĸ���
	vector<float> p1vect;   //ÿ�����ʳ������������ĵ��еĸ���
	float p_abusive;   //6���ĵ����������ĵ��ĸ��ʣ�1/2
public:
	CNaiveBayes();
	~CNaiveBayes();

public:
	void create_vocab_list();  //�����ʻ����
	void set_words_to_vec(int idx);   //ȥ���ĵ��еĴʻ㣬ת��0/1����
	void print_matrix();   //��ӡ����
	void get_train_matrix();    //��÷�������ѵ������
	void train_NB0();    //ѵ������������
	int classify_NB(string *doc_to_classify);   //���ر�Ҷ˹���Ժ���

};

#endif // !BAYES_HEADER

