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
static string Docs_list[6][10] = {//文档列表
{ "my","dog","has","flea","problems","help","please","null" },
{ "maybe","not","take","him","to","dog","park","stupid","null" },
{ "my","dalmation","is","so","cute","I","love","him","null" },
{ "stop","posting","stupid","worthless","garbage","null" },
{ "mr","licks","ate","my","steak","how","to","stop","him","null" },
{ "quit","buying","worthless","dog","food","stupid","null" }
};
static int class_vec[6] = { 0,1,0,1,0,1 };//对应文档的类别，1代表侮辱性文档，0代表正常文档

class CNaiveBayes      //朴素贝叶斯类
{
private:
	vector<vector<string>> list_of_Docs;   //文档转化后的列表
	vector<int> list_classes;   //文档类别
	map<string, int>  my_vocab_list;    //文档去重后的单词，用图存储
	int *return_vec;   //数目矩阵，用指针创建的
	vector<vector<int>> train_mat;    //二维矩阵，训练的测试矩阵

	vector<float> p0vect;   //每个单词出现在正常文档中的概率
	vector<float> p1vect;   //每个单词出现在侮辱性文档中的概率
	float p_abusive;   //6个文档中侮辱性文档的概率，1/2
public:
	CNaiveBayes();
	~CNaiveBayes();

public:
	void create_vocab_list();  //创建词汇矩阵
	void set_words_to_vec(int idx);   //去重文档中的词汇，转成0/1矩阵
	void print_matrix();   //打印矩阵
	void get_train_matrix();    //获得分类器的训练矩阵
	void train_NB0();    //训练分类器函数
	int classify_NB(string *doc_to_classify);   //朴素贝叶斯测试函数

};

#endif // !BAYES_HEADER

