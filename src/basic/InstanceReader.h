#ifndef _CONLL_READER_
#define _CONLL_READER_

#include "Reader.h"
#include "N3LDG.h"
#include <sstream>

using namespace std;
/*
 this class reads conll-format data (10 columns, no srl-info)
 */
class InstanceReader : public Reader {
public:
	InstanceReader() {
	}
	~InstanceReader() {
	}

	Instance *getNext() {
		m_instance.clear();
		string strLine1, strLine2;
		if (!my_getline(m_inf, strLine1))
			return NULL;
		if (!my_getline(m_inf, strLine2))
			return NULL;
		if (strLine1.empty())
			return NULL;


		vector<string> vecInfo;
		split_bychars(strLine1, vecInfo, "\t");
		m_instance.m_label = vecInfo[0];

		split_bychar(vecInfo[1], m_instance.m_words, ' ');
		split_bychar(strLine2, m_instance.m_sparse_feats, ' ');
		return &m_instance;
	}
};

#endif

