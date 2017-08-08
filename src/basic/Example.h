#ifndef _EXAMPLE_H_
#define _EXAMPLE_H_

#include <iostream>
#include <vector>

using namespace std;

class Feature {
  public:
    vector<string> m_words;
    vector<string> m_sparse_feats;
  public:
    void clear() {
        m_words.clear();
        m_sparse_feats.clear();
    }
};

class Example {
  public:
    Feature m_feature;
    vector<dtype> m_label;

  public:
    void clear() {
        m_feature.clear();
        m_label.clear();
    }
};

#endif /*_EXAMPLE_H_*/