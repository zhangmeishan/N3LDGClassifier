#ifndef SRC_NNCNNLabeler_H_
#define SRC_NNCNNLabeler_H_


#include "N3LDG.h"
#include "Driver.h"
#include "Options.h"
#include "Instance.h"
#include "Example.h"
#include "Pipe.h"
#include "Utf.h"

using namespace nr;
using namespace std;

class Classifier {


public:
	unordered_map<string, int> m_word_stats;
	unordered_map<string, int> m_feat_stats;

public:
	Options m_options;

	Driver m_driver;

	Pipe m_pipe;


public:
	Classifier(int memsize);
	virtual ~Classifier();

public:

	int createAlphabet(const vector<Instance>& vecTrainInsts);
	int addTestAlpha(const vector<Instance>& vecInsts);

	void extractFeature(Feature& feat, const Instance* pInstance);

	void convert2Example(const Instance* pInstance, Example& exam);
	void initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams);

public:
	void train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile);
	int predict(const Feature& feature, string& output);
	void test(const string& testFile, const string& outputFile, const string& modelFile);

	void writeModelFile(const string& outputModelFile);
	void loadModelFile(const string& inputModelFile);

};

#endif /* SRC_NNCNNLabeler_H_ */
