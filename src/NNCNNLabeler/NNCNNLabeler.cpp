#include "NNCNNLabeler.h"

#include "Argument_helper.h"

Classifier::Classifier(int memsize) :m_driver(memsize){
	// TODO Auto-generated constructor stub
	srand(0);
}

Classifier::~Classifier() {
	// TODO Auto-generated destructor stub
}

int Classifier::createAlphabet(const vector<Instance>& vecInsts) {
	if (vecInsts.size() == 0){
		std::cout << "training set empty" << std::endl;
		return -1;
	}
	cout << "Creating Alphabet..." << endl;

	int numInstance;

	m_driver._modelparams.labelAlpha.clear();

	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_words;
		const vector<string> &sparse_feats = pInstance->m_sparse_feats;
		const string &label = pInstance->m_label;

		m_driver._modelparams.labelAlpha.from_string(label);
		int words_num = words.size();
		for (int i = 0; i < words_num; i++)
		{
			string curword = normalize_to_lowerwithdigit(words[i]);
			m_word_stats[curword]++;
		}
		int feats_num = sparse_feats.size();
		for(int i = 0; i < feats_num; i++)
		{
			string curfeat = sparse_feats[i];
			m_feat_stats[curfeat]++;
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	cout << "Label num: " << m_driver._modelparams.labelAlpha.size() << endl;
	cout << "Sparse Feature num: " << m_feat_stats.size() << endl;
	cout << "Word num: " << m_word_stats.size() << endl;
	m_driver._modelparams.labelAlpha.set_fixed_flag(true);

	return 0;
}

int Classifier::addTestAlpha(const vector<Instance>& vecInsts) {
	cout << "Adding word Alphabet..." << endl;


	for (int numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_words;
		int curInstSize = words.size();
		for (int i = 0; i < curInstSize; ++i) {
			string curword = normalize_to_lowerwithdigit(words[i]);
			if (!m_options.wordEmbFineTune)m_word_stats[curword]++;
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}

	return 0;
}


void Classifier::extractFeature(Feature& feat, const Instance* pInstance) {
	feat.clear();
	feat.m_words = pInstance->m_words;
	feat.m_sparse_feats = pInstance->m_sparse_feats;
}

void Classifier::convert2Example(const Instance* pInstance, Example& exam) {
	exam.clear();
	const string &orcale = pInstance->m_label;
	int numLabel = m_driver._modelparams.labelAlpha.size();
	vector<dtype> curlabels;
	for (int j = 0; j < numLabel; ++j) {
		string str = m_driver._modelparams.labelAlpha.from_id(j);
		if (str.compare(orcale) == 0)
			curlabels.push_back(1.0);
		else
			curlabels.push_back(0.0);
	}

	exam.m_label = curlabels;
	Feature feat;
	extractFeature(feat, pInstance);
	exam.m_feature = feat;
}

void Classifier::initialExamples(const vector<Instance>& vecInsts, vector<Example>& vecExams) {
	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];
		Example curExam;
		convert2Example(pInstance, curExam);
		vecExams.push_back(curExam);

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}
		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
}

void Classifier::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	static vector<Instance> decodeInstResults;
	static Instance curDecodeInst;
	bool bCurIterBetter = false;

	m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
	if (devFile != "")
		m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	if (testFile != "")
		m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);

	//Ensure that each file in m_options.testFiles exists!
	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
	}

	createAlphabet(trainInsts);
	addTestAlpha(devInsts);
	addTestAlpha(testInsts);
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		addTestAlpha(otherInsts[idx]);
	}

	vector<Example> trainExamples, devExamples, testExamples;

	initialExamples(trainInsts, trainExamples);
	initialExamples(devInsts, devExamples);
	initialExamples(testInsts, testExamples);

	vector<int> otherInstNums(otherInsts.size());
	vector<vector<Example> > otherExamples(otherInsts.size());
	for (int idx = 0; idx < otherInsts.size(); idx++) {
		initialExamples(otherInsts[idx], otherExamples[idx]);
		otherInstNums[idx] = otherExamples[idx].size();
	}

	m_word_stats[unknownkey] = m_options.wordCutOff + 1;
	m_driver._modelparams.wordAlpha.initial(m_word_stats, m_options.wordCutOff);
	m_feat_stats[unknownkey] = m_options.featCutOff + 1;
	m_driver._modelparams.featAlpha.initial(m_feat_stats, m_options.featCutOff);
	if (m_options.wordFile != "") {
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordFile, m_options.wordEmbFineTune);
	}
	else{
		m_driver._modelparams.words.initial(&m_driver._modelparams.wordAlpha, m_options.wordEmbSize, m_options.wordEmbFineTune);
	}

	m_driver._hyperparams.setRequared(m_options);
	m_driver.initial();


	dtype bestDIS = 0;

	int inputSize = trainExamples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	static Metric eval, metric_dev, metric_test;
	static vector<Example> subExamples;
	int devNum = devExamples.size(), testNum = testExamples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;

		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		for (int updateIter = 0; updateIter < batchBlock; updateIter++) {
			subExamples.clear();
			int start_pos = updateIter * m_options.batchSize;
			int end_pos = (updateIter + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;

			for (int idy = start_pos; idy < end_pos; idy++) {
				subExamples.push_back(trainExamples[indexes[idy]]);
			}

			int curUpdateIter = iter * batchBlock + updateIter;
			dtype cost = m_driver.train(subExamples, curUpdateIter);

			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;

			if ((curUpdateIter + 1) % m_options.verboseIter == 0) {
				//m_driver.checkgrad(subExamples, curUpdateIter + 1);
				std::cout << "current: " << updateIter + 1 << ", total block: " << batchBlock << std::endl;
				std::cout << "Cost = " << cost << ", Tag Correct(%) = " << eval.getAccuracy() << std::endl;
			}
			m_driver.updateModel();

		}

		if (devNum > 0) {
			clock_t time_start = clock();
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			for (int idx = 0; idx < devExamples.size(); idx++) {
				string result_label;
				predict(devExamples[idx].m_feature, result_label);

				devInsts[idx].evaluate(result_label, metric_dev);

				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_label);
					decodeInstResults.push_back(curDecodeInst);
				}
			}

			std::cout << "Dev finished. Total time taken is: " << double(clock() - time_start) / CLOCKS_PER_SEC << std::endl;
			std::cout << "dev:" << std::endl;
			metric_dev.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestDIS) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				time_start = clock();
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idx = 0; idx < testExamples.size(); idx++) {
					string result_label;
					predict(testExamples[idx].m_feature, result_label);

					testInsts[idx].evaluate(result_label, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_label);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "Test finished. Total time taken is: " << double(clock() - time_start) / CLOCKS_PER_SEC << std::endl;
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherExamples.size(); idx++) {
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				for (int idy = 0; idy < otherExamples[idx].size(); idy++) {
					string result_label;
					predict(otherExamples[idx][idy].m_feature, result_label);

					otherInsts[idx][idy].evaluate(result_label, metric_test);

					if (bCurIterBetter && !m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
						curDecodeInst.assignLabel(result_label);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestDIS) {
				std::cout << "Exceeds best previous performance of " << bestDIS << ". Saving model file.." << std::endl;
				bestDIS = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}

		}
		// Clear gradients
	}
}

int Classifier::predict(const Feature& feature, string& output) {
	//assert(features.size() == words.size());
	int labelIdx;
	m_driver.predict(feature, labelIdx);
	output = m_driver._modelparams.labelAlpha.from_id(labelIdx, unknownkey);

	if (output == nullkey){
		std::cout << "predict error" << std::endl;
	}
	return 0;
}

void Classifier::test(const string& testFile, const string& outputFile, const string& modelFile) {
	loadModelFile(modelFile);
	m_driver.TestInitial();
	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts);

	vector<Example> testExamples;
	initialExamples(testInsts, testExamples);

	int testNum = testExamples.size();
	vector<Instance> testInstResults;
	Metric metric_test;
	metric_test.reset();
	for (int idx = 0; idx < testExamples.size(); idx++) {
		string result_label;
		predict(testExamples[idx].m_feature, result_label);
		testInsts[idx].evaluate(result_label, metric_test);
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(result_label);
		testInstResults.push_back(curResultInst);
	}
	std::cout << "test:" << std::endl;
	metric_test.print();

	m_pipe.outputAllInstances(outputFile, testInstResults);

}


void Classifier::loadModelFile(const string& inputModelFile) {
	ifstream is(inputModelFile);
	if (is.is_open()) {
		m_driver._hyperparams.loadModel(is);
		m_driver._modelparams.loadModel(is, &m_driver._aligned_mem);
		is.close();
	}
	else
		cout << "load model error" << endl;
}

void Classifier::writeModelFile(const string& outputModelFile) {
	ofstream os(outputModelFile);
	if (os.is_open()) {
		m_driver._hyperparams.saveModel(os);
		m_driver._modelparams.saveModel(os);
		os.close();
		cout << "write model ok. " << endl;
	}
	else
		cout << "open output file error" << endl;
}


int main(int argc, char* argv[]) {

	std::string trainFile = "", devFile = "", testFile = "", modelFile = "", optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	int memsize = 0;
 	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	ah.new_named_int("memsize", "memorySize", "named_int", "This argument decides the size of static memory allocation", memsize);

	ah.process(argc, argv);

	if (memsize < 0)
		memsize = 0;
	Classifier the_classifier(memsize);
	if (bTrain) {
		the_classifier.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		the_classifier.test(testFile, outputFile, modelFile);
	}
	//getchar();
	//test(argv);
	//ah.write_values(std::cout);
}
