#include "NNMLCNNLabeler.h"

#include <chrono>
#include "Argument_helper.h"

Classifier::Classifier(): m_driver(){
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


		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;

	cout << "Label num: " << m_driver._modelparams.labelAlpha.size() << endl;
	cout << "Sparse Feature num: " << m_feat_stats.size() << endl;
	cout << "Word num: " << m_word_stats.size() << endl;
	m_driver._modelparams.labelAlpha.set_fixed_flag(true);

	return 0;
}

int Classifier::addTestAlpha(const vector<Instance>& vecInsts) {
	cout << "Adding word Alphabet..." << endl;

	int numInstance;
	for (numInstance = 0; numInstance < vecInsts.size(); numInstance++) {
		const Instance *pInstance = &vecInsts[numInstance];

		const vector<string> &words = pInstance->m_words;
		int curInstSize = words.size();
		for (int i = 0; i < curInstSize; ++i) {
			string curword = normalize_to_lowerwithdigit(words[i]);
			if (!m_options.wordEmbFineTune)m_word_stats[curword]++;
		}

		if ((numInstance + 1) % m_options.verboseIter == 0) {
			cout << numInstance + 1 << " ";
			if ((numInstance + 1) % (40 * m_options.verboseIter) == 0)
				cout << std::endl;
			cout.flush();
		}

		if (m_options.maxInstance > 0 && numInstance == m_options.maxInstance)
			break;
	}
	cout << numInstance << " " << endl;

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
	cout << numInstance << " " << endl;
}

void Classifier::train(const string& trainFile, const string& devFile, const string& testFile, const string& modelFile, const string& optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	vector<Instance> trainInsts, devInsts, testInsts;
	vector<Instance> decodeInstResults;
	Instance curDecodeInst;
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


	dtype bestFmeasure = 0;

	int inputSize = trainExamples.size();

	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;

	srand(0);
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);

	Metric eval, metric_dev, metric_test;
	vector<Example> subExamples;
	vector<string> result_labels;
	int devNum = devExamples.size(), testNum = testExamples.size();
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter << std::endl;
		bool bEvaluate = false;
		if (m_options.batchSize == 1) {
			auto t_start_train = std::chrono::high_resolution_clock::now();
			eval.reset();
			bEvaluate = true;
			random_shuffle(indexes.begin(), indexes.end());
			std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;
			for (int idy = 0; idy < inputSize; idy++) {
				subExamples.clear();
				subExamples.push_back(trainExamples[indexes[idy]]);
				double cost = m_driver.train(subExamples);
				eval.overall_label_count += m_driver._eval.overall_label_count;
				eval.correct_label_count += m_driver._eval.correct_label_count;

				if ((idy + 1) % (m_options.verboseIter) == 0) {
					auto t_end_train = std::chrono::high_resolution_clock::now();
					std::cout << "current: " << idy + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
						<< ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
				}
				m_driver.checkgrad(subExamples, iter * inputSize + idy);
				m_driver.updateModel();
			}
			{
				auto t_end_train = std::chrono::high_resolution_clock::now();
				std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
					<< ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
			}
		}
		else {
			eval.reset();
			auto t_start_train = std::chrono::high_resolution_clock::now();
			bEvaluate = true;
			for (int idk = 0; idk < (inputSize + m_options.batchSize - 1) / m_options.batchSize; idk++) {
				random_shuffle(indexes.begin(), indexes.end());
				subExamples.clear();
				for (int idy = 0; idy < m_options.batchSize; idy++) {
					subExamples.push_back(trainExamples[indexes[idy]]);
				}
				double cost = m_driver.train(subExamples);

				eval.overall_label_count += m_driver._eval.overall_label_count;
				eval.correct_label_count += m_driver._eval.correct_label_count;

				if ((idk + 1) % (m_options.verboseIter) == 0) {
					auto t_end_train = std::chrono::high_resolution_clock::now();
					std::cout << "current: " << idk + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
						<< ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
				}

				m_driver.updateModel();
			}

			{
				auto t_end_train = std::chrono::high_resolution_clock::now();
				std::cout << "current: " << iter + 1 << ", Correct(%) = " << eval.getAccuracy()
					<< ", time = " << std::chrono::duration<double>(t_end_train - t_start_train).count() << std::endl;
			}
		}

		if (bEvaluate && devNum > 0) {
			auto t_start_dev = std::chrono::high_resolution_clock::now();
			std::cout << "Dev start." << std::endl;
			bCurIterBetter = false;
			if (!m_options.outBest.empty())
				decodeInstResults.clear();
			metric_dev.reset();
			predict(devExamples, result_labels);
			for (int idx = 0; idx < devExamples.size(); idx++) {
				devInsts[idx].evaluate(result_labels[idx], metric_dev);
				if (!m_options.outBest.empty()) {
					curDecodeInst.copyValuesFrom(devInsts[idx]);
					curDecodeInst.assignLabel(result_labels[idx]);
					decodeInstResults.push_back(curDecodeInst);
				}
			}
			auto t_end_dev = std::chrono::high_resolution_clock::now();
			std::cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(t_end_dev - t_start_dev).count() << std::endl;
			std::cout << "dev:" << std::endl;
			metric_dev.print();

			if (!m_options.outBest.empty() && metric_dev.getAccuracy() > bestFmeasure) {
				m_pipe.outputAllInstances(devFile + m_options.outBest, decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				auto t_start_test = std::chrono::high_resolution_clock::now();
				std::cout << "Test start." << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				predict(testExamples, result_labels);
				for (int idx = 0; idx < testInsts.size(); idx++) {
					testInsts[idx].evaluate(result_labels[idx], metric_test);
					if (!m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(testInsts[idx]);
						curDecodeInst.assignLabel(result_labels[idx]);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				auto t_end_test = std::chrono::high_resolution_clock::now();
				std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_test - t_start_test).count() << std::endl;
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest, decodeInstResults);
				}
			}

			for (int idx = 0; idx < otherInsts.size(); idx++) {
				auto t_start_other = std::chrono::high_resolution_clock::now();
				std::cout << "processing " << m_options.testFiles[idx] << std::endl;
				if (!m_options.outBest.empty())
					decodeInstResults.clear();
				metric_test.reset();
				predict(otherExamples[idx], result_labels);
				for (int idy = 0; idy < otherInsts[idx].size(); idy++) {
					otherInsts[idx][idy].evaluate(result_labels[idy], metric_test);
					if (!m_options.outBest.empty()) {
						curDecodeInst.copyValuesFrom(otherInsts[idx][idy]);
						curDecodeInst.assignLabel(result_labels[idy]);
						decodeInstResults.push_back(curDecodeInst);
					}
				}
				auto t_end_other = std::chrono::high_resolution_clock::now();
				std::cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_other - t_start_other).count() << std::endl;
				std::cout << "test:" << std::endl;
				metric_test.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(m_options.testFiles[idx] + m_options.outBest, decodeInstResults);
				}
			}

			if (m_options.saveIntermediate && metric_dev.getAccuracy() > bestFmeasure) {
				std::cout << "Exceeds best previous DIS of " << bestFmeasure << ". Saving model file.." << std::endl;
				bestFmeasure = metric_dev.getAccuracy();
				writeModelFile(modelFile);
			}
		}
	}
}

void Classifier::predict(const vector<Example>& inputs, vector<string>& outputs) {
	//assert(features.size() == words.size());
	int sentNum = inputs.size();
	if (sentNum <= 0) return;
	outputs.resize(sentNum);

	vector<Feature> batch_sentences;
	vector<int> batch_labelIdxs;
	vector<string> batch_outputs;
	int processed_count = 0;
	for (int idx = 0; idx < sentNum; idx++) {
		batch_sentences.push_back(inputs[idx].m_feature);
		if (batch_sentences.size() == m_options.batchSize || idx == sentNum - 1) {
			m_driver.predict(batch_sentences, batch_labelIdxs);
			batch_sentences.clear();
			for (int idy = 0; idy < batch_labelIdxs.size(); idy++) {
				outputs[processed_count] = m_driver._modelparams.labelAlpha.from_id(batch_labelIdxs[idy], unknownkey);
				processed_count++;
			}
		}
	}

	if (processed_count != sentNum) {
		std::cout << "decoded number not match" << std::endl;
	}
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

	vector<string> resulted_labels;
	predict(testExamples, resulted_labels);
	for (int idx = 0; idx < testExamples.size(); idx++) {
		testInsts[idx].evaluate(resulted_labels[idx], metric_test);
		Instance curResultInst;
		curResultInst.copyValuesFrom(testInsts[idx]);
		curResultInst.assignLabel(resulted_labels[idx]);
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
		m_driver._modelparams.loadModel(is);
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
 	dsr::Argument_helper ah;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);

	ah.process(argc, argv);

	Classifier the_classifier;
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
